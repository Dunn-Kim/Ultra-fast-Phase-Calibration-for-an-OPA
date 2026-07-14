# Gray-box 모델오차 보정 + 간격 공동최적화 실증
#
# 아이디어: AF(간섭항)는 점광원 가정상 정확 → 계통오차는 EF 포락선에 몰림.
#   EF에만 보정항 추가:  EF_corr(u) = EF_sinc(u) · exp(g(u; θ)),  g = 짝수 저차 다항
#   g(u) = a1·u² + a2·u⁴ + a3·u⁶  (g(0)=0 → 주엽 정규화 보존, smooth → 과적합 방지)
#
# 핵심 규율 (Kennedy-O'Hagan model-discrepancy confounding 회피):
#   θ(보정)  ← 관측데이터(단채널 far-field)로만 회귀 학습.  PSLL 목적과 분리.
#   x(간격)  ← 보정된 모델로만 PSLL 최적화.
#   둘을 같은 PSLL 목적에 공동최적화하면 θ가 부엽을 눌러 PSLL을 가짜로 낮춤(퇴화) — 실증 포함.
#
# 실측 Lumerical 부재 → "진짜 관측" 대역모델(Gaussian 모드 + obliquity)로 method 검증:
#   TRUE(관측 대용)  EF_true = exp(−(π·0.5·u/λ)²) · √(1−u²)   [Gaussian w0=0.5µm + cosθ]
#   NAIVE(현행)      EF_naive = sinc(w·u/λ), w=1
#   보정 목표: NAIVE·exp(g)가 TRUE에 근접하도록 θ를 소수 관측각에서 회귀.
#
# 실행: venv/bin/python experiment_graybox.py
import json
import math
import os
import time
import types

import numpy as np
import torch as th

from benchmark_common import check_constraints, evaluate_layout, gaps_to_positions, make_model
from config import JointConfig
from main_design_multiangle import run_multiangle_single, snap

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64
LAM, W = 1.55, 1.0


def ef_naive(u):
    return th.sinc(W * u / LAM)


def ef_true(u):
    # 관측 대용 진짜 소자인자: Gaussian 근접장(w0=0.5µm) FT × RS-I 경사인자
    return th.exp(-(math.pi * 0.5 * u / LAM) ** 2) * th.sqrt((1 - u ** 2).clamp(min=0))


def ef_corr_factory(theta):
    # EF_corr(u) = sinc · exp(a1 u² + a2 u⁴ + a3 u⁶)
    a1, a2, a3 = theta
    def ef(u):
        g = a1 * u ** 2 + a2 * u ** 4 + a3 * u ** 6
        return th.sinc(W * u / LAM) * th.exp(g)
    return ef


def calibrate(n_obs=15, obs_span=60.0, noise_db=0.1, seed=0):
    # 관측: 소수 각도에서 단채널 far-field 진폭(=EF_true) 측정 → log(EF_true/EF_naive) 회귀
    th.manual_seed(seed)
    ang = th.linspace(-obs_span, obs_span, n_obs, dtype=DT)
    u = th.sin(th.deg2rad(ang))
    with th.no_grad():
        target = th.log(ef_true(u).clamp(min=1e-6) / ef_naive(u).clamp(min=1e-6))
        target = target + (noise_db * math.log(10) / 10) * th.randn(n_obs, dtype=DT)  # 관측잡음
    # 선형 최소자승 (짝수 기저) — 볼록, 닫힌형
    B = th.stack([u ** 2, u ** 4, u ** 6], dim=1)
    theta = th.linalg.lstsq(B, target).solution
    with th.no_grad():
        resid = (B @ theta - target)
        rms = (resid ** 2).mean().sqrt().item() * 10 / math.log(10)  # dB
    return theta.tolist(), rms


def patch(model, ef_fn):
    m = make_model(model.cfg)
    m.element_factor_amp = types.MethodType(lambda self, u: ef_fn(u), m)
    return m


def design_on(ef_fn, cfg, seeds=8):
    # 주어진 EF로 간격 최적화 (여러 시드 → best). run_multiangle_single이 model.element_factor_amp 경유
    best_v, best_s, best_m = math.inf, None, None
    for r in range(seeds):
        m = patch(make_model(cfg), ef_fn)
        # run_multiangle_single은 내부에서 OPAModel(cfg) 새로 만드므로 monkeypatch 주입 위해 직접 루프 대신 래핑
        res = _run_with_model(m, cfg, cfg.seed + r)
        if res['worst'] < best_v:
            best_v, best_s, best_m = res['worst'], res['s'], m
    return best_s, best_m, best_v


def _run_with_model(model, cfg, seed):
    # run_multiangle_single의 핵심 루프를 주입된 model로 실행
    from losses import beta_schedule, multi_angle_worst_psll
    from metrics import hard_psll_db
    th.manual_seed(seed)
    n_gap = cfg.line_N - 1
    s = th.full((n_gap,), model.s_init_uniform(), dtype=DT) + cfg.s_init_std * th.randn(n_gap, dtype=DT)
    s.requires_grad_(True)
    dphi = th.zeros(len(cfg.design_angles_deg), cfg.line_N, dtype=DT, requires_grad=True)
    opt_s = th.optim.Adam([s], lr=cfg.lr_spacing)
    opt_p = th.optim.Adam([dphi], lr=cfg.lr_phase)
    for t in range(cfg.epochs_joint):
        beta = beta_schedule(t, cfg.epochs_joint, cfg.beta_start, cfg.beta_end)
        opt_s.zero_grad(); opt_p.zero_grad()
        loss = multi_angle_worst_psll(model, s, beta, cfg.design_angles_deg, cfg.angle_agg_gamma, dphi=dphi)
        loss.backward(); opt_s.step(); opt_p.step()
    with th.no_grad():
        x = model.positions(s.detach())
        worst = max(hard_psll_db(model, x,
                    model.steering_phase(x, math.sin(math.radians(a))) + dphi[i],
                    u=model.u_val, u0=math.sin(math.radians(a)))
                    for i, a in enumerate(cfg.design_angles_deg))
    return {'s': s.detach(), 'worst': worst}


def degeneracy_demo(cfg, seed=1):
    # 함정 실증: θ와 s를 같은 PSLL 목적에 공동최적화 → θ가 자유도로 PSLL 가짜 개선
    from losses import beta_schedule, soft_psll
    th.manual_seed(seed)
    m = make_model(cfg)
    n_gap = cfg.line_N - 1
    s = (th.full((n_gap,), m.s_init_uniform(), dtype=DT) + 0.5 * th.randn(n_gap, dtype=DT)).requires_grad_(True)
    theta = th.zeros(3, dtype=DT, requires_grad=True)   # 보정변수 — 앵커 없이 함께 최적화
    US = [math.sin(math.radians(a)) for a in cfg.design_angles_deg]
    opt = th.optim.Adam([s, theta], lr=1e-2)
    for t in range(600):
        beta = beta_schedule(t, 600, 0.2, 6.0)
        m.element_factor_amp = types.MethodType(
            lambda self, u, th_=theta: th.sinc(W * u / LAM) * th.exp(th_[0]*u**2 + th_[1]*u**4 + th_[2]*u**6), m)
        opt.zero_grad()
        x = m.positions(s)
        ls = th.stack([soft_psll(m, x, m.steering_phase(x, u0), beta, u0=u0) for u0 in US])
        loss = th.logsumexp(2.0 * ls, 0) / 2.0
        loss.backward(); opt.step()
    return loss.item(), theta.detach().tolist()


def main():
    t0 = time.time()
    cfg = JointConfig(design_angles_deg=tuple(float(a) for a in range(-30, 31, 5)),
                      beta_end=6.0, angle_agg_gamma=2.0, s_init_std=0.5,
                      epochs_joint=800)
    report = {}

    # 1) 보정 파라미터 회귀 (관측 → θ)
    theta, rms = calibrate()
    ef_corr = ef_corr_factory(theta)
    report['calibration'] = {'theta': [round(t, 4) for t in theta], 'fit_rms_db': round(rms, 3),
                             'note': 'log(EF_true/EF_naive)를 15관측각에서 짝수3차 회귀'}
    # 보정 후 EF 재현도
    uc = th.sin(th.deg2rad(th.tensor([0., 10, 20, 30, 45, 60], dtype=DT)))
    with th.no_grad():
        err_naive = (20 * th.log10(ef_naive(uc).abs() / ef_true(uc).abs())).tolist()
        err_corr = (20 * th.log10(ef_corr(uc).abs() / ef_true(uc).abs())).tolist()
    report['ef_error_db'] = {'angles': [0, 10, 20, 30, 45, 60],
                             'naive_vs_true': [round(e, 2) for e in err_naive],
                             'corr_vs_true': [round(e, 2) for e in err_corr]}
    print(f"[보정] θ={report['calibration']['theta']} fit RMS {rms:.3f} dB")
    print(f"[EF오차 30°] naive {err_naive[3]:+.2f} → corr {err_corr[3]:+.2f} dB")

    # 2) 간격 최적화: NAIVE 모델 vs CORRECTED 모델
    true_model = patch(make_model(cfg), ef_true)   # 평가는 항상 TRUE(관측 대용)로
    s_naive, _, _ = design_on(ef_naive, cfg)
    s_corr, _, _ = design_on(ef_corr, cfg)

    # 3) 스냅 + TRUE 모델 평가 (실제 관측에서 누가 더 좋은가)
    for tag, s in (('naive_optimized', s_naive), ('graybox_corrected', s_corr)):
        d_snap, x_snap = snap(cfg, true_model, s)
        ok, _ = check_constraints(x_snap)
        r = evaluate_layout(x_snap, model=true_model)          # TRUE 모델로 채점
        r_self = evaluate_layout(x_snap, model=patch(make_model(cfg),
                                 ef_naive if tag == 'naive_optimized' else ef_corr))
        report[tag] = {'worst_on_TRUE': r['worst_psll_db'], 'at0_TRUE': r['psll_at_0'],
                       'worst_on_designmodel': r_self['worst_psll_db'],
                       'constraints_ok': ok}
        print(f"[{tag:18s}] 설계모델 {r_self['worst_psll_db']:+.2f} → 실제(TRUE) {r['worst_psll_db']:+.2f} dB")

    gain = report['naive_optimized']['worst_on_TRUE'] - report['graybox_corrected']['worst_on_TRUE']
    report['graybox_gain_db'] = round(gain, 3)
    print(f"[이득] gray-box 보정이 실제 성능 {gain:+.2f} dB 개선")

    # 4) 퇴화 실증 (앵커 없는 공동최적화)
    deg_loss, deg_theta = degeneracy_demo(cfg)
    report['degeneracy_demo'] = {'fake_loss': round(deg_loss, 2), 'theta': [round(t, 2) for t in deg_theta],
                                 'note': 'θ 앵커 없이 PSLL 공동최적화 → 가짜 손실(무의미). 반드시 데이터 앵커 필요'}
    print(f"[퇴화] 앵커 없는 공동최적화 가짜손실 {deg_loss:.2f}, θ={report['degeneracy_demo']['theta']} (부엽 인위 억제)")

    report['runtime_sec'] = round(time.time() - t0, 1)
    json.dump(report, open(os.path.join(RESULTS, 'graybox_experiment.json'), 'w'), indent=2, ensure_ascii=False)
    print(f"\n총 {report['runtime_sec']} s → results/graybox_experiment.json")


if __name__ == '__main__':
    main()
