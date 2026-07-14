# Gray-box 모델오차 보정 + 간격 공동최적화 실증 (v2: obliquity 구조항 분리)
#
# 아이디어: AF(간섭항)는 점광원 가정상 정확 → 계통오차는 EF 포락선에 몰림.
#   EF_corr(u) = sinc(w·u/λ) · (1−u²)^(p/2) · exp(g(u;θ))
#     ├ (1−u²)^(p/2) : obliquity 구조항 — |u|→1서 EF→0 강제(자유 다항의 꼬리 발산 차단)
#     └ g = a1u²+a2u⁴+a3u⁶ : 대역 내 잔차만, g(0)=0 (주엽 게이지 보존)
#
# 규율 (Kennedy-O'Hagan confounding 회피 / opus 워크플로 검증):
#   θ=(p, a1,a2,a3)  ← 관측(단채널 far-field)으로만 회귀.  PSLL 목적과 완전 분리.
#   x(간격)          ← θ 동결한 보정모델로만 최적화.
#   공동최적화 시 inf_θ PSLL=−∞ 퇴화 → degeneracy_demo()로 실증.
#
# 실측 Lumerical 부재 → 관측 대용 TRUE = Gaussian(w0=0.5µm)×√(1−u²).
# 보정 인프라는 config(ef_w_eff/ef_oblq_p/ef_gcoef) → model.element_factor_amp (monkeypatch 불요).
#
# 실행: venv/bin/python experiment_graybox.py
import json
import math
import os
import time
import types

import pandas as pd
import torch as th

from benchmark_common import check_constraints, evaluate_layout, gaps_to_positions, make_model
from config import JointConfig
from main_design_multiangle import run_multiangle_single, snap

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64
LAM, W = 1.55, 1.0
DENSE = tuple(float(a) for a in range(-30, 31, 5))


def ef_naive(u):
    return th.sinc(W * u / LAM)


def ef_true(u):
    # 관측 대용 진짜 소자인자: Gaussian 근접장(w0=0.5µm) FT × RS-I 경사인자
    return th.exp(-(math.pi * 0.5 * u / LAM) ** 2) * th.sqrt((1 - u ** 2).clamp(min=0))


def calibrate(p_struct=1.0, n_obs=15, obs_span=60.0, noise_db=0.1, seed=0):
    # 구조항 obliquity p 고정(RS-I p=1) → 잔차 g만 회귀.
    #   g(u) = log( EF_true / [sinc·(1−u²)^(p/2)] )  를 짝수기저에 닫힌형 최소자승.
    th.manual_seed(seed)
    ang = th.linspace(-obs_span, obs_span, n_obs, dtype=DT)
    u = th.sin(th.deg2rad(ang))
    struct = ef_naive(u) * (1 - u ** 2).clamp(min=1e-9) ** (p_struct / 2)
    with th.no_grad():
        tgt = th.log(ef_true(u).clamp(min=1e-6) / struct.clamp(min=1e-6))
        tgt = tgt + (noise_db * math.log(10) / 10) * th.randn(n_obs, dtype=DT)
    B = th.stack([u ** 2, u ** 4, u ** 6], dim=1)
    g = th.linalg.lstsq(B, tgt).solution
    rms = ((B @ g - tgt) ** 2).mean().sqrt().item() * 10 / math.log(10)
    return p_struct, g.tolist(), rms


def cfg_with_ef(base, p, gcoef):
    from dataclasses import replace
    return replace(base, ef_oblq_p=p, ef_gcoef=tuple(gcoef))


def cfg_true(base):
    # TRUE(관측) 모델 config — element_factor_amp를 patch로 대체(해석족 밖이라 config 표현 불가)
    return base


def true_model(base):
    m = make_model(base)
    m.element_factor_amp = types.MethodType(lambda self, u: ef_true(u), m)
    return m


def design(cfg, seeds=8):
    best_v, best_s, best_m = math.inf, None, None
    for r in range(seeds):
        res = run_multiangle_single(cfg, cfg.seed + r)
        if res['worst_psll'] < best_v:
            best_v, best_s, best_m = res['worst_psll'], res['s'], res['model']
    return best_s, best_m, best_v


def degeneracy_demo(cfg, seed=1):
    # 함정 실증: θ와 s를 같은 PSLL 목적에 공동최적화 → θ가 자유도로 PSLL 가짜 개선
    from losses import beta_schedule, soft_psll
    th.manual_seed(seed)
    m = make_model(cfg)
    n_gap = cfg.line_N - 1
    s = (th.full((n_gap,), m.s_init_uniform(), dtype=DT) + 0.5 * th.randn(n_gap, dtype=DT)).requires_grad_(True)
    theta = th.zeros(3, dtype=DT, requires_grad=True)   # 앵커 없는 보정변수
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
    base = JointConfig(design_angles_deg=DENSE, beta_end=6.0, angle_agg_gamma=2.0,
                       s_init_std=0.5, epochs_joint=800)
    report = {}

    # 1) 보정 회귀 (obliquity 구조항 p=1 분리 + 잔차 g)
    p, g, rms = calibrate()
    cfg_corr = cfg_with_ef(base, p, g)
    report['calibration'] = {'oblq_p': p, 'g_coef': [round(v, 4) for v in g], 'fit_rms_db': round(rms, 3),
                             'note': 'obliquity p=1 구조분리 후 잔차 g만 회귀(꼬리 발산 차단)'}
    uc = th.sin(th.deg2rad(th.tensor([0., 10, 20, 30, 45, 60, 80], dtype=DT)))
    mn, mc, mt = make_model(base), make_model(cfg_corr), true_model(base)
    with th.no_grad():
        e_n = (20 * th.log10(mn.element_factor_amp(uc) / ef_true(uc).clamp(min=1e-9))).tolist()
        e_c = (20 * th.log10(mc.element_factor_amp(uc) / ef_true(uc).clamp(min=1e-9))).tolist()
    report['ef_error_db'] = {'angles': [0, 10, 20, 30, 45, 60, 80],
                             'naive_vs_true': [round(v, 2) for v in e_n],
                             'corr_vs_true': [round(v, 2) for v in e_c]}
    print(f"[보정] p={p}, g={report['calibration']['g_coef']}, RMS {rms:.3f} dB")
    print(f"[EF오차] 30°: naive {e_n[3]:+.2f}→corr {e_c[3]:+.2f} | 60°: {e_n[5]:+.2f}→{e_c[5]:+.2f} | 80°: {e_n[6]:+.2f}→{e_c[6]:+.2f} dB")

    # 2) N=32 설계: naive vs corrected, TRUE로 채점
    tm = true_model(base)
    s_naive, _, _ = design(base)
    s_corr, _, _ = design(cfg_corr)
    for tag, s, dm in (('naive_opt', s_naive, make_model(base)), ('graybox_opt', s_corr, make_model(cfg_corr))):
        d_snap, x = snap(base, dm, s)
        r_true = evaluate_layout(x, model=tm)
        r_self = evaluate_layout(x, model=dm)
        report[f'N32_{tag}'] = {'worst_TRUE': r_true['worst_psll_db'], 'worst_design': r_self['worst_psll_db'],
                                'optimism_db': round(r_self['worst_psll_db'] - r_true['worst_psll_db'], 2)}
        print(f"[N32 {tag:10s}] 설계 {r_self['worst_psll_db']:+.2f} → TRUE {r_true['worst_psll_db']:+.2f} (낙관 {report[f'N32_{tag}']['optimism_db']:+.2f})")
    report['N32_graybox_gain_db'] = round(report['N32_naive_opt']['worst_TRUE'] - report['N32_graybox_opt']['worst_TRUE'], 3)

    # 3) N=64/128: 기존 naive-설계 배치를 TRUE로 재채점 (모델오차의 실제 영향)
    for N, csv in ((64, 'final_spacing_N64.csv'), (128, 'final_spacing_N128.csv')):
        path = os.path.join(RESULTS, csv)
        if not os.path.exists(path):
            continue
        d = th.tensor(pd.read_csv(path)['d_um'].values, dtype=DT)
        x = gaps_to_positions(d)
        bN = JointConfig(line_N=N, design_angles_deg=DENSE, beta_end=6.0)
        r_naive = evaluate_layout(x, model=make_model(bN))
        r_true = evaluate_layout(x, model=true_model(bN))
        report[f'N{N}_reeval'] = {'reported_naive': r_naive['worst_psll_db'],
                                  'actual_TRUE': r_true['worst_psll_db'],
                                  'optimism_db': round(r_naive['worst_psll_db'] - r_true['worst_psll_db'], 2)}
        print(f"[N{N} 재평가] 보고값(naive) {r_naive['worst_psll_db']:+.2f} → 실제(TRUE) {r_true['worst_psll_db']:+.2f} (낙관 {report[f'N{N}_reeval']['optimism_db']:+.2f} dB)")

    # 4) 퇴화 실증
    dl, dt = degeneracy_demo(base)
    report['degeneracy_demo'] = {'fake_loss': round(dl, 2), 'theta': [round(v, 2) for v in dt],
                                 'note': 'θ 앵커 없이 공동최적화 → 가짜손실(부엽 인위억제). 분리 필수'}
    print(f"[퇴화] 앵커없는 공동최적화 가짜손실 {dl:.2f}")

    report['runtime_sec'] = round(time.time() - t0, 1)
    json.dump(report, open(os.path.join(RESULTS, 'graybox_experiment.json'), 'w'), indent=2, ensure_ascii=False)
    print(f"\n총 {report['runtime_sec']} s → results/graybox_experiment.json")


if __name__ == '__main__':
    main()
