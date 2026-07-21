# Stage B — 캘리브레이션: 간격 동결 + 위상만 (하드웨어 현실 반영)
#
# 제작된 칩의 간격은 런타임 변경 불가 → final_spacing.csv를 requires_grad=False로 동결.
# 무작위 위상 오차 U(−π, π) 주입(제작/열 오차 모사) 후 위상만 Adam 캘리브레이션 (E2).
# 간격 학습 요청 + 동결 간격 조합은 ValueError — 워크플로 코드 강제.
#
# 차후 Lumerical 연동: Opt3 Lumerical_Func.Phase를 path만 바꿔 재사용 (위상만 in-loop).
#
# 실행: venv/bin/python main_calibration.py [--trials 20]
import argparse
import json
import math
import os

import pandas as pd
import torch as th

from config import JointConfig
from losses import main_lobe_loss
from metrics import hard_psll_db
from model import OPAModel

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def load_frozen_positions(cfg, spacing_csv):
    d = th.tensor(pd.read_csv(spacing_csv)['d_um'].values, dtype=cfg.dtype, device=cfg.device)
    zero = th.zeros(1, dtype=cfg.dtype, device=cfg.device)
    x = th.cat([zero, th.cumsum(d, dim=0)])
    x.requires_grad_(False)
    return x


def calibrate_fast(cfg, model, x_frozen, shoot, frames=5, inner=250, lr=0.2, restarts=3):
    """
    프로덕션 캘리브레이션 — 카메라 프레임만 사용 (ε 미지, 실칩 적용 가능).

    shoot(phi_applied) -> 전체 far-field 강도 패턴 (model.u_train 격자, 1D 텐서)
      = 카메라 프레임 1장. 이것이 유일한 관측 통로.

    원리: MATLAB REV(getSpot)는 프레임에서 스팟 강도 1개만 뽑아 31개 미지수에 5N=155장을 쓴다.
    카메라는 매 프레임 전체 패턴(수천 픽셀 = 독립 방정식 다수)을 주므로, 전체 패턴에
    ε̂ 를 적합하면 5장으로 충분하다 (155 → 5, 31배). 벤치마크: calib_benchmark.py

    프로브: 랜덤 U(−π,π). Hadamard 는 ±π/2 2값뿐이라 프레임당 다양성이 빈약해 오히려 열세
    (calib_probe_design.json 실측). 멀티스타트로 국소최소 회피.
    식별성 요건: 프로브 간 랩 고려 평균 |Δp| ≳ 1 rad — 랜덤 프로브가 자동 보장.
    (유사 프로브(예: 최적화 궤적 프레임)는 단일 프레임 위상복원 급으로 퇴화 — MODE 로그 실증.)

    프레임별 강도 스케일 c_f 를 닫힌형 최소자승으로 동시 추정 — 실 카메라의 미지
    게인/노출 대응 (스케일 불변 적합). 초기화는 협역 N(0,0.3)·광역 U(−π,π) 혼합.

    반환: 인가할 보정 위상 φ_a (= −ε̂). 하드웨어 변경 없음 — 위상만 조율.
    """
    if x_frozen.requires_grad:
        raise ValueError('캘리브레이션 단계에서 간격은 변수가 될 수 없음 (Stage A에서 확정)')
    th.manual_seed(cfg.seed)
    probes, obs = [], []
    for f in range(frames):
        p = th.zeros(cfg.line_N, dtype=cfg.dtype) if f == 0 else \
            (2 * math.pi * th.rand(cfg.line_N, dtype=cfg.dtype) - math.pi)
        p[0] = 0.0                                  # 채널 0 = 위상 기준
        probes.append(p)
        obs.append(shoot(p))                        # 카메라 프레임 1장

    best, best_loss = None, float('inf')
    for r in range(restarts):
        th.manual_seed(cfg.seed + r)
        eh = (0.3 * th.randn(cfg.line_N, dtype=cfg.dtype) if r % 2 == 0 else
              2 * math.pi * th.rand(cfg.line_N, dtype=cfg.dtype) - math.pi).requires_grad_(True)
        opt = th.optim.Adam([eh], lr=lr)
        loss = None
        for _ in range(inner):
            opt.zero_grad()
            loss = eh.new_zeros(())
            for p, o in zip(probes, obs):
                m = model.intensity(x_frozen, -(p + eh), model.u_train)
                c = (m * o).sum() / (m * m).sum().clamp(min=1e-30)   # 프레임별 게인 (닫힌형)
                loss = loss + ((c * m - o) ** 2).mean()
            loss.backward()
            opt.step()
        if loss.item() < best_loss:
            best_loss, best = loss.item(), eh.detach().clone()
    return -best


def calibrate(cfg, model, x_frozen, phase_error, epochs=300):
    # 참고용(시뮬레이션 전용) — ε를 인자로 받으므로 실칩에는 쓸 수 없다. 실칩은 calibrate_fast 사용.
    if x_frozen.requires_grad:
        raise ValueError('캘리브레이션 단계에서 간격은 변수가 될 수 없음 (Stage A에서 확정)')
    phi = th.zeros(cfg.line_N, dtype=cfg.dtype, device=cfg.device, requires_grad=True)
    opt = th.optim.Adam([phi], lr=cfg.lr_phase)
    for _ in range(epochs):
        opt.zero_grad()
        loss, _ = main_lobe_loss(model, x_frozen, phi + phase_error)
        loss.backward()
        opt.step()
    assert x_frozen.grad is None, '간격으로 gradient가 흘렀음 — 동결 위반'
    return phi.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spacing', default=os.path.join(RESULTS, 'final_spacing.csv'))
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=300)
    args = ap.parse_args()

    cfg = JointConfig()
    model = OPAModel(cfg)
    x = load_frozen_positions(cfg, args.spacing)

    # 기준: 오차 없는 이상 상태 (φ = 해석 조향해)
    phi_ideal = model.steering_phase(x)
    with th.no_grad():
        i0_ideal = model.intensity_at_u0(x, phi_ideal).item()

    th.manual_seed(cfg.seed)
    rows = []
    for t in range(args.trials):
        err = (2 * th.rand(cfg.line_N, dtype=cfg.dtype) - 1) * math.pi  # U(−π, π)
        with th.no_grad():
            i0_before = model.intensity_at_u0(x, phi_ideal + err).item()
        phi_cal = calibrate(cfg, model, x, err, args.epochs)
        with th.no_grad():
            i0_after = model.intensity_at_u0(x, phi_cal + err).item()
            psll = hard_psll_db(model, x, phi_cal + err)
        rows.append({'trial': t, 'I0_corrupted': i0_before, 'I0_calibrated': i0_after,
                     'recovery': i0_after / i0_ideal, 'psll_db': psll})
        print(f"[E2] trial {t:02d}  손상 {i0_before:.4f} → 보정 {i0_after:.4f} "
              f"(회복률 {rows[-1]['recovery']*100:.1f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'calibration_trials.csv'), index=False)
    summary = {'trials': args.trials, 'I0_ideal': i0_ideal,
               'recovery_min': df['recovery'].min(), 'recovery_mean': df['recovery'].mean(),
               'converged_95pct': int((df['recovery'] >= 0.95).sum()),
               'psll_worst_db': df['psll_db'].max()}
    with open(os.path.join(RESULTS, 'calibration_report.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n회복률 평균 {summary['recovery_mean']*100:.1f}% / 최저 {summary['recovery_min']*100:.1f}%"
          f" / ≥95% 달성 {summary['converged_95pct']}/{args.trials}")
    return summary


if __name__ == '__main__':
    main()
