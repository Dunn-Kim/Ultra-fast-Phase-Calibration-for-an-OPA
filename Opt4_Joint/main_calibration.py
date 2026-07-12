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


def calibrate(cfg, model, x_frozen, phase_error, epochs=300):
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
