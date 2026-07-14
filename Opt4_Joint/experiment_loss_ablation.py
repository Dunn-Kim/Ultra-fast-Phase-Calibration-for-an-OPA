# 손실함수/최적화기 정비 필요성 검증 — 노브별 단일인자 ablation
#
# 질문: 손실(가드마스크 κ, soft-max 온도 β, 각도집계 γ, 각도밀도)이나 옵티마이저(lr)를
#       손보면 −11.6 dB 벽이 움직이는가?
# 프로토콜: 각 구성 = 챔피언 레시피에서 노브 1개만 변경, 동일 예산(6 restart × 800 epoch).
#           budget 대조군(base) 포함 — 예산 효과와 노브 효과 분리.
#           평가는 전부 고정 벤치마크(κ=2, ±30° 스윕, polish, 40001 격자) — 설계 노브만 변함.
#
# 실행: venv/bin/python experiment_loss_ablation.py
import json
import math
import os
import time

import torch as th

from benchmark_common import evaluate_layout, make_model
from config import JointConfig
from losses import beta_schedule, multi_angle_worst_psll
from main_design_multiangle import run_multiangle_single

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

BUDGET = dict(restarts=6, epochs_joint=800, s_init_std=0.5)
DENSE = tuple(float(a) for a in range(-30, 31, 5))

CONFIGS = {
    'base(κ2,β2,γ2)':      dict(),
    'κ=1.5(마스크 좁게)':    dict(guard_kappa=1.5),
    'κ=3.0(마스크 넓게)':    dict(guard_kappa=3.0),
    'β_end=6(hard 근접)':   dict(beta_end=6.0),
    'β_end=1(soft하게)':    dict(beta_end=1.0),
    'γ=4(worst 강화)':      dict(angle_agg_gamma=4.0),
    'γ=0.5(평균 근접)':     dict(angle_agg_gamma=0.5),
    '각도 5°조밀(13각)':     dict(design_angles_deg=DENSE),
    'lr_s=3e-2(옵티마이저)': dict(lr_spacing=3e-2),
}


def run_config(name, overrides):
    kw = {'angle_agg_gamma': 2.0, **BUDGET, **overrides}   # overrides 우선
    cfg = JointConfig(**kw)
    best_s, best_v, best_model = None, math.inf, None
    for r in range(cfg.restarts):
        res = run_multiangle_single(cfg, cfg.seed + r)
        if res['worst_psll'] < best_v:
            best_v, best_s, best_model = res['worst_psll'], res['s'], res['model']
    x = best_model.positions(best_s).detach()
    return x, best_v


def main():
    t0 = time.time()
    eval_model = make_model()          # 평가 규약 고정 (κ=2)
    report = {'reference': {'champion_full_budget': -11.54, 'bilevel_cma': -11.62,
                            'protocol': '설계 6restart×800ep(축소예산), 평가 고정 κ=2'}}
    for name, ov in CONFIGS.items():
        t1 = time.time()
        x, design_v = run_config(name, ov)
        r = evaluate_layout(x, model=eval_model)
        report[name] = {'worst_psll_db': r['worst_psll_db'], 'psll_at_0': r['psll_at_0'],
                        'design_internal_db': round(design_v, 2),
                        'aperture_um': r['aperture_um'], 'sec': round(time.time() - t1)}
        print(f"{name:22s} 설계내부 {design_v:+6.2f} → 벤치마크 worst {r['worst_psll_db']:+6.2f} dB "
              f"(@0° {r['psll_at_0']:+6.2f}) [{report[name]['sec']}s]")

    report['runtime_sec'] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, 'loss_ablation.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n총 {report['runtime_sec']} s → results/loss_ablation.json")


if __name__ == '__main__':
    main()
