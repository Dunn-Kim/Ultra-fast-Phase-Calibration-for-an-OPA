# 커플링 페널티 λ 스윕 — (a)물리형+(b)장벽형 혼합의 비용/효과 곡선과 무릎점 선택
#   과제: ±15° worst-case PSLL (설계각 {0,±7.5,±15}, 판정각 2.5° 간격, MODE 적합 EF)
#   λ = 페널티 전역 배율 (config 기본 가중 × λ). λ=0 = 페널티 없음(기존 챔피언 재현 조건).
#   보고: PSLL | 최소 간격 | 2.2µm 미만 개수 | crosstalk 프록시 10·log10(mean exp(−γ(d−w)))
#
# 실행: venv/bin/python experiment_coupling_penalty.py
import json
import math
import os

import pandas as pd
import torch as th

from config import JointConfig
from losses import coupling_penalty
from model import OPAModel
from experiment_real_ef_tables import phi_star, soft_psll, hard_worst_psll

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64
DESIGN = [0.0, 7.5, -7.5, 15.0, -15.0]
EVAL = [i * 2.5 for i in range(-6, 7)]


def optimize(model, lam, restarts=5, epochs=650, seed=42):
    cfg = model.cfg
    s0 = model.s_init_uniform()
    best, best_v = None, math.inf
    for r in range(restarts):
        th.manual_seed(seed + r)
        s = (s0 + 0.5 * th.randn(cfg.line_N - 1, dtype=DT)).requires_grad_(True)
        opt = th.optim.Adam([s], lr=cfg.lr_spacing)
        for t in range(epochs):
            beta = 0.2 * (6.0 / 0.2) ** (t / epochs)
            opt.zero_grad()
            x = model.positions(s)
            per = [soft_psll(model, x, math.sin(math.radians(a)), beta) for a in DESIGN]
            loss = th.logsumexp(2.0 * th.stack(per), 0) / 2.0
            if lam > 0:
                loss = loss + coupling_penalty(model, model.gaps(s), weight=lam)
            loss.backward()
            opt.step()
        x = model.positions(s.detach())
        v, _ = hard_worst_psll(model, x, EVAL)      # 판정은 항상 페널티 없는 PSLL
        if v < best_v:
            best_v, best = v, x
    return best, best_v


def xtalk_db(model, d):
    c = model.cfg
    return 10 * math.log10(th.exp(-c.cpl_gamma * (d - c.element_width)).mean().item())


def main():
    cfg = JointConfig()
    model = OPAModel(cfg)
    rows = {}
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        x, v = optimize(model, lam)
        d = x[1:] - x[:-1]
        rows[str(lam)] = {
            'psll_db': round(v, 2),
            'min_gap_um': round(d.min().item(), 3),
            'gaps_below_2p2': int((d < 2.2).sum().item()),
            'xtalk_proxy_db': round(xtalk_db(model, d), 1),
            'aperture_um': round(x[-1].item(), 1),
        }
        r = rows[str(lam)]
        print(f"λ={lam:4.2f}  PSLL {r['psll_db']:+.2f} dB | min d {r['min_gap_um']:.3f} µm"
              f" | <2.2µm {r['gaps_below_2p2']:2d}개 | xtalk {r['xtalk_proxy_db']:+.1f} dB"
              f" | 개구 {r['aperture_um']:.1f} µm", flush=True)
        if lam == 1.0:
            pd.DataFrame({'d_um': d.numpy()}).to_csv(
                os.path.join(RESULTS, 'final_spacing_realEF_pm15_cpl.csv'), index=False)

    json.dump({'task': 'pm15', 'design_angles': DESIGN,
               'penalty': {'gamma': cfg.cpl_gamma, 'd_safe': cfg.cpl_d_safe, 'tau': cfg.cpl_tau,
                           'w_phys': cfg.cpl_w_phys, 'w_barrier': cfg.cpl_w_barrier},
               'sweep': rows},
              open(os.path.join(RESULTS, 'coupling_penalty_sweep.json'), 'w'),
              indent=2, ensure_ascii=False)
    print('\n→ results/coupling_penalty_sweep.json, results/final_spacing_realEF_pm15_cpl.csv')


if __name__ == '__main__':
    main()
