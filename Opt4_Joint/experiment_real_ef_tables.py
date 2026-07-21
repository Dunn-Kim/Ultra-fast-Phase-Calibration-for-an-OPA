# 실-MODE 적합 EF(기본 config) 기준 공식 표 재발행
#   A. 조향 상한표 — 주엽 강도 임계별 최대 조향각 (φ는 φ*로 항상 최적 → 상한은 EF가 결정)
#   B. 조향폭별 비등간격 설계표 — worst-case PSLL (설계각 {0, ±max/2, ±max};
#      각도 밀도는 무관함이 실증됨: results/loss_angle_design.json)
#
# 실행: venv/bin/python experiment_real_ef_tables.py
import json
import math
import os

import pandas as pd
import torch as th

from config import JointConfig
from model import OPAModel

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64


def ceiling_table(model, model_sinc):
    u = th.linspace(0.0, 0.999, 20000, dtype=DT)
    out = {}
    for tag, m in (('sinc_legacy', model_sinc), ('mode_fitted', model)):
        e = m.element_factor_amp(u) ** 2
        e = e / e[0]
        row = {}
        for thr in (0.9, 0.8, 0.7, 0.5, 0.3):
            idx = int((e >= thr).sum().item()) - 1
            row[f'>={int(thr * 100)}%'] = round(math.degrees(math.asin(u[idx].item())), 1)
        out[tag] = row
    return out


def phi_star(model, x, u0):
    p = -model.k * x * u0
    return -(p - p[0])            # model.intensity 는 −φ 규약 → 부호 반전해 전달


def soft_psll(model, x, u0, beta):
    phi = phi_star(model, x, u0)
    I = model.intensity(x, phi, model.u_train)
    i0 = model.intensity_at_u0(x, phi, u0)
    L = (x[-1] - x[0]).detach()
    m = (model.u_train - u0).abs() > 2 * model.cfg.wavelength / L
    D = 10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)
    return th.logsumexp(beta * D, 0) / beta


@th.no_grad()
def hard_worst_psll(model, x, eval_angles):
    worst, mains = -math.inf, []
    for a in eval_angles:
        u0 = math.sin(math.radians(a))
        phi = phi_star(model, x, u0)
        I = model.intensity(x, phi, model.u_val)
        i0 = model.intensity_at_u0(x, phi, u0)
        L = x[-1] - x[0]
        m = (model.u_val - u0).abs() > 2 * model.cfg.wavelength / L
        worst = max(worst, (10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)).max().item())
        af0 = i0 / model.element_factor_amp(th.tensor(u0, dtype=DT)) ** 2
        mains.append(af0.item())
    return worst, min(mains)


def optimize_range(model, max_deg, restarts=6, epochs=700, seed=42):
    cfg = model.cfg
    design = sorted({0.0, max_deg / 2, -max_deg / 2, float(max_deg), -float(max_deg)})
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
            per = [soft_psll(model, x, math.sin(math.radians(a)), beta) for a in design]
            loss = th.logsumexp(2.0 * th.stack(per), 0) / 2.0
            loss.backward()
            opt.step()
        x = model.positions(s.detach())
        ev = [a for a in [i * 2.5 for i in range(-int(max_deg / 2.5), int(max_deg / 2.5) + 1)]]
        v, _ = hard_worst_psll(model, x, ev)
        if v < best_v:
            best_v, best = v, x
    ev = [a for a in [i * 2.5 for i in range(-int(max_deg / 2.5), int(max_deg / 2.5) + 1)]]
    _, main_min = hard_worst_psll(model, best, ev)
    return best, best_v, main_min, ev


def main():
    cfg = JointConfig()
    model = OPAModel(cfg)
    model_sinc = OPAModel(JointConfig(ef_oblq_p=0.0, ef_gcoef=(0.0, 0.0, 0.0)))

    print('A. 조향 상한표 (주엽 강도 임계)')
    ceil = ceiling_table(model, model_sinc)
    for tag, row in ceil.items():
        print(f'   [{tag}] ' + '  '.join(f'{k}→±{v}°' for k, v in row.items()))

    print('\nB. 조향폭별 설계표 (기본 config = MODE 적합 EF, 판정각 2.5° 간격)')
    table = {}
    for max_deg in (10, 15, 20, 30):
        x, v, af_min, ev = optimize_range(model, max_deg)
        xu = model.uniform_positions()
        vu, _ = hard_worst_psll(model, xu, ev)
        d = x[1:] - x[:-1]
        table[f'pm{max_deg}'] = {
            'worst_psll_db': round(v, 2), 'uniform_psll_db': round(vu, 2),
            'gain_db': round(vu - v, 2), 'min_af_eff': round(af_min, 4),
            'aperture_um': round(x[-1].item(), 1),
            'd_range_um': [round(d.min().item(), 3), round(d.max().item(), 3)],
        }
        print(f'   ±{max_deg:2d}°: 비등간격 {v:+.2f} dB | 등간격 {vu:+.2f} dB | 이득 {vu - v:+.2f} dB'
              f' | AF효율 최저 {af_min:.4f} | 개구 {x[-1]:.1f}µm')
        if max_deg == 15:
            pd.DataFrame({'d_um': d.numpy()}).to_csv(
                os.path.join(RESULTS, 'final_spacing_realEF_pm15.csv'), index=False)

    json.dump({'ceiling_deg': ceil, 'design': table,
               'ef': {'oblq_p': cfg.ef_oblq_p, 'gcoef': list(cfg.ef_gcoef),
                      'source': 'Opt3_Back_Forward/Resultants varFDTD 로그 적합 (N전이 R²0.996)'}},
              open(os.path.join(RESULTS, 'real_ef_tables.json'), 'w'),
              indent=2, ensure_ascii=False)
    print('\n→ results/real_ef_tables.json, results/final_spacing_realEF_pm15.csv')


if __name__ == '__main__':
    main()
