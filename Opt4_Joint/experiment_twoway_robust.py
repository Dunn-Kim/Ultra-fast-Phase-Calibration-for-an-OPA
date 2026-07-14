# Two-way 곱패턴의 이상화 이득이 실세계(위상오차)에서 살아남는가?
#
# 이상화 결과: two-way 32+32 = −39dB (단일 −11.5 대비 −27dB). 그러나 Khachaturian
# 실측 transceiver는 −11.3dB(64+64). 괴리 원인 가설 = 무작위 위상오차.
# 곱패턴의 깊은 널(상쇄)이 위상오차에 취약 → 널이 메워지면 이득 붕괴 예상.
#
# 검증: 설계 후 각 소자에 독립 위상오차 ε~N(0,σ²) 주입(Tx/Rx 독립), MC로 worst PSLL 분포.
# 실행: venv/bin/python experiment_twoway_robust.py
import json
import math
import os

import torch as th

from experiment_twoway import (ANG, EF2V, LAM, UVAL, af_abs2, design_1way,
                               design_2way, ef2_at)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64


@th.no_grad()
def worst_1way_noisy(x, sigma, seed):
    th.manual_seed(seed)
    N = x.numel()
    worst = -math.inf
    for u0 in ANG:
        eps = sigma * th.randn(N, dtype=DT)
        ph = K * (UVAL.reshape(-1, 1) - u0) * x.reshape(1, -1) - eps.reshape(1, -1)
        I = EF2V * th.exp(1j * ph).sum(1).abs() ** 2 / N ** 2
        i0 = ef2_at(u0) * (th.exp(1j * (-eps)).sum().abs() ** 2) / N ** 2
        guard = 2 * LAM / (x[-1] - x[0])
        m = (UVAL - u0).abs() > guard
        worst = max(worst, (10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)).max().item())
    return worst


@th.no_grad()
def worst_2way_noisy(xt, xr, sigma, seed):
    th.manual_seed(seed)
    Nt, Nr = xt.numel(), xr.numel()
    worst = -math.inf
    for u0 in ANG:
        et, er = sigma * th.randn(Nt, dtype=DT), sigma * th.randn(Nr, dtype=DT)
        pht = K * (UVAL.reshape(-1, 1) - u0) * xt.reshape(1, -1) - et.reshape(1, -1)
        phr = K * (UVAL.reshape(-1, 1) - u0) * xr.reshape(1, -1) - er.reshape(1, -1)
        At = th.exp(1j * pht).sum(1).abs() ** 2
        Ar = th.exp(1j * phr).sum(1).abs() ** 2
        I = (EF2V ** 2) * At * Ar / (Nt ** 2 * Nr ** 2)
        i0t = th.exp(1j * (-et)).sum().abs() ** 2
        i0r = th.exp(1j * (-er)).sum().abs() ** 2
        i0 = (ef2_at(u0) ** 2) * i0t * i0r / (Nt ** 2 * Nr ** 2)
        guard = 2 * LAM / max((xt[-1] - xt[0]).item(), (xr[-1] - xr[0]).item())
        m = (UVAL - u0).abs() > guard
        worst = max(worst, (10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)).max().item())
    return worst


K = 2 * math.pi / LAM


def mc(fn, args, sigma, n=40):
    vals = [fn(*args, sigma, s) for s in range(n)]
    t = th.tensor(vals)
    return {'p50': round(t.median().item(), 2), 'p95': round(t.kthvalue(int(0.95*n)).values.item(), 2),
            'worst': round(t.max().item(), 2)}


def main():
    print('설계 중...')
    x1, v1 = design_1way(32, seeds=6, epochs=800)
    (xt, xr), v2 = design_2way(32, seeds=6, epochs=800)
    print(f'단일 32: {v1:+.2f} dB | two-way 32+32: {v2:+.2f} dB (무오차)')

    report = {'ideal': {'single32': round(v1, 2), 'twoway32+32': round(v2, 2)}}
    print('\n=== 위상오차 σ[rad] 스윕 (worst PSLL, MC 40) ===')
    print(f"{'σ[rad]':>7} {'σ[deg]':>7} | {'단일32 p50/p95':>18} | {'two-way p50/p95':>18}")
    sweep = []
    for sig in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        r1 = mc(worst_1way_noisy, (x1,), sig)
        r2 = mc(worst_2way_noisy, (xt, xr), sig)
        sweep.append({'sigma_rad': sig, 'sigma_deg': round(math.degrees(sig), 1),
                      'single': r1, 'twoway': r2})
        print(f"{sig:>7.2f} {math.degrees(sig):>7.1f} | {r1['p50']:>8.1f}/{r1['p95']:<8.1f} | {r2['p50']:>8.1f}/{r2['p95']:<8.1f}")
    report['phase_error_sweep'] = sweep
    json.dump(report, open(os.path.join(RESULTS, 'twoway_robust.json'), 'w'), indent=2, ensure_ascii=False)
    print(f"\n→ results/twoway_robust.json")


if __name__ == '__main__':
    main()
