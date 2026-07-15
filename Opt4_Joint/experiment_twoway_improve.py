# Two-way 개선 탐구 — 핵심 3문제
#
# Q1 동일 하드웨어에서도 이기는가?  two-way 16+16(=32채널) vs 단일 32
#      (앞선 32+32는 64채널이라 단일64와 비교해야 공정했음 → -39.1 vs -14.5로 이미 승)
# Q2 총 채널 고정 시 최적 Tx/Rx 분할은?  (8,56)/(16,48)/(24,40)/(32,32) @ total 64
# Q3 잡음강건 설계가 이득인가?  σ=0.1 위상오차 하 기대 PSLL을 직접 최적화 vs nominal 설계
#
# 실행: venv/bin/python experiment_twoway_improve.py
import json
import math
import os
import time

import torch as th

from experiment_twoway import (ANG, EF2, EF2V, LAM, UTR, UVAL, af_abs2,
                               design_1way, ef2_at, gaps, hard_worst_2way,
                               positions)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64
K = 2 * math.pi / LAM
DMIN, DMAX = 2.0, 5.0
S0 = math.log((3 - DMIN) / (DMAX - 3))


def soft_psll_2way_gen(xt, xr, u0, beta, eps_t=None, eps_r=None):
    Nt, Nr = xt.numel(), xr.numel()
    def af2(x, eps):
        ph = K * (UTR.reshape(-1, 1) - u0) * x.reshape(1, -1)
        if eps is not None:
            ph = ph - eps.reshape(1, -1)
        return th.exp(1j * ph).sum(1).abs() ** 2
    At, Ar = af2(xt, eps_t), af2(xr, eps_r)
    I = (EF2 ** 2) * At * Ar / (Nt ** 2 * Nr ** 2)
    # 실제 주엽 (오차 있으면 |AF(u0)|<N)
    i0t = (th.exp(1j * (-eps_t)).sum().abs() ** 2) if eps_t is not None else th.tensor(Nt ** 2, dtype=DT)
    i0r = (th.exp(1j * (-eps_r)).sum().abs() ** 2) if eps_r is not None else th.tensor(Nr ** 2, dtype=DT)
    i0 = (ef2_at(u0) ** 2) * i0t * i0r / (Nt ** 2 * Nr ** 2)
    guard = 2 * LAM / max((xt[-1] - xt[0]).item(), (xr[-1] - xr[0]).item())
    m = (UTR - u0).abs() > guard
    D = 10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)
    return th.logsumexp(beta * D, 0) / beta


def design_2way_gen(Nt, Nr, seeds=4, epochs=600, sigma=0.0, mc=4, seed0=400):
    best, best_v = None, math.inf
    for r in range(seeds):
        th.manual_seed(seed0 + r)
        st = (S0 + 0.5 * th.randn(Nt - 1, dtype=DT)).requires_grad_(True)
        sr = (math.log((3.3 - DMIN) / (DMAX - 3.3)) + 0.5 * th.randn(Nr - 1, dtype=DT)).requires_grad_(True)
        opt = th.optim.Adam([st, sr], lr=1e-2)
        for t in range(epochs):
            beta = 0.2 * (6 / 0.2) ** (t / epochs)
            opt.zero_grad()
            xt, xr = positions(st), positions(sr)
            if sigma > 0:   # 잡음강건: MC 샘플 기대 손실
                per = []
                for u0 in ANG:
                    acc = []
                    for m_ in range(mc):
                        et = sigma * th.randn(Nt, dtype=DT)
                        er = sigma * th.randn(Nr, dtype=DT)
                        acc.append(soft_psll_2way_gen(xt, xr, u0, beta, et, er))
                    per.append(th.stack(acc).mean())
                loss = th.logsumexp(2.0 * th.stack(per), 0) / 2.0
            else:
                loss = th.logsumexp(2.0 * th.stack([soft_psll_2way_gen(xt, xr, u0, beta) for u0 in ANG]), 0) / 2.0
            loss.backward(); opt.step()
        with th.no_grad():
            xt, xr = positions(st.detach()), positions(sr.detach())
            v = hard_worst_2way(xt, xr)
        if v < best_v:
            best_v, best = v, (xt, xr)
    return best, best_v


@th.no_grad()
def noisy_worst_2way(xt, xr, sigma, seed):
    th.manual_seed(seed)
    Nt, Nr = xt.numel(), xr.numel()
    worst = -math.inf
    for u0 in ANG:
        et, er = sigma * th.randn(Nt, dtype=DT), sigma * th.randn(Nr, dtype=DT)
        def af2(x, eps):
            ph = K * (UVAL.reshape(-1, 1) - u0) * x.reshape(1, -1) - eps.reshape(1, -1)
            return th.exp(1j * ph).sum(1).abs() ** 2
        I = (EF2V ** 2) * af2(xt, et) * af2(xr, er) / (Nt ** 2 * Nr ** 2)
        i0 = (ef2_at(u0) ** 2) * (th.exp(1j * (-et)).sum().abs() ** 2) * (th.exp(1j * (-er)).sum().abs() ** 2) / (Nt ** 2 * Nr ** 2)
        guard = 2 * LAM / max((xt[-1] - xt[0]).item(), (xr[-1] - xr[0]).item())
        m = (UVAL - u0).abs() > guard
        worst = max(worst, (10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)).max().item())
    return worst


def mc_stats(xt, xr, sigma, n=30):
    v = th.tensor([noisy_worst_2way(xt, xr, sigma, s) for s in range(n)])
    return {'p50': round(v.median().item(), 2), 'p95': round(v.kthvalue(int(0.95 * n)).values.item(), 2)}


def main():
    t0 = time.time()
    rep = {}

    print('=== Q1: 동일 하드웨어 대결 ===')
    x32, v_s32 = design_1way(32, seeds=6, epochs=600)
    (t16, r16), v_t16 = design_2way_gen(16, 16, seeds=6)
    rep['Q1'] = {'single_32ch': round(v_s32, 2), 'twoway_16+16_32ch': round(v_t16, 2),
                 'gain_db': round(v_t16 - v_s32, 2)}
    print(f"  단일 32채널      : {v_s32:+.2f} dB")
    print(f"  two-way 16+16(32채널): {v_t16:+.2f} dB  → 동일HW 이득 {v_t16 - v_s32:+.2f} dB")

    print('\n=== Q2: 총 64채널 고정, Tx/Rx 분할 스윕 ===')
    splits = []
    for nt, nr in ((8, 56), (16, 48), (24, 40), (32, 32)):
        (a, b), v = design_2way_gen(nt, nr, seeds=3, epochs=500)
        splits.append({'split': f'{nt}+{nr}', 'worst': round(v, 2)})
        print(f"  Tx {nt:2d} + Rx {nr:2d}: {v:+.2f} dB")
    rep['Q2_splits'] = splits

    print('\n=== Q3: 잡음강건 설계 (σ=0.1 하 기대손실 직접 최적화) ===')
    (rt, rr), v_rob_ideal = design_2way_gen(32, 32, seeds=3, epochs=500, sigma=0.1, mc=4)
    (nt2, nr2), v_nom_ideal = design_2way_gen(32, 32, seeds=3, epochs=500, sigma=0.0)
    s_rob = mc_stats(rt, rr, 0.1)
    s_nom = mc_stats(nt2, nr2, 0.1)
    rep['Q3'] = {'nominal_design': {'ideal': round(v_nom_ideal, 2), 'at_sigma0.1': s_nom},
                 'robust_design': {'ideal': round(v_rob_ideal, 2), 'at_sigma0.1': s_rob},
                 'robust_gain_p50_db': round(s_rob['p50'] - s_nom['p50'], 2)}
    print(f"  nominal 설계: 무오차 {v_nom_ideal:+.2f} | σ0.1 p50/p95 {s_nom['p50']:+.1f}/{s_nom['p95']:+.1f}")
    print(f"  강건  설계: 무오차 {v_rob_ideal:+.2f} | σ0.1 p50/p95 {s_rob['p50']:+.1f}/{s_rob['p95']:+.1f}")
    print(f"  → 강건설계 이득(σ0.1 p50) {s_rob['p50'] - s_nom['p50']:+.2f} dB")

    rep['runtime_sec'] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(RESULTS, 'twoway_improve.json'), 'w'), indent=2, ensure_ascii=False)
    print(f"\n총 {rep['runtime_sec']} s → results/twoway_improve.json")


if __name__ == '__main__':
    main()
