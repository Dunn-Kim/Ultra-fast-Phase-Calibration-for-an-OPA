# Two-way(송수신) 곱패턴 재정식화 실증 — 전역 재설계 #1 후보 수치 검증
#
# 주장(문헌): LIDAR는 송수신 시스템 → 유효 패턴 = I_tx(u)·I_rx(u).
#   Tx의 사이드로브가 Rx의 저엽/영점에 놓이면 곱에서 상쇄 → 단일배열 PSLL 바닥 돌파.
#   앵커: Khachaturian-Hajimiri co-prime transceiver(arXiv:2108.10223),
#         Dostart Vernier OPA(Opt.Express 30:24589, grating-lobe +6.4dB).
#
# 정직한 대조 (핵심): two-way(Tx N + Rx N)는 하드웨어 2N. 공정 비교 =
#   (a) 단일배열 N,  (b) 단일배열 2N,  (c) two-way N+N (곱).
#   곱 구조가 '단순히 소자 2배'를 넘는 이득을 주는가? → (c) vs (b) 가 진짜 질문.
#
# 목적: 조향 ±30°(5°,13각) worst-case PSLL. 위상=각 배열 해석 조향해. 균일진폭·d_min=2µm 유지.
# 실행: venv/bin/python experiment_twoway.py
import json
import math
import os
import time

import torch as th

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64
LAM, W = 1.55, 1.0
DMIN, DMAX = 2.0, 5.0
K = 2 * math.pi / LAM
ANG = [math.sin(math.radians(a)) for a in range(-30, 31, 5)]
UVAL = th.linspace(-1, 1, 40001, dtype=DT)
UTR = th.linspace(-1, 1, 4001, dtype=DT)
EF2 = th.sinc(W * UTR / LAM) ** 2
EF2V = th.sinc(W * UVAL / LAM) ** 2


def gaps(s):
    return DMIN + (DMAX - DMIN) * th.sigmoid(s)


def positions(s):
    return th.cat([th.zeros(1, dtype=DT), th.cumsum(gaps(s), 0)])


def af_abs2(x, u, u0):
    # |AF(u)|² with 해석 조향해 φ=k·x·u0
    ph = K * (u.reshape(-1, 1) - u0) * x.reshape(1, -1)
    return th.exp(1j * ph).sum(1).abs() ** 2


def soft_psll_1way(x, u0, uf, ef2, beta):
    N = x.numel()
    I = ef2 * af_abs2(x, uf, u0) / N ** 2
    i0 = af_abs2(x, th.tensor([u0], dtype=DT), u0)[0] / N ** 2  # EF(u0)² 공통이라 비에서 소거
    guard = 2 * LAM / (x[-1] - x[0])
    m = (uf - u0).abs() > guard
    D = 10 * th.log10(I[m] / (ef2_at(u0) * i0 + 1e-30) + 1e-30)
    return th.logsumexp(beta * D, 0) / beta


def ef2_at(u0):
    return th.sinc(th.tensor(W * u0 / LAM, dtype=DT)) ** 2


def soft_psll_2way(xt, xr, u0, uf, ef2, beta):
    Nt, Nr = xt.numel(), xr.numel()
    I = (ef2 ** 2) * af_abs2(xt, uf, u0) * af_abs2(xr, uf, u0) / (Nt ** 2 * Nr ** 2)
    i0 = (ef2_at(u0) ** 2) * 1.0  # |AF_tx(u0)|²|AF_rx(u0)|²/(Nt²Nr²) = 1 at u0
    guard = 2 * LAM / max((xt[-1] - xt[0]).item(), (xr[-1] - xr[0]).item())
    m = (uf - u0).abs() > guard
    D = 10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)
    return th.logsumexp(beta * D, 0) / beta


@th.no_grad()
def hard_worst_1way(x):
    worst = -math.inf
    for u0 in ANG:
        N = x.numel()
        I = EF2V * af_abs2(x, UVAL, u0) / N ** 2
        i0 = ef2_at(u0)
        guard = 2 * LAM / (x[-1] - x[0])
        m = (UVAL - u0).abs() > guard
        worst = max(worst, (10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)).max().item())
    return worst


@th.no_grad()
def hard_worst_2way(xt, xr):
    Nt, Nr = xt.numel(), xr.numel()
    worst = -math.inf
    for u0 in ANG:
        I = (EF2V ** 2) * af_abs2(xt, UVAL, u0) * af_abs2(xr, UVAL, u0) / (Nt ** 2 * Nr ** 2)
        i0 = ef2_at(u0) ** 2
        guard = 2 * LAM / max((xt[-1] - xt[0]).item(), (xr[-1] - xr[0]).item())
        m = (UVAL - u0).abs() > guard
        worst = max(worst, (10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)).max().item())
    return worst


def design_1way(N, seeds=8, epochs=800):
    best_x, best_v = None, math.inf
    for r in range(seeds):
        th.manual_seed(100 + r)
        s = (math.log((3 - DMIN) / (DMAX - 3)) + 0.5 * th.randn(N - 1, dtype=DT)).requires_grad_(True)
        opt = th.optim.Adam([s], lr=1e-2)
        for t in range(epochs):
            beta = 0.2 * (6 / 0.2) ** (t / epochs)
            opt.zero_grad()
            x = positions(s)
            loss = th.logsumexp(2.0 * th.stack([soft_psll_1way(x, u0, UTR, EF2, beta) for u0 in ANG]), 0) / 2.0
            loss.backward(); opt.step()
        v = hard_worst_1way(positions(s.detach()))
        if v < best_v:
            best_v, best_x = v, positions(s.detach())
    return best_x, best_v


def design_2way(N, seeds=8, epochs=800):
    best, best_v = None, math.inf
    for r in range(seeds):
        th.manual_seed(200 + r)
        st = (math.log((3 - DMIN) / (DMAX - 3)) + 0.5 * th.randn(N - 1, dtype=DT)).requires_grad_(True)
        sr = (math.log((3.3 - DMIN) / (DMAX - 3.3)) + 0.5 * th.randn(N - 1, dtype=DT)).requires_grad_(True)  # Rx 다른 피치 출발
        opt = th.optim.Adam([st, sr], lr=1e-2)
        for t in range(epochs):
            beta = 0.2 * (6 / 0.2) ** (t / epochs)
            opt.zero_grad()
            xt, xr = positions(st), positions(sr)
            loss = th.logsumexp(2.0 * th.stack([soft_psll_2way(xt, xr, u0, UTR, EF2, beta) for u0 in ANG]), 0) / 2.0
            loss.backward(); opt.step()
        v = hard_worst_2way(positions(st.detach()), positions(sr.detach()))
        if v < best_v:
            best_v, best = v, (positions(st.detach()), positions(sr.detach()))
    return best, best_v


def main():
    t0 = time.time()
    report = {}
    print('=== 단일배열 baseline ===')
    for N in (32, 64):
        x, v = design_1way(N)
        report[f'single_N{N}'] = round(v, 2)
        print(f"  single N={N}: worst {v:+.2f} dB")

    print('=== two-way 곱패턴 (Tx N + Rx N) ===')
    for N in (32,):
        (xt, xr), v = design_2way(N)
        report[f'twoway_{N}+{N}'] = round(v, 2)
        report[f'twoway_{N}+{N}_apertures'] = [round((xt[-1]-xt[0]).item(), 1), round((xr[-1]-xr[0]).item(), 1)]
        print(f"  two-way {N}+{N} (곱): worst {v:+.2f} dB | 개구 Tx {xt[-1]-xt[0]:.0f} Rx {xr[-1]-xr[0]:.0f}µm")

    # 핵심 판정
    s32, s64, t32 = report['single_N32'], report['single_N64'], report['twoway_32+32']
    report['verdict'] = {
        'twoway_vs_single32': round(t32 - s32, 2),
        'twoway_vs_single64_sameHW': round(t32 - s64, 2),
        'note': 'two-way 32+32 = 하드웨어 64. single_N64와 비교가 공정. <0이면 곱구조가 소자2배 넘는 이득'}
    print(f"\n[판정] two-way 32+32 {t32:+.2f} vs 단일 32 {s32:+.2f} ({t32-s32:+.2f}) vs 단일 64(동일HW) {s64:+.2f} ({t32-s64:+.2f})")
    report['runtime_sec'] = round(time.time() - t0, 1)
    json.dump(report, open(os.path.join(RESULTS, 'twoway_experiment.json'), 'w'), indent=2, ensure_ascii=False)
    print(f"총 {report['runtime_sec']} s → results/twoway_experiment.json")


if __name__ == '__main__':
    main()
