# 문헌 선정 모델 구현 (도메인 조사 워크플로 판정)
#   M-A: SLP-minimax 폴리시 — You et al., IEEE AWPL 16:3126 (2017) 계보 이식.
#        위치 1차 테일러 전개 + 신뢰영역 LP로 하드 minimax를 직접 하강.
#        간격 제약 2≤d≤5µm를 페널티가 아닌 '선형 부등식'으로 정확히 부과.
#   M-B: 대량 랜덤 시딩 — Yu/Wu/Yi, arXiv:2402.10928 (2024) 셔플 계보.
#        uniform / Fisher-Yates 셔플(개구 고정) / 밀도 테이퍼 30k 후보
#        → v-domain 스크리닝 → top-K Adam 정제 → SLP 폴리시.
# 목적: 13각(±30°, 5°) 하드 worst PSLL (해석 조향해, EF 정확) — 벤치마크와 동일 구조.
# 실행: venv/bin/python experiment_slp.py
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch as th
from scipy.optimize import linprog

from benchmark_common import check_constraints, evaluate_layout, gaps_to_positions, make_model
from config import JointConfig
from main_design_multiangle import run_multiangle_single

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64
LAM, W_EL = 1.55, 1.0
DMIN, DMAX = 2.0, 5.0
K = 2 * math.pi / LAM
ANGLES = [math.sin(math.radians(a)) for a in range(-30, 31, 5)]
U = th.linspace(-1.0, 1.0, 4001, dtype=DT)
EF2 = th.sinc(W_EL * U / LAM) ** 2


def hard_terms(x):
    # 각도별 상대 사이드로브 D[dB]와 해석 gradient ∂D/∂x — 전부 벡터화
    N = x.numel()
    guard = 2 * LAM / (x[-1] - x[0]).item()
    Ds, Gs = [], []
    for u0 in ANGLES:
        ef2_0 = th.sinc(th.tensor(W_EL * u0 / LAM, dtype=DT)) ** 2
        v = U - u0
        mask = v.abs() > guard
        ph = K * v[mask].reshape(-1, 1) * x.reshape(1, -1)      # (M, N)
        e = th.exp(1j * ph)
        af = e.sum(dim=1)
        af2 = (af.abs() ** 2).clamp(min=1e-300)
        rel = (EF2[mask] / ef2_0) * af2 / N ** 2
        D = 10 * th.log10(rel + 1e-15)
        # ∂|AF|²/∂x_n = −2k·v·Im[conj(AF)·e^{jk v x_n}] ;  dD/dx = (10/ln10)·(∂|AF|²/∂x)/|AF|²
        dAF2 = -2 * K * v[mask].reshape(-1, 1) * (af.conj().reshape(-1, 1) * e).imag
        G = (10 / math.log(10)) * dAF2 / af2.reshape(-1, 1)
        Ds.append(D)
        Gs.append(G)
    return th.cat(Ds), th.cat(Gs)


def hard_worst(x):
    D, _ = hard_terms(x)
    return D.max().item()


def slp_polish(x0, rho0=0.05, rho_max=0.1, iters=60, active_db=5.0):
    x = x0.clone()
    cur = hard_worst(x)
    rho = rho0
    N = x.numel()
    for it in range(iters):
        D, G = hard_terms(x)
        act = D >= D.max() - active_db
        Da, Ga = D[act].numpy(), G[act].numpy()
        m = Da.shape[0]
        # 변수 [δx_1..δx_{N-1}, t]  (δx_0 = 0 고정)
        nv = N - 1 + 1
        c = np.zeros(nv); c[-1] = 1.0
        A_ub = np.zeros((m + 2 * (N - 1), nv))
        b_ub = np.zeros(m + 2 * (N - 1))
        A_ub[:m, :N - 1] = Ga[:, 1:]
        A_ub[:m, -1] = -1.0
        b_ub[:m] = -Da
        d_now = (x[1:] - x[:-1]).numpy()
        # 간격 상한: (δx_{n+1} − δx_n) ≤ 5 − d_n ; 하한: −(δx_{n+1} − δx_n) ≤ d_n − 2
        Ddiff = np.zeros((N - 1, N - 1))
        for n in range(N - 1):
            Ddiff[n, n] = 1.0
            if n > 0:
                Ddiff[n, n - 1] = -1.0
        A_ub[m:m + N - 1, :N - 1] = Ddiff
        b_ub[m:m + N - 1] = DMAX - d_now
        A_ub[m + N - 1:, :N - 1] = -Ddiff
        b_ub[m + N - 1:] = d_now - DMIN
        r = linprog(c, A_ub=A_ub, b_ub=b_ub,
                    bounds=[(-rho, rho)] * (N - 1) + [(None, None)], method='highs')
        if not r.success:
            rho /= 2
            if rho < 1e-3: break
            continue
        dx = th.zeros(N, dtype=DT)
        dx[1:] = th.tensor(r.x[:N - 1], dtype=DT)
        cand = hard_worst(x + dx)
        if cand < cur - 1e-4:
            x, cur = x + dx, cand
            rho = min(rho * 1.5, rho_max)
        else:
            rho /= 2
            if rho < 1e-3: break
    return x, cur


def vdomain_screen_batch(D_gaps):
    # Yu 2024식 대량 스크리닝: v-domain W(v) 상계 하드 PSLL (B, ) — 순위용
    from experiment_beyond import V_GRID, W_V
    B = D_gaps.shape[0]
    x = th.cat([th.zeros(B, 1, dtype=DT), th.cumsum(D_gaps, dim=1)], dim=1)
    out = th.empty(B, dtype=DT)
    for i in range(0, B, 512):
        xb = x[i:i + 512]
        af = th.exp(1j * K * V_GRID.reshape(1, -1, 1) * xb.reshape(xb.shape[0], 1, -1)).sum(-1)
        l_ap = xb[:, -1] - xb[:, 0]
        for j in range(xb.shape[0]):
            mask = (V_GRID > 2 * LAM / l_ap[j]) & th.isfinite(W_V)
            out[i + j] = (10 * th.log10(W_V[mask] * (af[j, mask].abs() / 32) ** 2 + 1e-12)).max()
    return out


def mass_seeds(n_each=10000, seed=99):
    th.manual_seed(seed)
    pool = [DMIN + (DMAX - DMIN) * th.rand(n_each, 31, dtype=DT)]                 # uniform
    base = th.linspace(DMIN, DMAX, 31, dtype=DT)                                   # 셔플(개구 고정)
    pool.append(th.stack([base[th.randperm(31)] for _ in range(n_each)]))
    n = th.arange(31, dtype=DT)                                                    # 밀도 테이퍼 + 지터
    tapers = []
    for _ in range(n_each):
        p = 0.5 + 2.5 * th.rand(1)
        c = th.rand(1)
        prof = c * (n / 30) ** p + (1 - c) * (1 - n / 30) ** p
        d = DMIN + (DMAX - DMIN) * prof + 0.3 * th.randn(31)
        tapers.append(d.clamp(DMIN, DMAX))
    pool.append(th.stack(tapers))
    return th.cat(pool)


def main():
    model = make_model()
    t0 = time.time()
    report = {'reference': {'lossfix_champion': -11.75}}

    # --- M-A: 챔피언 SLP 폴리시 ---
    d_champ = th.tensor(pd.read_csv(os.path.join(RESULTS, 'final_spacing_lossfix.csv'))['d_um'].values, dtype=DT)
    x_champ = gaps_to_positions(d_champ)
    print(f"[M-A] 챔피언 hard obj {hard_worst(x_champ):+.2f} dB → SLP...")
    x_a, v_a = slp_polish(x_champ)
    print(f"[M-A] SLP 후 {v_a:+.2f} dB ({time.time()-t0:.0f}s)")

    # --- M-B: 대량 시딩 → 스크리닝 → Adam 정제 → SLP ---
    seeds = mass_seeds()
    scr = vdomain_screen_batch(seeds)
    top = th.argsort(scr)[:16]
    print(f"[M-B] 30k 스크리닝 완료: top {scr[top[0]]:+.2f} dB (v-domain 상계) ({time.time()-t0:.0f}s)")
    DENSE = tuple(float(a) for a in range(-30, 31, 5))
    best_b, xb_best = math.inf, None
    for i, idx in enumerate(top[:6]):
        d0 = seeds[idx]
        p = ((d0 - DMIN) / (DMAX - DMIN)).clamp(1e-3, 1 - 1e-3)
        cfg = JointConfig(restarts=1, epochs_joint=600, design_angles_deg=DENSE,
                          beta_end=6.0, angle_agg_gamma=2.0)
        th.manual_seed(int(idx))
        m2 = make_model()
        s = th.log(p / (1 - p)).clone().requires_grad_(True)
        dphi = th.zeros(len(DENSE), 32, dtype=DT, requires_grad=True)
        from losses import beta_schedule, multi_angle_worst_psll
        opt_s = th.optim.Adam([s], lr=cfg.lr_spacing)
        opt_p = th.optim.Adam([dphi], lr=cfg.lr_phase)
        for t in range(cfg.epochs_joint):
            beta = beta_schedule(t, cfg.epochs_joint, cfg.beta_start, cfg.beta_end)
            opt_s.zero_grad(); opt_p.zero_grad()
            loss = multi_angle_worst_psll(m2, s, beta, DENSE, 2.0, dphi=dphi)
            loss.backward(); opt_s.step(); opt_p.step()
        x_r = m2.positions(s.detach())
        x_r, v_r = slp_polish(x_r)
        print(f"[M-B] seed {i}: 정제+SLP {v_r:+.2f} dB")
        if v_r < best_b:
            best_b, xb_best = v_r, x_r

    # --- 최종 벤치마크 (5nm 스냅 포함) ---
    print('\n=== 최종 벤치마크 ===')
    for name, x in (('SLP_champion', x_a), ('massSeed_SLP', xb_best)):
        d = x[1:] - x[:-1]
        d = (th.round(d / 0.005) * 0.005).clamp(DMIN, DMAX)
        xs = gaps_to_positions(d)
        ok, _ = check_constraints(xs)
        r = evaluate_layout(xs, model=model)
        report[name] = {'worst_psll_db': r['worst_psll_db'], 'psll_at_0': r['psll_at_0'],
                        'aperture_um': r['aperture_um'], 'constraints_ok': ok,
                        'd_um': [round(v, 4) for v in d.tolist()]}
        print(f"  {name:16s} worst {r['worst_psll_db']:+6.2f} dB | @0° {r['psll_at_0']:+6.2f} | 제약 {ok}")

    report['runtime_sec'] = round(time.time() - t0, 1)
    json.dump(report, open(os.path.join(RESULTS, 'slp_experiment.json'), 'w'),
              indent=2, ensure_ascii=False)
    print(f"\n총 {report['runtime_sec']} s → results/slp_experiment.json")


if __name__ == '__main__':
    main()
