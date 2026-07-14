# M5 하이브리드 — 정확한 목적함수 × 전역 탐색
#
# beyond_experiment 교훈: v-domain W(v) 상계는 ~2 dB 헐거움 → 상계 최적화는 실측에서 짐.
# 해법: 벤치마크와 동일한 이산 각도(±30°, 5°) hard PSLL을 목적으로 직접 사용
# (위상 = 해석 조향해, EF 각도별 정확, polish만 미모델링 — 균일 오프셋이라 순위 보존).
#   M5a: 챔피언 배치 초기화 + 이산 좌표하강 (0.025µm 격자)
#   M5b: CMA-ES(챔피언 로짓 중심, σ=0.3) — 국소 분지 탈출 시도
#   M5c: M1(v-domain 해) 초기화 + 좌표하강 — 다른 분지에서 출발
# 실행: venv/bin/python experiment_exact.py
import json
import math
import os
import time

import pandas as pd
import torch as th

from benchmark_common import SWEEP_DEG, check_constraints, evaluate_layout, gaps_to_positions, make_model

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64
LAM, W_EL, N = 1.55, 1.0, 32
DMIN, DMAX = 2.0, 5.0
K = 2 * math.pi / LAM

U = th.linspace(-1.0, 1.0, 4001, dtype=DT)
EF2 = th.sinc(W_EL * U / LAM) ** 2
U0S = [math.sin(math.radians(a)) for a in SWEEP_DEG]
EF2_U0 = [th.sinc(th.tensor(W_EL * u0 / LAM, dtype=DT)) ** 2 for u0 in U0S]


def exact_worst_psll(x):
    # 벤치마크 프로토콜과 동일 구조(polish 제외): max over 13 angles of hard PSLL
    l_ap = x[-1] - x[0]
    guard = 2 * LAM / l_ap
    worst = -math.inf
    for u0, ef2_0 in zip(U0S, EF2_U0):
        v = U - u0
        af = th.exp(1j * (K * v.reshape(-1, 1) * x.reshape(1, -1))).sum(dim=1)
        rel = (EF2 / ef2_0) * (af.abs() / N) ** 2
        mask = v.abs() > guard
        p = 10 * th.log10(rel[mask].max() + 1e-12).item()
        if p > worst:
            worst = p
    return worst


def champion_x():
    d = th.tensor(pd.read_csv(os.path.join(RESULTS, 'final_spacing_multiangle.csv'))['d_um'].values, dtype=DT)
    return gaps_to_positions(d)


def coord_descent(x0, grid_step=0.025, max_pass=15):
    cand = th.arange(DMIN, DMAX + 1e-9, grid_step, dtype=DT)
    d = (x0[1:] - x0[:-1]).clone()
    d = th.round(d / grid_step) * grid_step
    d = d.clamp(DMIN, DMAX)
    cur = exact_worst_psll(gaps_to_positions(d))
    for p in range(max_pass):
        improved = False
        for i in range(N - 1):
            base = d[i].item()
            best_v, best_c = cur, None
            for c in cand:
                d[i] = c
                val = exact_worst_psll(gaps_to_positions(d))
                if val < best_v - 1e-4:
                    best_v, best_c = val, c.item()
            d[i] = best_c if best_c is not None else base
            if best_c is not None:
                cur = best_v
                improved = True
        print(f"    pass {p}: {cur:+.2f} dB")
        if not improved:
            break
    return gaps_to_positions(d), cur


def cmaes_exact(x0, gens=250, sigma0=0.3, seed=17):
    th.manual_seed(seed)
    n = N - 1
    d0 = (x0[1:] - x0[:-1]).clamp(DMIN + 1e-3, DMAX - 1e-3)
    p0 = (d0 - DMIN) / (DMAX - DMIN)
    m = th.log(p0 / (1 - p0))
    lam, mu = 20, 10
    w = th.log(th.tensor(mu + 0.5)) - th.log(th.arange(1, mu + 1, dtype=DT))
    w = w / w.sum()
    mu_eff = 1.0 / (w ** 2).sum()
    c_sig = (mu_eff + 2) / (n + mu_eff + 5)
    d_sig = 1 + c_sig
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    sig = sigma0
    C = th.eye(n, dtype=DT)
    p_sig = th.zeros(n, dtype=DT)
    p_c = th.zeros(n, dtype=DT)
    gaps = lambda s: DMIN + (DMAX - DMIN) * th.sigmoid(s)
    best_x, best_val = None, math.inf
    for g in range(gens):
        ev, B = th.linalg.eigh(C)
        Dg = ev.clamp(min=1e-12).sqrt()
        z = th.randn(lam, n, dtype=DT)
        y = z @ th.diag(Dg) @ B.T
        xs = m.reshape(1, -1) + sig * y
        vals = th.tensor([exact_worst_psll(gaps_to_positions(gaps(xs[i]))) for i in range(lam)], dtype=DT)
        idx = th.argsort(vals)[:mu]
        if vals[idx[0]].item() < best_val:
            best_val = vals[idx[0]].item()
            best_x = gaps_to_positions(gaps(xs[idx[0]]))
        y_w = (w.reshape(-1, 1) * y[idx]).sum(dim=0)
        m = m + sig * y_w
        Cih = B @ th.diag(1.0 / Dg) @ B.T
        p_sig = (1 - c_sig) * p_sig + math.sqrt(c_sig * (2 - c_sig) * mu_eff) * (Cih @ y_w)
        h = 1.0 if p_sig.norm() / math.sqrt(1 - (1 - c_sig) ** (2 * (g + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
        p_c = (1 - c_c) * p_c + h * math.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
        rmu = sum(w[i] * th.outer(y[idx[i]], y[idx[i]]) for i in range(mu))
        C = (1 - c_1 - c_mu) * C + c_1 * (th.outer(p_c, p_c) + (1 - h) * c_c * (2 - c_c) * C) + c_mu * rmu
        sig = min(sig * math.exp((c_sig / d_sig) * (p_sig.norm() / chi_n - 1)), 2.0)
    return best_x, best_val


def main():
    model = make_model()
    t0 = time.time()
    xc = champion_x()
    base = exact_worst_psll(xc)
    print(f"챔피언 exact 목적값: {base:+.2f} dB (실측 worst −11.54, polish 미포함이라 약간 높음)")
    report = {'champion_exact_obj': round(base, 2)}

    print('=== M5a 좌표하강 (챔피언 초기화) ===')
    x5a, v5a = coord_descent(xc)
    print('=== M5b CMA-ES (챔피언 중심) ===')
    x5b, v5b = cmaes_exact(xc)
    print(f"    {v5b:+.2f} dB")

    try:
        prev = json.load(open(os.path.join(RESULTS, 'beyond_experiment.json')))
        x1 = th.tensor(prev['M1_vdomain_adam']['x_um'], dtype=DT)
        print('=== M5c 좌표하강 (M1 v-domain 해 초기화) ===')
        x5c, v5c = coord_descent(x1)
    except Exception as e:
        x5c, v5c = None, math.inf
        print('  M1 해 로드 실패:', e)

    print('\n=== 최종 벤치마크 (polish 포함, 챔피언과 동일 프로토콜) ===')
    entries = {'M5a_cd_champion': x5a, 'M5b_cmaes_champion': x5b}
    if x5c is not None:
        entries['M5c_cd_vdomain'] = x5c
    for name, x in entries.items():
        ok, _ = check_constraints(x)
        r = evaluate_layout(x, model=model)
        r['constraints_ok'] = ok
        r['exact_obj_db'] = round(exact_worst_psll(x), 2)
        r['x_um'] = [round(v, 4) for v in x.tolist()]
        report[name] = r
        print(f"  {name:20s} worst {r['worst_psll_db']:+6.2f} dB | @0° {r['psll_at_0']:+6.2f} | 제약 {ok}")

    report['runtime_sec'] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, 'exact_experiment.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n총 {report['runtime_sec']} s → results/exact_experiment.json")


if __name__ == '__main__':
    main()
