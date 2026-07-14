# 최적화 너머 — 조향 강건 배치의 4개 대안 방법 벤치마크
#
# 핵심 항등식: 위상=해석 조향해일 때 AF(u)=Σe^{jk·x_n·(u−u0)} = AF(v), v=u−u0.
# 조향은 같은 |AF(v)| 위에서 가시창만 이동 → 조향각 집합 이산 샘플링(기존) 대신
# 확장창 v ∈ (Δ, 1+sin θ_max]에서 단일 패턴 PSLL을 직접 설계하면 각도 사이 구멍이
# 원리적으로 사라진다. EF는 u의 함수라 v-불변이 아님 → worst-case EF비 W(v)로 가중:
#   W(v) = max_{|u0|≤s_max, |u0+v|≤1} [EF(u0+v)/EF(u0)]²   (x와 무관, 사전계산)
#   D(v) = 10log10( W(v)·|AF(v)|²/N² )  → PSLL 상계(모든 조향각 커버)
#
# 방법:
#   M1 v-domain Adam   : 위 손실로 sigmoid 박스 간격 연속 최적화 (멀티스타트)
#   M2 CMA-ES          : 동일 목적(하드 max)을 gradient 없이 전역 탐색 (자체 구현)
#   M3 이산 좌표하강    : 간격을 0.05µm 격자 [2,5]µm로 이산화, 갭별 전수 스캔 반복
#   M4 결정론적 배열    : golden-ratio / chirp / prime 간격 (닫힌형, 최적화 0회) ± Adam 정제
# 평가: benchmark_common.evaluate_layout — 조향 스윕 ±30°(5°), 위상 polish, 40001 격자.
# 챔피언(다각도 Adam, worst −11.54 dB)과 직접 비교.
#
# 실행: venv/bin/python experiment_beyond.py
import json
import math
import os
import time

import torch as th

from benchmark_common import check_constraints, evaluate_layout, gaps_to_positions, make_model

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64
LAM, W_EL, N = 1.55, 1.0, 32
DMIN, DMAX = 2.0, 5.0
K = 2 * math.pi / LAM
S_MAX = math.sin(math.radians(30))
V_MAX = 1.0 + S_MAX


def precompute_W(n_v=3001, n_u0=401):
    # W(v): x와 무관한 worst-case EF 강도비. 실현 불가능한 v 구간은 -inf 처리.
    v = th.linspace(0.0, V_MAX, n_v, dtype=DT)
    u0 = th.linspace(-S_MAX, S_MAX, n_u0, dtype=DT)
    ef = lambda u: th.sinc(W_EL * u / LAM)
    u = u0.reshape(1, -1) + v.reshape(-1, 1)          # (v, u0)
    feas = u.abs() <= 1.0
    ratio = (ef(u) / ef(u0.reshape(1, -1))) ** 2
    ratio = th.where(feas, ratio, th.full_like(ratio, -math.inf))
    return v, ratio.max(dim=1).values                  # W(v), -inf if infeasible


V_GRID, W_V = precompute_W()


def design_psll(x, beta=None, guard_kappa=2.0):
    # v-domain 설계 목적: beta=None → 하드 max [dB], 아니면 soft LSE.
    af = th.exp(1j * (K * V_GRID.reshape(-1, 1) * x.reshape(1, -1))).sum(dim=1)
    l_ap = (x[-1] - x[0]).detach()
    mask = (V_GRID > guard_kappa * LAM / l_ap) & th.isfinite(W_V)
    d_db = 10 * th.log10(W_V[mask] * (af.abs()[mask] / N) ** 2 + 1e-12)
    if beta is None:
        return d_db.max()
    return th.logsumexp(beta * d_db, dim=0) / beta


def sig_gaps(s):
    return DMIN + (DMAX - DMIN) * th.sigmoid(s)


S0 = math.log((3.0 - DMIN) / (DMAX - 3.0))


# --- M1: v-domain Adam ---
def m1_vdomain_adam(restarts=16, epochs=800):
    best_x, best_val = None, math.inf
    for r in range(restarts):
        th.manual_seed(1000 + r)
        s = (S0 + 0.3 * th.randn(N - 1, dtype=DT)).requires_grad_(True)
        opt = th.optim.Adam([s], lr=1e-2)
        for t in range(epochs):
            beta = 0.2 * (2.0 / 0.2) ** (t / epochs)
            opt.zero_grad()
            loss = design_psll(gaps_to_positions(sig_gaps(s)), beta=beta)
            loss.backward()
            opt.step()
        with th.no_grad():
            x = gaps_to_positions(sig_gaps(s))
            val = design_psll(x).item()
        if val < best_val:
            best_val, best_x = val, x
    return best_x, best_val


# --- M2: CMA-ES (표준 구현, 목적 = 하드 v-domain PSLL) ---
def m2_cmaes(gens=400, seed=7):
    th.manual_seed(seed)
    n = N - 1
    lam = 20
    mu = lam // 2
    w = th.log(th.tensor(mu + 0.5)) - th.log(th.arange(1, mu + 1, dtype=DT))
    w = w / w.sum()
    mu_eff = 1.0 / (w ** 2).sum()
    c_sig = (mu_eff + 2) / (n + mu_eff + 5)
    d_sig = 1 + 2 * max(0.0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sig
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    m = th.full((n,), S0, dtype=DT)
    sig = 0.8
    C = th.eye(n, dtype=DT)
    p_sig = th.zeros(n, dtype=DT)
    p_c = th.zeros(n, dtype=DT)
    best_x, best_val = None, math.inf

    def f(s_vec):
        return design_psll(gaps_to_positions(sig_gaps(s_vec))).item()

    for g in range(gens):
        evals_D, evals_B = th.linalg.eigh(C)
        evals_D = evals_D.clamp(min=1e-12).sqrt()
        z = th.randn(lam, n, dtype=DT)
        y = z @ th.diag(evals_D) @ evals_B.T
        xs = m.reshape(1, -1) + sig * y
        vals = th.tensor([f(xs[i]) for i in range(lam)], dtype=DT)
        idx = th.argsort(vals)[:mu]
        if vals[idx[0]].item() < best_val:
            best_val = vals[idx[0]].item()
            best_x = gaps_to_positions(sig_gaps(xs[idx[0]]))
        y_w = (w.reshape(-1, 1) * y[idx]).sum(dim=0)
        m = m + sig * y_w
        C_inv_half = evals_B @ th.diag(1.0 / evals_D) @ evals_B.T
        p_sig = (1 - c_sig) * p_sig + math.sqrt(c_sig * (2 - c_sig) * mu_eff) * (C_inv_half @ y_w)
        h_sig = 1.0 if p_sig.norm() / math.sqrt(1 - (1 - c_sig) ** (2 * (g + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
        p_c = (1 - c_c) * p_c + h_sig * math.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
        rank_mu = sum(w[i] * th.outer(y[idx[i]], y[idx[i]]) for i in range(mu))
        C = (1 - c_1 - c_mu) * C + c_1 * (th.outer(p_c, p_c) + (1 - h_sig) * c_c * (2 - c_c) * C) + c_mu * rank_mu
        sig = sig * math.exp((c_sig / d_sig) * (p_sig.norm() / chi_n - 1))
        sig = min(sig, 5.0)
    return best_x, best_val


# --- M3: 이산 좌표하강 (갭별 전수 스캔, Gibbs 스타일) ---
def m3_coord_descent(init_d=None, grid_step=0.05, max_pass=20, seed=3):
    th.manual_seed(seed)
    cand = th.arange(DMIN, DMAX + 1e-9, grid_step, dtype=DT)
    d = init_d.clone() if init_d is not None else DMIN + (DMAX - DMIN) * th.rand(N - 1, dtype=DT)
    d = th.round(d / grid_step) * grid_step
    cur = design_psll(gaps_to_positions(d)).item()
    for p in range(max_pass):
        improved = False
        for i in th.randperm(N - 1).tolist():
            base = d[i].item()
            vals = []
            for c in cand:
                d[i] = c
                vals.append(design_psll(gaps_to_positions(d)).item())
            j = int(th.tensor(vals).argmin())
            if vals[j] < cur - 1e-6:
                d[i] = cand[j]
                cur = vals[j]
                improved = True
            else:
                d[i] = base
        if not improved:
            break
    return gaps_to_positions(d), cur


# --- M4: 결정론적 배열 ---
def m4_deterministic():
    out = {}
    phi_g = (1 + math.sqrt(5)) / 2
    frac = lambda z: z - math.floor(z)
    layouts = {
        'golden': th.tensor([DMIN + (DMAX - DMIN) * frac((n + 1) * phi_g) for n in range(N - 1)], dtype=DT),
        'chirp': th.linspace(DMIN, DMAX, N - 1, dtype=DT),
        'sym_chirp': th.cat([th.linspace(DMAX, DMIN, 16, dtype=DT),
                             th.linspace(DMIN, DMAX, 16, dtype=DT)[1:]]),
        'prime_mod': th.tensor([DMIN + (DMAX - DMIN) * ((p % 97) / 96.0) for p in
                                [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127]], dtype=DT),
    }
    for name, d in layouts.items():
        x = gaps_to_positions(d)
        out[name] = {'x': x, 'design_psll': design_psll(x).item()}
    return out


def refine_adam(x_init, epochs=400):
    # 임의 초기 배치에서 v-domain Adam 정제
    d0 = (x_init[1:] - x_init[:-1]).clamp(DMIN + 1e-4, DMAX - 1e-4)
    p = (d0 - DMIN) / (DMAX - DMIN)
    s = th.log(p / (1 - p)).requires_grad_(True)
    opt = th.optim.Adam([s], lr=5e-3)
    for t in range(epochs):
        beta = 0.5 * (2.0 / 0.5) ** (t / epochs)
        opt.zero_grad()
        loss = design_psll(gaps_to_positions(sig_gaps(s)), beta=beta)
        loss.backward()
        opt.step()
    with th.no_grad():
        x = gaps_to_positions(sig_gaps(s))
    return x, design_psll(x).item()


def main():
    os.makedirs(RESULTS, exist_ok=True)
    model = make_model()
    report = {'champion_ref': {'worst_psll_db': -11.54, 'note': '다각도 Adam (기존)'}}
    t0 = time.time()

    print('=== M4 결정론적 배열 (설계값만, 즉시) ===')
    det = m4_deterministic()
    for name, r in det.items():
        print(f"  {name:10s} design {r['design_psll']:+6.2f} dB")

    print('=== M1 v-domain Adam (멀티스타트 16) ===')
    x1, v1 = m1_vdomain_adam()
    print(f"  design {v1:+.2f} dB")

    print('=== M2 CMA-ES (400세대) ===')
    x2, v2 = m2_cmaes()
    print(f"  design {v2:+.2f} dB")

    print('=== M3 이산 좌표하강 (M1 해 초기화 + 랜덤 2회) ===')
    x3a, v3a = m3_coord_descent(init_d=(x1[1:] - x1[:-1]))
    x3b, v3b = m3_coord_descent(seed=11)
    x3, v3 = (x3a, v3a) if v3a <= v3b else (x3b, v3b)
    print(f"  design {v3:+.2f} dB (adam-init {v3a:+.2f} / random {v3b:+.2f})")

    print('=== M4+ 정제: golden 초기 Adam ===')
    x4, v4 = refine_adam(det['golden']['x'])
    print(f"  design {v4:+.2f} dB")

    print('\n=== 최종 벤치마크 (조향 스윕 ±30°, 위상 polish, 40001 격자) ===')
    entries = {
        'M1_vdomain_adam': x1, 'M2_cmaes': x2, 'M3_coord_descent': x3,
        'M4_golden_raw': det['golden']['x'], 'M4_golden_refined': x4,
        'M4_chirp_raw': det['chirp']['x'],
    }
    for name, x in entries.items():
        ok, st = check_constraints(x)
        r = evaluate_layout(x, model=model)
        r['constraints_ok'] = ok
        r['design_psll_db'] = round(design_psll(x).item(), 2)
        r['x_um'] = [round(v, 4) for v in x.tolist()]
        report[name] = r
        print(f"  {name:20s} worst {r['worst_psll_db']:+6.2f} dB | @0° {r['psll_at_0']:+6.2f} "
              f"| 개구 {r['aperture_um']:6.1f}µm | 제약 {ok}")

    report['runtime_sec'] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, 'beyond_experiment.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n총 {report['runtime_sec']} s → results/beyond_experiment.json")


if __name__ == '__main__':
    main()
