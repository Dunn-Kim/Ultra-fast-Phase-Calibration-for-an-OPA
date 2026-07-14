# DL 기반 최적화 실증 — 회귀 surrogate / 신경망 재파라미터화가 −11.62 dB 벽을 넘는가?
#
# 방법 A (회귀론/NN surrogate): 간격→worst PSLL 을 MLP 회귀로 학습(라벨 = polish 내장
#   fast 목적, 순위역전 교훈 반영) → surrogate 를 통과하는 gradient 로 간격 최적화
#   → 진짜 벤치마크로 검증. 표준 "learned surrogate inverse design" 패턴.
# 방법 B (딥러닝 재파라미터화, deep-prior): 간격 로짓 = MLP_θ(z) 과잉파라미터 생성기.
#   목적은 챔피언과 동일(다각도 soft-PSLL + 각도별 δφ) — DL 파라미터화가 더 나은
#   분지를 찾는가만 분리 검증.
# 판정: benchmark_common.evaluate_layout (±30° 스윕, polish, 40001 격자) — 챔피언과 동일.
#
# 실행: venv/bin/python experiment_dl.py
import json
import math
import os
import time

import torch as th

from benchmark_common import check_constraints, evaluate_layout, gaps_to_positions, make_model
from losses import multi_angle_worst_psll

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64
LAM, W_EL, N = 1.55, 1.0, 32
DMIN, DMAX = 2.0, 5.0
K = 2 * math.pi / LAM
ANGLES4 = (0.0, 10.0, 20.0, 30.0)          # ±대칭 → 반쪽만
U_FAST = th.linspace(-1.0, 1.0, 1201, dtype=DT)
EF2_FAST = th.sinc(W_EL * U_FAST / LAM) ** 2


def fast_polished_worst(x, T=15, lr=3e-2, beta=2.0):
    # polish 내장 fast 목적 (라벨/스크리닝용) — 순위역전 방지 위해 반드시 polish 포함
    worst = -math.inf
    guard = 2 * LAM / (x[-1] - x[0]).item()
    for a in ANGLES4:
        u0 = math.sin(math.radians(a))
        ef2_0 = math.sin(math.pi * W_EL * u0 / LAM) ** 2 / (math.pi * W_EL * u0 / LAM) ** 2 if u0 else 1.0
        v = U_FAST - u0
        mask = v.abs() > guard
        phase_geo = K * v.reshape(-1, 1) * x.reshape(1, -1)     # (M, N) 상수
        phi = th.remainder(K * x * u0, 2 * math.pi).clone().requires_grad_(True)
        opt = th.optim.Adam([phi], lr=lr)
        for _ in range(T):
            opt.zero_grad()
            af = th.exp(1j * (phase_geo - (phi - K * x * u0).reshape(1, -1))).sum(dim=1)
            i0_af = th.exp(1j * (-(phi - K * x * u0))).sum() .abs() ** 2
            rel = (EF2_FAST / ef2_0) * af.abs() ** 2 / (i0_af + 1e-12)
            d_db = 10 * th.log10(rel[mask] + 1e-12)
            loss = -10 * th.log10(i0_af / N ** 2 + 1e-12) + th.logsumexp(beta * d_db, 0) / beta
            loss.backward()
            opt.step()
        with th.no_grad():
            af = th.exp(1j * (phase_geo - (phi - K * x * u0).reshape(1, -1))).sum(dim=1)
            i0_af = th.exp(1j * (-(phi - K * x * u0))).sum().abs() ** 2
            rel = (EF2_FAST / ef2_0) * af.abs() ** 2 / (i0_af + 1e-12)
            p = 10 * th.log10(rel[mask].max() + 1e-12).item()
        worst = max(worst, p)
    return worst


def load_champions():
    import pandas as pd
    out = []
    for f in ('final_spacing_beyond.csv', 'final_spacing_multiangle.csv'):
        p = os.path.join(RESULTS, f)
        if os.path.exists(p):
            d = th.tensor(pd.read_csv(p)['d_um'].values, dtype=DT)
            out.append(d)
    return out


# ---------- 방법 A: 회귀 surrogate ----------
def method_a(n_uniform=3000, n_pert=3000, seed=42):
    th.manual_seed(seed)
    champs = load_champions()
    xs, ys = [], []
    samples = [DMIN + (DMAX - DMIN) * th.rand(N - 1, dtype=DT) for _ in range(n_uniform)]
    for i in range(n_pert):
        base = champs[i % len(champs)]
        samples.append((base + 0.2 * th.randn(N - 1, dtype=DT)).clamp(DMIN, DMAX))
    t0 = time.time()
    for i, d in enumerate(samples):
        xs.append(d)
        ys.append(fast_polished_worst(gaps_to_positions(d)))
        if (i + 1) % 1000 == 0:
            print(f"    라벨 {i+1}/{len(samples)} ({time.time()-t0:.0f}s)")
    X = (th.stack(xs) - DMIN) / (DMAX - DMIN)
    Y = th.tensor(ys, dtype=DT)
    y_mu, y_sd = Y.mean(), Y.std()
    Yn = (Y - y_mu) / y_sd
    n_tr = int(0.9 * len(Y))
    perm = th.randperm(len(Y))
    tr, va = perm[:n_tr], perm[n_tr:]

    mlp = th.nn.Sequential(
        th.nn.Linear(N - 1, 256), th.nn.SiLU(),
        th.nn.Linear(256, 256), th.nn.SiLU(),
        th.nn.Linear(256, 256), th.nn.SiLU(),
        th.nn.Linear(256, 1)).to(DT)
    opt = th.optim.Adam(mlp.parameters(), lr=1e-3)
    for ep in range(300):
        idx = tr[th.randperm(len(tr))]
        for b in range(0, len(idx), 512):
            j = idx[b:b + 512]
            opt.zero_grad()
            loss = ((mlp(X[j]).squeeze(-1) - Yn[j]) ** 2).mean()
            loss.backward()
            opt.step()
    with th.no_grad():
        pv = mlp(X[va]).squeeze(-1)
        r2 = 1 - ((pv - Yn[va]) ** 2).mean() / Yn[va].var()
        mae = ((pv - Yn[va]) * y_sd).abs().mean()
    print(f"    surrogate R²={r2:.3f}, val MAE={mae:.2f} dB")

    # surrogate 관통 최적화 (256 멀티스타트)
    s = th.randn(256, N - 1, dtype=DT).requires_grad_(True)
    opt2 = th.optim.Adam([s], lr=3e-2)
    for _ in range(400):
        opt2.zero_grad()
        mlp_in = th.sigmoid(s)                      # (0,1) = 정규화 간격
        pred = mlp(mlp_in).squeeze(-1)
        pred.sum().backward()
        opt2.step()
    with th.no_grad():
        d_all = DMIN + (DMAX - DMIN) * th.sigmoid(s)
        pred = mlp(th.sigmoid(s)).squeeze(-1) * y_sd + y_mu
    top = th.argsort(pred)[:20]
    scored = sorted(((fast_polished_worst(gaps_to_positions(d_all[i])), i.item()) for i in top))
    print(f"    [자유탐색] surrogate 예측 top: {pred[top[0]]:.2f} dB → 실제 top3 fast: "
          f"{[round(v,2) for v,_ in scored[:3]]}")
    exploit_gap = round(scored[0][0] - pred[top[0]].item(), 2)

    # trust-region 변형: 챔피언 ±0.3µm 박스(학습 분포 내부)로 제한 — 공정한 2차 시도
    champ = champs[0]
    lo = (champ - 0.3).clamp(DMIN, DMAX)
    hi = (champ + 0.3).clamp(DMIN, DMAX)
    s2 = th.randn(128, N - 1, dtype=DT).requires_grad_(True)
    opt3 = th.optim.Adam([s2], lr=3e-2)
    for _ in range(400):
        opt3.zero_grad()
        d2 = lo + (hi - lo) * th.sigmoid(s2)
        mlp(( d2 - DMIN) / (DMAX - DMIN)).squeeze(-1).sum().backward()
        opt3.step()
    with th.no_grad():
        d2_all = lo + (hi - lo) * th.sigmoid(s2)
        pred2 = mlp((d2_all - DMIN) / (DMAX - DMIN)).squeeze(-1) * y_sd + y_mu
    top2 = th.argsort(pred2)[:10]
    scored2 = sorted(((fast_polished_worst(gaps_to_positions(d2_all[i])), i.item()) for i in top2))
    print(f"    [trust-region] 예측 top: {pred2[top2[0]]:.2f} dB → 실제 top3 fast: "
          f"{[round(v,2) for v,_ in scored2[:3]]}")

    cands = [d_all[i] for _, i in scored[:2]] + [d2_all[i] for _, i in scored2[:2]]
    return cands, {'r2': round(r2.item(), 3), 'mae_db': round(mae.item(), 2),
                   'n_samples': len(samples), 'exploit_gap_db_free': exploit_gap}


# ---------- 방법 B: 신경망 재파라미터화 (deep prior) ----------
def method_b(model, restarts=8, epochs=1500):
    best = []
    n_ang = len(model.cfg.design_angles_deg)
    for r in range(restarts):
        th.manual_seed(500 + r)
        gen = th.nn.Sequential(
            th.nn.Linear(32, 128), th.nn.SiLU(),
            th.nn.Linear(128, 128), th.nn.SiLU(),
            th.nn.Linear(128, N - 1)).to(DT)
        z = th.randn(32, dtype=DT)
        dphi = th.zeros(n_ang, N, dtype=DT, requires_grad=True)
        opt = th.optim.Adam([{'params': gen.parameters(), 'lr': 2e-3},
                             {'params': [dphi], 'lr': 3e-3}])
        for t in range(epochs):
            beta = 0.2 * (2.0 / 0.2) ** (t / epochs)
            opt.zero_grad()
            s_out = gen(z)
            loss = multi_angle_worst_psll(model, s_out, beta,
                                          model.cfg.design_angles_deg, gamma=2.0, dphi=dphi)
            loss.backward()
            opt.step()
        with th.no_grad():
            d = model.gaps(gen(z)).detach()
        v = fast_polished_worst(gaps_to_positions(d))   # polish는 내부 grad 필요 — no_grad 밖
        best.append((v, d))
        print(f"    restart {r}: fast {v:+.2f} dB")
    best.sort(key=lambda z: z[0])
    return [d for _, d in best[:2]]


def main():
    model = make_model()
    t0 = time.time()
    report = {'reference': {'champion_bilevel_cma': -11.62, 'champion_adam': -11.54}}

    print('=== 방법 A: NN 회귀 surrogate (라벨 6000개 생성 → MLP → 관통 최적화) ===')
    cand_a, meta_a = method_a()
    report['A_surrogate_meta'] = meta_a

    print('=== 방법 B: 신경망 재파라미터화 (deep prior, 목적은 챔피언과 동일) ===')
    cand_b = method_b(model)

    print('\n=== 최종 벤치마크 (챔피언과 동일 프로토콜) ===')
    entries = {f'A_surrogate_{i}': d for i, d in enumerate(cand_a)}
    entries.update({f'B_deepprior_{i}': d for i, d in enumerate(cand_b)})
    for name, d in entries.items():
        x = gaps_to_positions(d)
        ok, _ = check_constraints(x)
        r = evaluate_layout(x, model=model)
        report[name] = {'worst_psll_db': r['worst_psll_db'], 'psll_at_0': r['psll_at_0'],
                        'aperture_um': r['aperture_um'], 'constraints_ok': ok,
                        'd_um': [round(v, 4) for v in d.tolist()]}
        print(f"  {name:16s} worst {r['worst_psll_db']:+6.2f} dB | @0° {r['psll_at_0']:+6.2f} | 제약 {ok}")

    report['runtime_sec'] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, 'dl_experiment.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n총 {report['runtime_sec']} s → results/dl_experiment.json")


if __name__ == '__main__':
    main()
