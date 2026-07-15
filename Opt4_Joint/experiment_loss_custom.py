# 커스텀 손실 — "주엽 높이면 수렴, 사이드로브 총합(전력) 강하면 벌점"
#
# 제안된 손실을 정식화하면 곧 주엽 전력비 η의 최대화:
#   η = ∫_ML I·J du / ∫_all I·J du      (J = 1/√(1−u²), 전력 적분 Jacobian)
#   L_eta = −10·log10(η)   ← 주엽↑ 보상 + 사이드로브 총전력↑ 벌점을 동시에 내포
# 기존 손실 L_PSLL = soft-max 사이드로브 '피크 비율' (총합 아님)
#
# 검증: (1) L_eta로 최적화하면 다른 해가 나오는가? (2) PSLL은 어떻게 되는가?
#       (3) 가중합 α·L_PSLL + (1−α)·L_eta 의 Pareto 전선
# 가설: η 최대화 = 앨리어싱 최소화 → 간격을 d_min으로 밀어 '등간격 2µm 회귀' 예상.
#       그렇다면 PSLL은 급격히 악화 → 두 목표 정면 충돌.
#
# 실행: venv/bin/python experiment_loss_custom.py
import json
import math
import os

import torch as th

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64
LAM, W_EL, N = 1.55, 1.0, 32
DMIN, DMAX = 2.0, 5.0
K = 2 * math.pi / LAM
ANG = [math.sin(math.radians(a)) for a in range(-30, 31, 10)]   # 설계각(속도 위해 7각)
U = th.linspace(-0.9999, 0.9999, 4001, dtype=DT)
JAC = 1.0 / th.sqrt(1 - U ** 2)
EF2 = th.sinc(W_EL * U / LAM) ** 2
UV = th.linspace(-0.9999, 0.9999, 40001, dtype=DT)
JACV = 1.0 / th.sqrt(1 - UV ** 2)
EF2V = th.sinc(W_EL * UV / LAM) ** 2


def gaps(s):
    return DMIN + (DMAX - DMIN) * th.sigmoid(s)


def positions(s):
    return th.cat([th.zeros(1, dtype=DT), th.cumsum(gaps(s), 0)])


def af2(x, u0, u=U):
    return th.exp(1j * (K * (u.reshape(-1, 1) - u0) * x.reshape(1, -1))).sum(1).abs() ** 2


def ef2_at(u0):
    return th.sinc(th.tensor(W_EL * u0 / LAM, dtype=DT)) ** 2


def eta_diff(x, u0):
    # 주엽 전력비 (미분가능). 마스크 경계의 L은 detach.
    I = EF2 * af2(x, u0)
    L = (x[-1] - x[0]).detach()
    ml = (U - u0).abs() <= LAM / L
    num = th.trapz((I * JAC)[ml], U[ml])
    den = th.trapz(I * JAC, U)
    return num / (den + 1e-30)


def soft_psll(x, u0, beta):
    I = EF2 * af2(x, u0) / N ** 2
    i0 = ef2_at(u0)
    L = (x[-1] - x[0]).detach()
    m = (U - u0).abs() > 2 * LAM / L
    D = 10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)
    return th.logsumexp(beta * D, 0) / beta


@th.no_grad()
def hard_metrics(x):
    worst, etas = -math.inf, []
    for u0 in ANG:
        I = EF2V * af2(x, u0, UV) / N ** 2
        i0 = ef2_at(u0)
        L = x[-1] - x[0]
        m = (UV - u0).abs() > 2 * LAM / L
        worst = max(worst, (10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)).max().item())
        Ip = EF2V * af2(x, u0, UV)
        ml = (UV - u0).abs() <= LAM / L
        etas.append((th.trapz((Ip * JACV)[ml], UV[ml]) / th.trapz(Ip * JACV, UV)).item())
    return worst, sum(etas) / len(etas)


def design(alpha, seeds=4, epochs=600, seed0=700):
    # L = α·(worst soft-PSLL) + (1−α)·(worst −10log10 η)
    best, best_obj = None, math.inf
    for r in range(seeds):
        th.manual_seed(seed0 + r)
        s = (math.log((3 - DMIN) / (DMAX - 3)) + 0.5 * th.randn(N - 1, dtype=DT)).requires_grad_(True)
        opt = th.optim.Adam([s], lr=1e-2)
        for t in range(epochs):
            beta = 0.2 * (6 / 0.2) ** (t / epochs)
            opt.zero_grad()
            x = positions(s)
            L = 0.0
            if alpha > 0:
                L = L + alpha * th.logsumexp(2.0 * th.stack([soft_psll(x, u0, beta) for u0 in ANG]), 0) / 2.0
            if alpha < 1:
                le = th.stack([-10 * th.log10(eta_diff(x, u0) + 1e-30) for u0 in ANG])
                L = L + (1 - alpha) * th.logsumexp(2.0 * le, 0) / 2.0
            L.backward(); opt.step()
        with th.no_grad():
            x = positions(s.detach())
            p, e = hard_metrics(x)
            obj = alpha * p + (1 - alpha) * (-10 * math.log10(e))
        if obj < best_obj:
            best_obj, best = obj, (x, p, e, gaps(s.detach()))
    return best


def main():
    rep = {}
    print('=== Pareto: α·L_PSLL + (1−α)·L_eta ===')
    print(f"{'α':>5} | {'PSLL[dB]':>9} | {'η':>6} | {'d̄[µm]':>7} | 간격분포")
    rows = []
    for a in (1.0, 0.75, 0.5, 0.25, 0.0):
        x, p, e, d = design(a)
        dbar = d.mean().item()
        spread = d.std().item()
        rows.append({'alpha': a, 'psll_db': round(p, 2), 'eta': round(e, 4),
                     'd_bar': round(dbar, 3), 'd_std': round(spread, 3),
                     'd_min': round(d.min().item(), 3), 'd_max': round(d.max().item(), 3)})
        tag = '등간격화' if spread < 0.15 else '비등간격'
        print(f"{a:>5.2f} | {p:>+9.2f} | {e:>6.3f} | {dbar:>7.3f} | std={spread:.2f} [{d.min():.2f},{d.max():.2f}] {tag}")
    rep['pareto'] = rows

    # 참조점
    print('\n=== 참조 ===')
    x_uni2 = th.arange(N, dtype=DT) * 2.0
    p2, e2 = hard_metrics(x_uni2)
    x_uni3 = th.arange(N, dtype=DT) * 3.0
    p3, e3 = hard_metrics(x_uni3)
    rep['ref'] = {'uniform_2um': {'psll': round(p2, 2), 'eta': round(e2, 4)},
                  'uniform_3um': {'psll': round(p3, 2), 'eta': round(e3, 4)}}
    print(f"  등간격 2µm: PSLL {p2:+.2f} dB | η {e2:.3f}")
    print(f"  등간격 3µm: PSLL {p3:+.2f} dB | η {e3:.3f}")

    json.dump(rep, open(os.path.join(RESULTS, 'loss_custom_pareto.json'), 'w'), indent=2, ensure_ascii=False)
    print('\n→ results/loss_custom_pareto.json')


if __name__ == '__main__':
    main()
