# 빔포밍 최적화 — 정준 수식으로부터 재구성
#
# 수식 (이미지 그대로, A_n = 1):
#     E(θ) = Σ_{n=1..N} exp( j( k·x_n·sinθ + φ_n ) ),   k = 2π/λ
#   비등간격: 변수 = x_n AND φ_n      등간격(x_n=(n−1)d): 변수 = φ_n 만
#   기준 고정: x_1 = 0, φ_1 = 0  →  각각 N−1 개
#
# 목적: 빔포밍 = 주엽 강화 + 사이드로브(잔차 로브) 약화
#     max |E(θ₀)|²          (주엽)
#     min max_{θ∈SL} |E(θ)|²/|E(θ₀)|²   (사이드로브, 주엽 상대)
#
# ── 재구성의 핵심: 두 변수는 대등하지 않다 (전부 본 저장소에서 실측·검증됨) ──
#  정리1  임의의 x 에 대해  φ*_n = −k·x_n·sinθ₀  가 |E(θ₀)| = N (전역최적·유일).
#         (variable_map.py: 섭동 1000회 중 φ* 를 이긴 것 0)
#  정리2  φ=φ* 를 대입하면 E(u) = Σ exp( j·k·x_n·(u−u₀) )  → 사이드로브는 순전히 x 의 함수.
#  반례3  φ 를 자유변수로 풀면 사이드로브는 못 잡고 주엽만 붕괴 (등간격+자유위상 32개:
#         PSLL +1.54 dB 로 그대로인데 주엽효율 0.066)
#  ⇒ φ 는 탐색 대상이 아니라 닫힌형. 유일한 탐색 변수는 x.
#     등간격(변수=φ 뿐)은 따라서 최적화할 것이 없다 — 성능은 d 가 이미 결정.
#
# 런타임의 제작오차 ε_n 은 φ*_n − ε_n 로 흡수 (ε 추정 = calibrate_fast, 5프레임).
#
# 실행: venv/bin/python beamform_optimize.py
import json
import math
import os

import torch as th

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64
LAM, N = 1.55, 32
K = 2 * math.pi / LAM
D_MIN, D_MAX, D_UNI = 2.0, 5.0, 3.0          # 하드웨어 제약 (변경 금지)
W_EL = 1.0                                   # 소자폭 (고정)

U_TR = th.linspace(-1, 1, 4001, dtype=DT)    # 설계 격자
U_VAL = th.linspace(-1, 1, 40001, dtype=DT)  # 판정 격자
EF_TR = th.sinc(W_EL * U_TR / LAM)           # 소자인자 (x, φ 와 무관한 공통 포락선)
EF_VAL = th.sinc(W_EL * U_VAL / LAM)


# ── 변수 → 기하 ──────────────────────────────────────────────
def positions(s):
    """비등간격 변수 s (N−1) → x_n.  x_1 = 0 고정.  d_n ∈ [d_min,d_max] 구조 보장."""
    d = D_MIN + (D_MAX - D_MIN) * th.sigmoid(s)
    return th.cat([th.zeros(1, dtype=DT), th.cumsum(d, 0)])


def positions_uniform(d=D_UNI):
    """등간격: x_n = (n−1)d.  변수 없음 (d 는 하드웨어)."""
    return th.arange(N, dtype=DT) * d


def phi_star(x, u0):
    """정리1: 주엽 |E(θ₀)|=N 을 달성하는 닫힌형 위상.  φ_1 = 0 기준으로 정규화."""
    p = -K * x * u0
    return p - p[0]


# ── 수식 (정준형 그대로) ─────────────────────────────────────
def E(x, phi, u):
    """E(θ) = Σ_n exp(j(k·x_n·sinθ + φ_n)),  A_n = 1,  u = sinθ."""
    ph = K * u.reshape(-1, 1) * x.reshape(1, -1) + phi.reshape(1, -1)
    return th.exp(1j * ph).sum(-1)


def intensity(x, phi, u, ef):
    return (ef * E(x, phi, u).abs() / N) ** 2


# ── 목적 ────────────────────────────────────────────────────
def sidelobe_mask(x, u, u0):
    """주엽 가드밴드 밖 = 잔차 로브. 개구 L 로 폭 결정 (경계는 stop-gradient)."""
    L = (x[-1] - x[0]).detach()
    return (u - u0).abs() > 2 * LAM / L


def soft_psll(x, phi, u0, beta, u=U_TR, ef=EF_TR):
    """주엽 상대 사이드로브의 soft-max [dB]. 주엽으로 나누므로 '주엽 낮춰 속이기' 차단."""
    I = intensity(x, phi, u, ef)
    i0 = intensity(x, phi, th.tensor([u0], dtype=DT), th.sinc(th.tensor(W_EL * u0 / LAM, dtype=DT)))[0]
    m = sidelobe_mask(x, u, u0)
    D = 10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)
    return th.logsumexp(beta * D, 0) / beta


@th.no_grad()
def evaluate(x, angles_deg):
    """판정: 각 조향각에서 φ* 인가 후 주엽 강도와 하드 PSLL."""
    worst, mains = -math.inf, []
    for a in angles_deg:
        u0 = math.sin(math.radians(a))
        phi = phi_star(x, u0)
        I = intensity(x, phi, U_VAL, EF_VAL)
        ef0 = th.sinc(th.tensor(W_EL * u0 / LAM, dtype=DT))
        i0 = intensity(x, phi, th.tensor([u0], dtype=DT), ef0)[0]
        mains.append((E(x, phi, th.tensor([u0], dtype=DT)).abs()[0] / N).item() ** 2)  # 배열인자 효율
        m = sidelobe_mask(x, U_VAL, u0)
        worst = max(worst, (10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)).max().item())
    return worst, min(mains)


# ── 알고리즘 ────────────────────────────────────────────────
def optimize_nonuniform(angles_deg, restarts=8, epochs=800, seed=42):
    """
    ① 비등간격: 변수 = x_n (N−1).  φ 는 정리1의 닫힌형 → 탐색 안 함.
    목적 = 조향각 집합의 worst-case soft-PSLL (주엽은 φ* 가 이미 최대 보장).
    """
    best, best_v = None, math.inf
    s0 = math.log((D_UNI - D_MIN) / (D_MAX - D_UNI))
    for r in range(restarts):
        th.manual_seed(seed + r)
        s = (s0 + 0.5 * th.randn(N - 1, dtype=DT)).requires_grad_(True)
        opt = th.optim.Adam([s], lr=1e-2)
        for t in range(epochs):
            beta = 0.2 * (6.0 / 0.2) ** (t / epochs)      # soft→hard 어닐링
            opt.zero_grad()
            x = positions(s)
            per = [soft_psll(x, phi_star(x, math.sin(math.radians(a))),
                             math.sin(math.radians(a)), beta) for a in angles_deg]
            loss = th.logsumexp(2.0 * th.stack(per), 0) / 2.0   # 각도 worst-case
            loss.backward()
            opt.step()
        with th.no_grad():
            x = positions(s.detach())
        v, _ = evaluate(x, angles_deg)
        if v < best_v:
            best_v, best = v, x
    return best, best_v


def optimize_uniform(angles_deg, d=D_UNI):
    """
    ② 등간격: 변수 = φ_n 뿐.  정리1 에 의해 φ* 가 주엽 전역최적이고,
    반례3 에 의해 φ 로 사이드로브를 낮출 수 없다 → 최적화할 것이 없다.
    성능은 d 가 이미 결정 (하드웨어). 닫힌형 φ* 를 그대로 반환.
    """
    return positions_uniform(d)


def main():
    angles = [float(a) for a in range(-30, 31, 5)]
    rep = {}

    print('② 등간격 (변수 = φ 뿐) — φ* 닫힌형, 최적화 여지 없음')
    xu = optimize_uniform(angles)
    vu, mu = evaluate(xu, angles)
    rep['uniform'] = {'vars': 'φ_n only (N−1=31)', 'worst_psll_db': round(vu, 2),
                      'min_array_eff': round(mu, 4), 'note': 'd=3µm 가 성능을 이미 결정'}
    print(f'   worst PSLL {vu:+.2f} dB | 최저 배열인자효율 {mu:.4f}  (grating lobe — φ로 제거 불가)')

    print('\n① 비등간격 (변수 = x_n, φ는 닫힌형) — x 만 탐색')
    xn, vn = optimize_nonuniform(angles)
    _, mn = evaluate(xn, angles)
    d = xn[1:] - xn[:-1]
    rep['nonuniform'] = {'vars': 'x_n (N−1=31), φ_n = 닫힌형',
                         'worst_psll_db': round(vn, 2), 'min_array_eff': round(mn, 4),
                         'aperture_um': round((xn[-1] - xn[0]).item(), 2),
                         'd_min_um': round(d.min().item(), 3), 'd_max_um': round(d.max().item(), 3)}
    print(f'   worst PSLL {vn:+.2f} dB | 최저 배열인자효율 {mn:.4f} | 개구 {xn[-1]-xn[0]:.1f}µm'
          f' | 간격 [{d.min():.2f},{d.max():.2f}]µm')
    print(f'\n   → 비등간격이 등간격 대비 {vu - vn:+.2f} dB (사이드로브), 주엽은 양쪽 다 φ*가 최대 보장')

    rep['delta_db'] = round(vn - vu, 2)
    json.dump(rep, open(os.path.join(RESULTS, 'beamform_reconstructed.json'), 'w'),
              indent=2, ensure_ascii=False)
    print(f'\n→ results/beamform_reconstructed.json')


if __name__ == '__main__':
    main()
