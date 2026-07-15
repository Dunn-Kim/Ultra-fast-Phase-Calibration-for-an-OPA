# 변수 대응 정밀 검증 — 이미지 수식의 어느 자리가 우리의 자유 변수인가
#
# ① 비등간격 : E(θ) = Σ_n A_n·exp(j(k·x_n·sinθ + φ_n))          변수 = x_n, φ_n
# ② 등간격   : x_n = (n−1)d  →  E(θ) = Σ_n A_n·exp(j(k(n−1)d·sinθ + φ_n))   변수 = φ_n
# ③ 단순화AF : A_n=A_0, φ_n=(n−1)Δφ  →  E(θ) = A_0·sin(N/2·ψ)/sin(1/2·ψ),  ψ = kd·sinθ+Δφ
#                                                                   변수 = Δφ (스칼라 1개)
#
# 우리 칩 = ②.  x_n 은 제작 확정(3µm), A_n=1 고정  →  유일한 자유변수 = φ_n (N−1개, φ_1=0 기준).
# 제작 오차 ε_n 이 실제 위상에 더해짐:  실제 = φ_n(인가) + ε_n.
#
# 주장(본 파일이 검증):
#   (A) ε=0 이고 φ_n=(n−1)Δφ 이면 ②는 정확히 ③의 닫힌형과 일치한다.
#   (B) ε≠0 이면 ③이 깨진다 (등차 위상 전제 붕괴) → 반드시 ②의 일반형을 써야 한다.
#   (C) 최적해는 닫힌형이다:  φ_n* = (n−1)Δφ − ε_n,  Δφ = −k·d·sinθ_0  →  |E(θ_0)| = N (전역최적).
#       즉 위상 자체엔 반복 최적화가 불필요. 유일한 문제는 ε 추정(=캘리브레이션).
#
# 실행: venv/bin/python variable_map.py
import math

import torch as th

DT = th.float64
LAM, N, D = 1.55, 32, 3.0            # 하드웨어 고정: λ, 채널 수, 등간격 피치
K = 2 * math.pi / LAM
A0 = 1.0                             # A_n = 1 (진폭 하드웨어 없음)


def E_general(x, phi, theta_deg):
    """① 일반형 (비등간격 포함). phi = 실제 위상(인가+오차)."""
    u = th.sin(th.deg2rad(th.as_tensor(theta_deg, dtype=DT)))
    ph = K * u.reshape(-1, 1) * x.reshape(1, -1) + phi.reshape(1, -1)
    return (A0 * th.exp(1j * ph)).sum(-1)


def E_closed_form(dphi, theta_deg):
    """③ 단순화 AF 닫힌형 — 등간격 + 균일진폭 + 등차위상 φ_n=(n−1)Δφ 일 때만 성립."""
    u = th.sin(th.deg2rad(th.as_tensor(theta_deg, dtype=DT)))
    psi = K * D * u + dphi                                  # ψ = kd·sinθ + Δφ
    num = th.sin(N / 2 * psi)
    den = th.sin(0.5 * psi)
    # ψ→0 (동상) 극한: N
    return A0 * th.where(den.abs() < 1e-12, th.full_like(num, float(N)), num / den)


def uniform_x():
    return th.arange(N, dtype=DT) * D                       # x_n = (n−1)d, x_1 = 0


def main():
    x = uniform_x()
    th_grid = th.linspace(-90, 90, 1801, dtype=DT)

    # (A) ε=0, 등차위상 → ② 일반형 == ③ 닫힌형 ?
    print('(A) ε=0, φ_n=(n−1)Δφ  →  ② 일반형 vs ③ 닫힌형')
    for th0 in (0.0, 10.0, 25.0):
        dphi = -K * D * math.sin(math.radians(th0))         # Δφ = −kd·sinθ₀ (조향)
        phi = th.arange(N, dtype=DT) * dphi                 # φ_n = (n−1)Δφ
        Eg = E_general(x, phi, th_grid).abs()
        Ec = E_closed_form(dphi, th_grid).abs()
        err = (Eg - Ec).abs().max().item()
        peak = th_grid[Eg.argmax()].item()
        print(f'   θ₀={th0:4.1f}° | Δφ={dphi:+7.4f} rad | 최대오차 {err:.2e} | 주엽 피크 {peak:+.2f}°'
              f' | |E(θ₀)|={E_general(x, phi, th0).abs().item():.4f} (=N이면 완전정렬)')
        assert err < 1e-8, '② 와 ③ 불일치 — 변수 대응 이해 오류'

    # (B) ε≠0 → ③ 붕괴
    print('\n(B) ε≠0 (제작오차) → ③ 닫힌형 전제(등차위상) 붕괴')
    th.manual_seed(0)
    eps = 2 * math.pi * th.rand(N, dtype=DT) - math.pi
    eps[0] = 0.0
    dphi = -K * D * math.sin(math.radians(0.0))             # θ₀=0 → Δφ=0
    phi_applied = th.arange(N, dtype=DT) * dphi             # 조향 램프만 (보정 없음)
    Eg = E_general(x, phi_applied + eps, th_grid).abs()      # 실제 = 인가 + ε
    Ec = E_closed_form(dphi, th_grid).abs()                 # ③ 이 예측하는 것
    print(f'   ② 실제 vs ③ 예측 최대오차 {(Eg - Ec).abs().max().item():.3f}  (0이 아니면 ③ 무효)')
    print(f'   |E(θ₀)| 실제 {E_general(x, phi_applied + eps, 0.0).abs().item():.3f} / 이상 {N}'
          f'  → 주엽 {100*(E_general(x, phi_applied+eps, 0.0).abs().item()/N)**2:.1f}%')

    # (C) 닫힌형 최적해 φ* = (n−1)Δφ − ε
    print('\n(C) 닫힌형 최적해  φ_n* = (n−1)Δφ − ε_n   (반복 최적화 불필요)')
    for th0 in (0.0, 15.0, 30.0):
        dphi = -K * D * math.sin(math.radians(th0))
        phi_star = th.arange(N, dtype=DT) * dphi - eps       # ← 최적해
        amp = E_general(x, phi_star + eps, th0).abs().item() # 실제 = φ* + ε
        print(f'   θ₀={th0:4.1f}° | |E(θ₀)| = {amp:.6f} / N={N}  → 복원율 {(amp/N)**2:.6f}')
        assert abs(amp - N) < 1e-8, '닫힌형 최적해가 |E|=N 을 못 냄'

    # (D) 전역최적성: 임의 위상 섭동은 반드시 |E| 를 낮춤
    print('\n(D) 전역최적성 확인 — φ* 주변 임의 섭동 1000회')
    dphi = 0.0
    phi_star = th.arange(N, dtype=DT) * dphi - eps
    base = E_general(x, phi_star + eps, 0.0).abs().item()
    worse = 0
    for t in range(1000):
        th.manual_seed(1000 + t)
        pert = 0.2 * th.randn(N, dtype=DT)
        pert[0] = 0
        a = E_general(x, phi_star + pert + eps, 0.0).abs().item()
        if a <= base + 1e-9:
            worse += 1
    print(f'   φ* 보다 나은 섭동: {1000 - worse}/1000  (0이면 φ*가 전역최적)')
    assert worse == 1000, 'φ* 가 전역최적이 아님'

    print('\n결론: 우리 칩(②)의 자유변수는 φ_n 뿐이고, 그 최적해는 닫힌형 φ*=(n−1)Δφ−ε_n 이다.')
    print('      → 위상에 반복 최적화는 불필요. 유일한 미지수는 ε_n → 문제는 최적화가 아니라 "추정".')


if __name__ == '__main__':
    main()
