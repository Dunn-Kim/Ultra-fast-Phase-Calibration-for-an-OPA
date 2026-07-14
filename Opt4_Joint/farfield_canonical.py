# 정준형 far-field 수식 (피드백 반영) — 교과서/레퍼런스 형태
#
#   비등간격(nonuniform):  E(θ) = Σ_{n} A_n·exp(j(k·x_n·sinθ + φ_n)),  A_n = 1
#     변수: 채널위치 x_n  AND  위상 φ_n
#   등간격(uniform):        E(θ) = Σ_{n} A_n·exp(j(k·(n−1)·d·sinθ + φ_n)),  A_n = 1, x_n=(n−1)d 고정
#     변수: 위상 φ_n 만 (간격 d 고정)
#
# 규약: u = sinθ, k = 2π/λ. 강도 I(θ) = |E(θ)|². PSLL = 점별 강도비이므로 A_0 등 전역상수 소거.
#
# 기존 model.py는 exp(j(k·x·u − φ)) (−φ) 규약 → 본 정준형은 +φ.
# |E|는 φ↔−φ에 불변이므로(verify_equivalence 참조) 두 규약의 모든 PSLL 결과가 동일.
# A_n = 1은 균일진폭 가정과 일치(진폭 하드웨어 없음).
import math
import torch as th

LAM = 1.55
K = 2.0 * math.pi / LAM


def field_nonuniform(x, phi, u):
    # E(u) = Σ_n exp(j(k·x_n·u + φ_n)),  A_n=1.  변수 = (x, phi)
    u_ = u.reshape(-1, 1) if u.dim() else u.reshape(1, 1)
    return th.exp(1j * (K * u_ * x.reshape(1, -1) + phi.reshape(1, -1))).sum(dim=1)


def field_uniform(d, phi, u, N):
    # E(u) = Σ_n exp(j(k·(n−1)·d·u + φ_n)),  A_n=1, x_n=(n−1)d 고정.  변수 = (phi)
    n = th.arange(N, dtype=phi.dtype, device=phi.device)  # n−1 → 0..N-1 (x_0=0)
    x = n * d
    return field_nonuniform(x, phi, u)


def intensity(field):
    return field.abs() ** 2


def steering_phase_canonical(x, u0):
    # +φ 규약의 조향해: φ_n = −k·x_n·u0  →  exp(j(k·x_n·u + φ_n)) = exp(j·k·x_n·(u−u0)),  |E(u0)|=N
    return th.remainder(-K * x.detach() * u0, 2.0 * math.pi)
