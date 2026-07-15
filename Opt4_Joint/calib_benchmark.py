# 위상 캘리브레이션 벤치마크 — "측정(카메라 프레임) 몇 회로 주엽을 되찾는가"
#
# 문제(재정렬): 송신 OPA, 간격 고정, A_n=1. 제작 칩에 미지 위상오차 ε_n (31 미지수).
#   제어 = 인가위상 φ_a.  실제위상 = φ_a + ε.  관측 = 카메라 강도.
#   목표: φ_a = −ε 복원 → |AF(u0)| = N.  **최소 프레임으로.**
#
# 측정 모델 2종:
#   'spot' : 프레임당 스칼라 1개 (I(u0)) — MATLAB REV/getSpot 이 실제로 쓰는 정보량
#   'full' : 프레임당 전체 패턴 I(u) 1801점 — 카메라가 실제로 주는 정보량
#   → REV는 프레임당 정보의 대부분을 버림. 'full'이면 프레임 수를 크게 줄일 수 있는가?
#
# 기준선: REV-5 = 5×31 = 155 프레임 (MATLAB 현행), REV-3 = 93 (코사인 3미지수 최소)
# 실행: venv/bin/python calib_benchmark.py
import json
import math
import os

import torch as th

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64
LAM, W_EL, N = 1.55, 1.0, 32
K = 2 * math.pi / LAM
U0 = 0.0                                    # 목표 조향각 (보어사이트)
THETA = th.linspace(-90, 90, 1801, dtype=DT)
U = th.sin(th.deg2rad(THETA))
EF = th.sinc(W_EL * U / LAM)


class Chip:
    """제작된 칩 시뮬레이터 — ε는 미지. 카메라 프레임만 반환. 프레임 수 카운트."""

    def __init__(self, x, eps, noise=0.01, seed=0):
        self.x = x
        self.eps = eps                      # 미지 (알고리즘이 접근 금지)
        self.noise = noise
        self.frames = 0
        self.g = th.Generator().manual_seed(seed)

    def _field(self, phi_a, u):
        tot = phi_a + self.eps              # 실제 위상
        ph = K * u.reshape(-1, 1) * self.x.reshape(1, -1) + tot.reshape(1, -1)
        return th.exp(1j * ph).sum(1)

    def shoot(self, phi_a, mode='full'):
        """카메라 프레임 1장 촬영."""
        self.frames += 1
        if mode == 'spot':
            u = th.tensor([U0], dtype=DT)
            I = (th.sinc(W_EL * u / LAM) * self._field(phi_a, u).abs() / N) ** 2
        else:
            I = (EF * self._field(phi_a, U).abs() / N) ** 2
        # 카메라 잡음 (상대 가우시안)
        I = I * (1 + self.noise * th.randn(I.shape, generator=self.g, dtype=DT))
        return I.clamp(min=0)


def main_lobe_recovery(x, phi_a, eps):
    """복원율 = |AF(u0)|²/N²  (1.0 = 완전 보정)"""
    tot = phi_a + eps
    af = th.exp(1j * (K * U0 * x + tot)).sum()
    return (af.abs() ** 2 / N ** 2).item()


# ---------- 기준선 1: REV (MATLAB 현행) ----------
def rev(chip, n_sample=5):
    """채널별로 위상 스캔 → 코사인 최소자승 피팅 → ε_n 추정. 프레임 = n_sample × (N−1)."""
    scan = th.linspace(-120, 120, n_sample, dtype=DT) * math.pi / 180
    phi = th.zeros(N, dtype=DT)
    for ch in range(1, N):                  # ch 0 = 기준
        Is = []
        for s in scan:
            trial = phi.clone()
            trial[ch] = phi[ch] + s
            Is.append(chip.shoot(trial, 'spot').item())
        # I(s) = y0 + y1·cos(s − x0)  →  선형기저 [1, cos s, sin s] 최소자승
        #   c1 = y1·cos x0, c2 = y1·sin x0  →  최대점 s* = x0 = atan2(c2, c1)
        #   해당 채널을 나머지 필드와 정렬시키려면 그 최대점만큼 '더한다' (MATLAB REV_method 동일)
        A = th.stack([th.ones(n_sample, dtype=DT), th.cos(scan), th.sin(scan)], 1)
        c = th.linalg.lstsq(A, th.tensor(Is, dtype=DT)).solution
        phi[ch] = phi[ch] + math.atan2(c[2].item(), c[1].item())   # 즉시 적용(좌표상승)
    return phi


# ---------- 기준선 2: SPGD (적응광학 표준) ----------
def spgd(chip, iters=150, delta=0.3, lr=10.0):
    """동시 랜덤 섭동 ±δ → 2 프레임/iter. lr은 게인 스윕(1~3000)으로 공정 튜닝한 최적값=10."""
    phi = th.zeros(N, dtype=DT)
    g = th.Generator().manual_seed(1)
    for t in range(iters):
        d = delta * (th.randint(0, 2, (N,), generator=g, dtype=DT) * 2 - 1)
        d[0] = 0
        Ip = chip.shoot(phi + d, 'spot').item()
        Im = chip.shoot(phi - d, 'spot').item()
        phi = phi + lr * (Ip - Im) * d      # 강도 최대화 방향
    return phi


# ---------- 후보: 전체 패턴 + 수식 loss + Adam (surrogate) ----------
def full_pattern_adam(chip, frames=40, inner=60, lr=0.15, seed=0):
    """
    프레임 1장 = 전체 패턴(1801점). 모델의 미지 ε̂ 를 '측정 패턴에 맞도록' 회귀(수식 기반 loss),
    그 후 φ_a = −ε̂ 인가. 프레임마다 랜덤 프로브 위상을 걸어 정보 다양화.
    """
    th.manual_seed(seed)
    eps_hat = th.zeros(N, dtype=DT, requires_grad=True)
    probes, obs = [], []
    opt = th.optim.Adam([eps_hat], lr=lr)
    for f in range(frames):
        # 프로브: 첫 프레임은 0, 이후 랜덤 (정보 다양성)
        p = th.zeros(N, dtype=DT) if f == 0 else (2 * math.pi * th.rand(N, dtype=DT) - math.pi)
        p[0] = 0
        probes.append(p)
        obs.append(chip.shoot(p, 'full'))
        # 수집된 전 프레임에 대해 모델 적합 (수식 기반 loss)
        for _ in range(inner):
            opt.zero_grad()
            loss = 0.0
            for pp, oo in zip(probes, obs):
                ph = K * U.reshape(-1, 1) * chip.x.reshape(1, -1) + (pp + eps_hat).reshape(1, -1)
                Im = (EF * th.exp(1j * ph).sum(1).abs() / N) ** 2
                loss = loss + ((Im - oo) ** 2).mean()
            loss.backward()
            opt.step()
    return -eps_hat.detach()


def run(alg, chip_factory, **kw):
    chip = chip_factory()
    phi = alg(chip, **kw)
    return chip.frames, main_lobe_recovery(chip.x, phi, chip.eps)


def main():
    import pandas as pd
    # 배열: 원본 등간격 3µm (포스터 조건)
    x_uni = th.arange(N, dtype=DT) * 3.0
    rep = {}

    print('=== 위상 캘리브레이션: 프레임 수 vs 주엽 복원율 (N=32, 미지 ε~U(−π,π), 잡음 1%) ===')
    trials = 5
    for name, alg, kw in (
        ('REV-5 (MATLAB 현행)', rev, {'n_sample': 5}),
        ('REV-3 (최소 코사인)', rev, {'n_sample': 3}),
        ('SPGD 150iter', spgd, {'iters': 150}),
        ('SPGD 50iter', spgd, {'iters': 50}),
        ('전체패턴+Adam 8프레임', full_pattern_adam, {'frames': 8}),
        ('전체패턴+Adam 4프레임', full_pattern_adam, {'frames': 4}),
        ('전체패턴+Adam 2프레임', full_pattern_adam, {'frames': 2}),
    ):
        fs, rs = [], []
        for t in range(trials):
            th.manual_seed(100 + t)
            eps = 2 * math.pi * th.rand(N, dtype=DT) - math.pi
            eps[0] = 0
            f, r = run(alg, lambda: Chip(x_uni, eps.clone(), seed=t), **kw)
            fs.append(f); rs.append(r)
        rep[name] = {'frames': fs[0], 'recovery_mean': round(sum(rs) / len(rs), 4),
                     'recovery_min': round(min(rs), 4)}
        print(f"  {name:24s} | 프레임 {fs[0]:4d} | 복원율 평균 {sum(rs)/len(rs):.3f} 최저 {min(rs):.3f}")

    json.dump(rep, open(os.path.join(RESULTS, 'calib_benchmark.json'), 'w'), indent=2, ensure_ascii=False)
    print('\n→ results/calib_benchmark.json')


if __name__ == '__main__':
    main()
