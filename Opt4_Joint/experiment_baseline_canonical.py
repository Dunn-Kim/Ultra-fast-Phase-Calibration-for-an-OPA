# 비교군 수식 대체 — 정준형(피드백)으로 재정의·재실행
#
#   등간격(uniform):    x_n=(n−1)·d 고정, 변수 = φ_n 만.      A_n=1.  E=Σexp(j(k(n−1)d·u+φ_n))
#   비등간격(nonuniform): 변수 = x_n AND φ_n.                A_n=1.  E=Σexp(j(k·x_n·u+φ_n))
#   조향 ±30°(5°,13각) worst-case PSLL. 위상은 각도별 변수(다각도).
#
# 목적: (1) +φ 정준형이 기존 −φ 규약과 |E| 동일함을 증명(모든 PSLL 결과 이월),
#       (2) 피드백의 변수집합으로 두 비교군을 재실행해 수치 재현.
# 실행: venv/bin/python experiment_baseline_canonical.py
import json
import math
import os

import torch as th

from farfield_canonical import (K, LAM, field_nonuniform, field_uniform,
                                intensity, steering_phase_canonical)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64
N = 32
DMIN, DMAX, DINIT = 2.0, 5.0, 3.0
ANG = [math.sin(math.radians(a)) for a in range(-30, 31, 5)]
UTR = th.linspace(-1, 1, 4001, dtype=DT)
UVAL = th.linspace(-1, 1, 40001, dtype=DT)
EF2 = th.sinc(1.0 * UTR / LAM) ** 2          # A_n=1, 소자인자는 전역 포락선 (강도)
EF2V = th.sinc(1.0 * UVAL / LAM) ** 2


def gaps(s):
    return DMIN + (DMAX - DMIN) * th.sigmoid(s)


def positions(s):
    return th.cat([th.zeros(1, dtype=DT), th.cumsum(gaps(s), 0)])


def ef2_at(u0):
    return th.sinc(th.tensor(1.0 * u0 / LAM, dtype=DT)) ** 2


def soft_psll(field, i0_field, u_grid, ef2, x_ap, u0, beta):
    # i0 = 실제 주엽 강도 |E(u0)|² (고정 N² 아님) → 주엽 약화로 PSLL 속이는 퇴화 차단 (losses.py 규약)
    I = ef2 * intensity(field)
    i0 = ef2_at(u0) * intensity(i0_field)
    guard = 2 * LAM / x_ap
    m = (u_grid - u0).abs() > guard
    D = 10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)
    return th.logsumexp(beta * D, 0) / beta


@th.no_grad()
def hard_worst(kind, s_or_d, phis):
    # 반환: (worst PSLL[dB], 최저 주엽효율 |E(u0)|²/N²) — 주엽 약화 감시
    worst, min_eff = -math.inf, math.inf
    u0t = lambda v: th.tensor([v], dtype=DT)
    for i, u0 in enumerate(ANG):
        if kind == 'uni':
            x = th.arange(N, dtype=DT) * s_or_d
            E = field_uniform(s_or_d, phis[i], UVAL, N)
            E0 = field_uniform(s_or_d, phis[i], u0t(u0), N)[0]
        else:
            x = positions(s_or_d)
            E = field_nonuniform(x, phis[i], UVAL)
            E0 = field_nonuniform(x, phis[i], u0t(u0))[0]
        eff = (E0.abs() ** 2 / N ** 2).item()
        min_eff = min(min_eff, eff)
        I = EF2V * intensity(E)
        i0 = ef2_at(u0) * E0.abs() ** 2          # 실제 주엽 강도 기준
        guard = 2 * LAM / (x[-1] - x[0])
        m = (UVAL - u0).abs() > guard
        worst = max(worst, (10 * th.log10(I[m] / (i0 + 1e-30) + 1e-30)).max().item())
    return worst, min_eff


def verify_equivalence():
    # |E|_canonical(x, +φ) == |E|_model(x, −φ)  → 부호는 순수 규약
    from model import OPAModel
    from config import JointConfig
    m = OPAModel(JointConfig())
    x = positions(th.randn(N - 1, dtype=DT))
    phi = th.randn(N, dtype=DT)
    u = th.linspace(-1, 1, 501, dtype=DT)
    E_canon = field_nonuniform(x, phi, u).abs()            # +φ
    af_model = th.exp(1j * (m.k * u.reshape(-1, 1) * x.reshape(1, -1) - (-phi).reshape(1, -1))).sum(1).abs()
    err = (E_canon - af_model).abs().max().item()
    return err


def design_uniform(d=DINIT, epochs=600):
    # 변수 = φ_n 만 (각도별). 간격 d 고정.
    phis = []
    for u0 in ANG:
        x = th.arange(N, dtype=DT) * d
        phi = steering_phase_canonical(x, u0).clone().requires_grad_(True)
        opt = th.optim.Adam([phi], lr=3e-2)
        for t in range(epochs):
            beta = 0.2 * (6 / 0.2) ** (t / epochs)
            opt.zero_grad()
            E = field_uniform(d, phi, UTR, N)
            E0 = field_uniform(d, phi, th.tensor([u0], dtype=DT), N)
            loss = soft_psll(E, E0, UTR, EF2, x[-1] - x[0], u0, beta)
            loss.backward(); opt.step()
        phis.append(phi.detach())
    return d, phis


def design_nonuniform(seeds=8, epochs=800):
    best = (None, None, math.inf)
    for r in range(seeds):
        th.manual_seed(300 + r)
        s = (math.log((DINIT - DMIN) / (DMAX - DINIT)) + 0.5 * th.randn(N - 1, dtype=DT)).requires_grad_(True)
        dphi = th.zeros(len(ANG), N, dtype=DT, requires_grad=True)   # 각도별 위상잔차
        opt = th.optim.Adam([{'params': [s], 'lr': 1e-2}, {'params': [dphi], 'lr': 3e-2}])
        for t in range(epochs):
            beta = 0.2 * (6 / 0.2) ** (t / epochs)
            opt.zero_grad()
            x = positions(s)
            per = []
            for i, u0 in enumerate(ANG):
                phi = steering_phase_canonical(x, u0) + dphi[i]
                E = field_nonuniform(x, phi, UTR)
                E0 = field_nonuniform(x, phi, th.tensor([u0], dtype=DT))
                per.append(soft_psll(E, E0, UTR, EF2, x[-1] - x[0], u0, beta))
            loss = th.logsumexp(2.0 * th.stack(per), 0) / 2.0
            loss.backward(); opt.step()
        with th.no_grad():
            x = positions(s.detach())
            phis = [steering_phase_canonical(x, u0) + dphi[i].detach() for i, u0 in enumerate(ANG)]
            v, _ = hard_worst('non', s.detach(), phis)
        if v < best[2]:
            best = (s.detach(), phis, v)
    return best


def main():
    report = {}
    err = verify_equivalence()
    report['equivalence_maxerr'] = err
    print(f"[등가] |E|_정준형(+φ) vs 기존(−φ) 최대오차 = {err:.2e}  → 부호는 순수 규약, 결과 이월")

    # 대조: 등간격 + 선형 조향램프만 (고전 baseline, grating lobe)
    x_u = th.arange(N, dtype=DT) * DINIT
    v_steer, eff_steer = hard_worst('uni', DINIT, [steering_phase_canonical(x_u, u0) for u0 in ANG])
    report['uniform_steering_only'] = {'worst_psll_db': round(v_steer, 2), 'min_mainlobe_eff': round(eff_steer, 3),
                                       'vars': '선형 조향램프 φ_n=−k·x_n·u0'}
    print(f"[등간격] 선형조향만(고전): worst {v_steer:+.2f} dB | 주엽효율 {eff_steer:.3f}  (grating lobe)")

    d, uphi = design_uniform()
    v_uni, eff_uni = hard_worst('uni', d, uphi)
    report['uniform_phase_only'] = {'worst_psll_db': round(v_uni, 2), 'min_mainlobe_eff': round(eff_uni, 3),
                                    'vars': 'φ_n only (d=3µm 고정, 자유 32위상)'}
    print(f"[등간격] 자유위상만(A_n=1): worst {v_uni:+.2f} dB | 최저 주엽효율 {eff_uni:.3f}")

    s, nphi, v_non = design_nonuniform()
    _, eff_non = hard_worst('non', s, nphi)
    report['nonuniform_pos_and_phase'] = {'worst_psll_db': round(v_non, 2), 'min_mainlobe_eff': round(eff_non, 3),
                                          'vars': 'x_n AND φ_n'}
    print(f"[비등간격] 위치+위상(A_n=1): worst {v_non:+.2f} dB | 최저 주엽효율 {eff_non:.3f}")

    report['note'] = '정준형 E=Σexp(j(k·x_n·u+φ_n)), A_n=1. 부호 규약만 정렬, 물리 동일.'
    json.dump(report, open(os.path.join(RESULTS, 'baseline_canonical.json'), 'w'), indent=2, ensure_ascii=False)
    print(f"\n→ results/baseline_canonical.json")


if __name__ == '__main__':
    main()
