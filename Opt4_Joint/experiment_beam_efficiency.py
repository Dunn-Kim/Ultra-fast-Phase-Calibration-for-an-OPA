# 사이드로브 전력을 주엽으로 회수하려면? — 빔 효율(main-lobe power fraction) 측정
#
# PSLL(우리가 최적화한 것) = 피크 '비율'. 사이드로브 피크를 평탄화해도 그 '전력'은 그대로 남음.
# 주엽 전력비 η = ∫_주엽 I dθ / ∫_전체 I dθ  ← 이게 "출력을 중앙으로 모으는" 지표.
#
# 전력 적분이므로 Jacobian 필요: P ∝ ∫I(θ)dθ = ∫ I(u)/√(1−u²) du  (점별 PSLL과 달리 소거 안 됨)
#
# 이론 예측: 희소배열에서 η ≈ λ/(2·d̄) — 평균간격만의 함수. 간격 '패턴'과 무관.
#   d̄=3µm → η≈0.26 (74% 전력이 사이드로브!),  d̄=λ/2=0.775 → η≈1.0
# 검증 대상: (1) 비등간격 최적화가 η를 개선하는가? (2) w 축소(PSLL +1dB 레버)가 η에 주는 영향?
#
# 실행: venv/bin/python experiment_beam_efficiency.py
import json
import math
import os

import pandas as pd
import torch as th

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DT = th.float64
LAM = 1.55
K = 2 * math.pi / LAM
N = 32

# u 격자: 적분 정확도 위해 조밀. |u|<1 (가시영역). 끝점 특이(1/√(1−u²)) 회피 위해 여유
U = th.linspace(-0.99999, 0.99999, 200001, dtype=DT)
JAC = 1.0 / th.sqrt(1 - U ** 2)          # dθ/du


def af2(x, u0):
    ph = K * (U.reshape(-1, 1) - u0) * x.reshape(1, -1)
    return th.exp(1j * ph).sum(1).abs() ** 2


def ef2(w):
    return th.sinc(w * U / LAM) ** 2


def beam_efficiency(x, u0=0.0, w=1.0, ml_factor=1.0):
    # η = ∫_ML I·J du / ∫_all I·J du,  ML = |u−u0| ≤ ml_factor·λ/L (첫 널 근사)
    I = ef2(w) * af2(x, u0)
    L = (x[-1] - x[0]).item()
    half = ml_factor * LAM / L
    ml = (U - u0).abs() <= half
    num = th.trapz((I * JAC)[ml], U[ml])
    den = th.trapz(I * JAC, U)
    return (num / den).item()


def uniform_x(d, n=N):
    return th.arange(n, dtype=DT) * d


def main():
    rep = {}
    champ_d = th.tensor(pd.read_csv(os.path.join(RESULTS, 'final_spacing_lossfix.csv'))['d_um'].values, dtype=DT)
    champ_x = th.cat([th.zeros(1, dtype=DT), th.cumsum(champ_d, 0)])
    dbar_champ = champ_d.mean().item()

    print('=== 주엽 전력비 η (θ=0°, w=1µm) — 이론 η≈λ/(2·d̄) ===')
    cases = [
        ('등간격 3µm',            uniform_x(3.0), 3.0),
        ('비등간격 챔피언(우리)', champ_x, dbar_champ),
        ('등간격 2µm (d_min)',    uniform_x(2.0), 2.0),
        ('등간격 1.0µm',          uniform_x(1.0), 1.0),
        ('등간격 0.775µm (λ/2)',  uniform_x(0.775), 0.775),
    ]
    rows = []
    for name, x, dbar in cases:
        eta = beam_efficiency(x, 0.0, 1.0)
        theory = min(LAM / (2 * dbar), 1.0)
        rows.append({'case': name, 'd_bar_um': round(dbar, 3), 'eta': round(eta, 4),
                     'theory_lam_2d': round(theory, 3)})
        print(f"  {name:22s} d̄={dbar:5.3f}µm | η={eta:.3f} | 이론 λ/(2d̄)={theory:.3f}")
    rep['eta_vs_spacing'] = rows

    print('\n=== 비등간격 패턴이 η를 바꾸는가? (동일 d̄ 비교) ===')
    # 챔피언과 같은 평균간격의 등간격 배열
    x_eq = uniform_x(dbar_champ)
    e_champ = beam_efficiency(champ_x, 0.0, 1.0)
    e_eq = beam_efficiency(x_eq, 0.0, 1.0)
    rep['pattern_effect'] = {'aperiodic_champion': round(e_champ, 4), 'uniform_same_dbar': round(e_eq, 4),
                             'delta': round(e_champ - e_eq, 4), 'd_bar': round(dbar_champ, 3)}
    print(f"  비등간격 챔피언 η={e_champ:.3f}  vs  등간격 동일 d̄={dbar_champ:.2f}µm η={e_eq:.3f}")
    print(f"  → 간격 '패턴'의 η 효과: {e_champ - e_eq:+.4f}  (0에 가까우면 패턴 무관 = 충전율만이 지렛대)")

    print('\n=== 소자폭 w의 영향 (PSLL 레버 w:1.0→0.4가 η에 주는 효과) ===')
    wrows = []
    for w in (0.4, 0.7, 1.0, 1.5, 2.0):
        e = beam_efficiency(champ_x, 0.0, w)
        wrows.append({'w_um': w, 'eta': round(e, 4)})
        print(f"  w={w:.1f}µm | η={e:.3f}")
    rep['eta_vs_width'] = wrows

    print('\n=== 조향 시 η (챔피언, w=1) ===')
    srows = []
    for a in (0, 15, 30):
        u0 = math.sin(math.radians(a))
        e = beam_efficiency(champ_x, u0, 1.0)
        srows.append({'theta_deg': a, 'eta': round(e, 4)})
        print(f"  θ={a:2d}° | η={e:.3f}")
    rep['eta_vs_steer'] = srows

    json.dump(rep, open(os.path.join(RESULTS, 'beam_efficiency.json'), 'w'), indent=2, ensure_ascii=False)
    print('\n→ results/beam_efficiency.json')


if __name__ == '__main__':
    main()
