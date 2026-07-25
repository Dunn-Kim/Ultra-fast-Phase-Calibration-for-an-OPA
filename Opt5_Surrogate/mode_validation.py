# MODE 검증 — 수식 모델이 비등간격 설계에서도 맞는지 확인하는 도구
#
# 현 EF 모델은 d=3µm 등간격 varFDTD 로그로 적합됐다. 챔피언은 비등간격이므로
# 소자 간 결합이 달라져 EF 가 어긋날 수 있다 — 이것이 남은 유일한 미검증 항목이다.
# 이 PC 에는 Lumerical 이 없으므로 세 가지를 제공한다:
#
#   --spec        랩에서 돌릴 MODE 실행 명세 출력 (간격·위상·격자·저장형식)
#   --check CSV   MODE 결과를 수식 예측과 대조 (R², PSLL 오차, 합격 판정)
#   --crosscheck  기존 로그(N=32/64/128)로 EF 모델 자체를 재검증 — 지금 실행 가능
import os
import json
import math
import argparse
import numpy as np
import torch as th

from physics_oracle import PhysicsOracle
from opt_core import batched_phi_star

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'Opt3_Back_Forward', 'Resultants')
CHAMP = 'results/final_spacing_champion_track1.csv'

# 합격 기준 — 이보다 나쁘면 EF 모델을 비등간격 데이터로 재적합해야 한다
GATE_R2 = 0.95          # 강도 상관 (프레임별 LS 게인 흡수 후)
GATE_PSLL_DB = 1.0      # worst-angle PSLL 절대오차 [dB]


def load_champion():
    d = np.loadtxt(os.path.join(HERE, CHAMP), skiprows=1)
    return th.tensor(d, dtype=th.float64)


def print_spec():
    o = PhysicsOracle()
    d = load_champion()
    x = th.cat([th.zeros(1, dtype=th.float64), th.cumsum(d, 0)])
    print('=' * 72)
    print('MODE(varFDTD) 검증 실행 명세 — 이 조건 그대로 돌린 뒤 --check 로 대조')
    print('=' * 72)
    print(f'\n[구조] 소자 {o.jcfg.line_N}개, 폭 {o.jcfg.element_width}µm, '
          f'두께 0.5µm, λ={o.jcfg.wavelength}µm, 진폭 균일')
    print(f'       개구 {float(d.sum()):.3f}µm, 최소 간격 {float(d.min()):.3f}µm')
    print('\n[소자 위치 x, µm] — 등간격 아님, 아래 값을 그대로 배치')
    for i in range(0, 32, 8):
        print('  ' + '  '.join(f'{float(v):8.3f}' for v in x[i:i + 8]))
    print('\n[인가 위상, deg] — 조향각별. 소자0 = 0° 기준')
    for deg in (0.0, 7.5, 15.0):
        ph = th.rad2deg(th.remainder(o.k * x * math.sin(math.radians(deg)),
                                     2 * math.pi))
        print(f'  θ={deg:+.1f}°:')
        for i in range(0, 32, 8):
            print('    ' + '  '.join(f'{float(v):7.1f}' for v in ph[i:i + 8]))
    print('  (음의 각도는 위상 부호 반전: φ(−θ) = 360° − φ(θ), 별도 실행 불필요)')
    print('\n[관측] 원거리장 θ = −90°..+90°, 0.1° 간격 (1801점)')
    print('[저장] CSV 2열 — theta_deg, intensity (스케일 임의, 자동 정규화)')
    print('       파일명 예: mode_champion_pm15.csv  (조향각별 1개씩)')
    print('\n[대조] python mode_validation.py --check mode_champion_pm15.csv --angle 15')
    print(f'[합격] 강도 R² ≥ {GATE_R2}, PSLL 오차 ≤ {GATE_PSLL_DB} dB')
    print('       미달 시 → EF 모델을 비등간격 로그로 재적합 후 챔피언 재설계 필요\n')


def check(csv_path, angle_deg):
    o = PhysicsOracle()
    d = load_champion()
    raw = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    th_obs, I_obs = raw[:, 0], raw[:, 1]

    u0 = th.tensor([math.sin(math.radians(angle_deg))], dtype=th.float64)
    dRA, phi, u0RA = batched_phi_star(o.k, d.unsqueeze(0), u0)
    I_pred_full = o.intensity(dRA, phi)[0].numpy()
    th_pred = np.degrees(np.arcsin(np.clip(o.u.numpy(), -1, 1)))
    I_pred = np.interp(th_obs, th_pred, I_pred_full)   # 관측 격자로 보간

    obs = I_obs / I_obs.max()
    prd = I_pred / I_pred.max()
    c = (prd * obs).sum() / max((prd * prd).sum(), 1e-30)   # 프레임 LS 게인
    r2 = 1.0 - ((c * prd - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()

    # PSLL 은 관측 격자에서 직접 산출 (가드밴드는 수식과 동일 규약)
    guard = 2.0 * o.jcfg.wavelength / float(d.sum())
    u_obs = np.sin(np.radians(th_obs))
    m = np.abs(u_obs - float(u0)) < guard
    p_obs = 10 * np.log10(max(obs[~m].max(), 1e-30) / max(obs[m].max(), 1e-30))
    p_prd = 10 * np.log10(max(prd[~m].max(), 1e-30) / max(prd[m].max(), 1e-30))

    ok = (r2 >= GATE_R2) and (abs(p_obs - p_prd) <= GATE_PSLL_DB)
    out = dict(csv=csv_path, angle_deg=angle_deg,
               intensity_r2=round(float(r2), 5),
               psll_mode_db=round(float(p_obs), 3),
               psll_formula_db=round(float(p_prd), 3),
               psll_err_db=round(abs(float(p_obs - p_prd)), 3),
               gate_r2=GATE_R2, gate_psll_db=GATE_PSLL_DB,
               verdict='PASS' if ok else 'FAIL')
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if not ok:
        print('\n판정: 수식이 비등간격 배열을 충분히 설명하지 못한다.')
        print('  → EF 를 비등간격 로그로 재적합하고 챔피언을 재설계할 것.')
    return out


def crosscheck():
    # 기존 등간격 로그(N=32/64/128)로 EF 모델을 재검증 — 지금 바로 실행 가능
    o = PhysicsOracle()
    print('[EF 모델 교차검증] 등간격 d=3µm varFDTD 로그, 프레임별 LS 게인 흡수 후 R²')
    rows = []
    for n_elem in (32, 64, 128):
        pp = os.path.join(RES, f'phase_tracked_{n_elem}.csv')
        tp = os.path.join(RES, f'pattern_tracked_{n_elem}.csv')
        if not (os.path.exists(pp) and os.path.exists(tp)):
            print(f'  N={n_elem}: 로그 없음 (건너뜀)')
            continue
        ph = np.loadtxt(pp, delimiter=',', skiprows=1)[:, 1:]
        pt = np.loadtxt(tp, delimiter=',', skiprows=1)[:, 1:]
        phi = np.deg2rad(ph[1:1 + pt.shape[0]])
        phi = np.concatenate([np.zeros((phi.shape[0], 1)), phi], axis=1)
        obs = pt / pt.max(axis=1, keepdims=True)
        x = th.arange(n_elem, dtype=th.float64) * o.jcfg.d_init
        ef = o.opa.element_factor_amp(o.u)
        r2s = []
        for j in range(obs.shape[0]):
            p = th.tensor(phi[j], dtype=th.float64)
            E = th.exp(1j * (o.k * o.u.reshape(-1, 1) * x.reshape(1, -1)
                             - p.reshape(1, -1))).sum(1)
            prd = ((ef * E.abs() / n_elem) ** 2).numpy()
            ob = obs[j]
            c = (prd * ob).sum() / max((prd * prd).sum(), 1e-30)
            r2s.append(1 - ((c * prd - ob) ** 2).sum() / ((ob - ob.mean()) ** 2).sum())
        rows.append((n_elem, float(np.mean(r2s)), float(np.min(r2s))))
        fit = ' (적합에 사용)' if n_elem == 32 else ' (홀드아웃)'
        print(f'  N={n_elem:3d}: 평균 R² {np.mean(r2s):.4f}  최저 {np.min(r2s):.4f}{fit}')
    print('\n한계: 세 로그 모두 d=3µm 등간격 — 비등간격 검증은 --spec 실행이 유일한 길.')
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', action='store_true', help='MODE 실행 명세 출력')
    ap.add_argument('--check', metavar='CSV', help='MODE 결과 대조')
    ap.add_argument('--angle', type=float, default=15.0, help='--check 의 조향각')
    ap.add_argument('--crosscheck', action='store_true', help='기존 로그로 EF 재검증')
    a = ap.parse_args()
    if a.spec:
        print_spec()
    if a.crosscheck:
        crosscheck()
    if a.check:
        check(a.check, a.angle)
    if not (a.spec or a.check or a.crosscheck):
        ap.print_help()
