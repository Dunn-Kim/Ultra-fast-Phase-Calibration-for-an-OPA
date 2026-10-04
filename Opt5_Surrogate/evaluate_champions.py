# 챔피언 4축 재평가 — 사용자 지정 평가 기준으로 트랙 ①/② 후보를 동일 척도에 올린다
#
# 축 1. 사이드로브 최소 (메인로브 제외 전부)
#      - PSLL      : 최대 사이드로브 / 메인로브 피크 [dB], 설계각 중 최악
#      - ISL       : 가드밴드 밖 총 에너지 / 메인로브 에너지 [dB] ("전부" 축)
#        (θ-균등 격자이므로 Δu = grad(u) 가중 적분 — 균등합은 각도 편향)
# 축 2. 소요 시간·세대 (모사 학습은 상각으로 제외 — 사용자 규약)
# 축 3. 메인로브 출력 최대
#      - eta_main  : I(u0) / EF(u0)²  — 동상 합 이론상한 대비 효율 (1.0 = 완전 정합)
# 축 4. 조향 시 출력 유지
#      - keep_db   : 최외곽 조향각 I(u0) / 정면 I(0) [dB] (0 에 가까울수록 유지)
#      - keep_worst: 설계각 중 최악 유지율 [dB]
#
# 커플링 규약(d>2µm, d_safe=2.3µm)은 부가 열로 병기 — J = PSLL + 페널티 (IMP5 회계).
import os
import json
import math
import argparse
import numpy as np
import torch as th

from physics_oracle import PhysicsOracle
from opt_core import (DESIGN_ANGLES_DEG, coupling_penalty, batched_phi_star,
                      du_weights, guard_mask, mc_psll, u0_vector)

HERE = os.path.dirname(os.path.abspath(__file__))


def metrics(oracle, d, angles_deg=DESIGN_ANGLES_DEG):
    jc = oracle.jcfg
    u0_vec = u0_vector(angles_deg=angles_deg)
    dRA, phi, u0RA = batched_phi_star(oracle.k, d.unsqueeze(0), u0_vec)
    I = oracle.intensity(dRA, phi)                    # (A, G)
    u = oracle.u
    du = du_weights(u)                                # Δu 가중
    M = guard_mask(u, u0RA, dRA.sum(dim=1))           # 메인로브 가드밴드 (A, G)
    ef = oracle.opa.element_factor_amp(u0_vec) ** 2    # 각도별 이론상한

    psll, isl, eta, peak, hpbw = [], [], [], [], []
    for i in range(I.shape[0]):
        m, Ii = M[i], I[i]
        pk = float(Ii[m].max())
        side = float(th.where(m, th.full_like(Ii, 0.0), Ii).max())
        psll.append(10.0 * math.log10(max(side, 1e-30) / max(pk, 1e-30)))
        e_main = float((Ii * du).abs()[m].sum())
        e_side = float((Ii * du).abs().sum() - e_main)
        isl.append(10.0 * math.log10(max(e_side, 1e-30) / max(e_main, 1e-30)))
        peak.append(pk)
        eta.append(pk / max(float(ef[i]), 1e-30))
        # 빔폭(HPBW): 메인로브 −3dB 교차폭 [deg] — 개구가 넓을수록 좁다
        half = Ii >= 0.5 * pk
        idx = th.nonzero(half & m).flatten()
        if idx.numel() >= 2:
            th_lo = math.degrees(math.asin(max(-1.0, min(1.0, float(u[idx[0]])))))
            th_hi = math.degrees(math.asin(max(-1.0, min(1.0, float(u[idx[-1]])))))
            hpbw.append(abs(th_hi - th_lo))
        else:
            hpbw.append(float('nan'))
    keep = [10.0 * math.log10(max(p, 1e-30) / max(peak[0], 1e-30))
            for p in peak]
    cpl = float(coupling_penalty(d, jc))
    return dict(
        psll_worst_db=round(max(psll), 3),
        psll_per_angle=[round(x, 2) for x in psll],
        isl_worst_db=round(max(isl), 3),
        isl_mean_db=round(float(np.mean(isl)), 3),
        eta_main_worst=round(min(eta), 4),
        eta_main_per_angle=[round(x, 3) for x in eta],
        hpbw_deg=round(float(np.mean(hpbw)), 3),
        keep_outer_db=round(min(keep), 3),
        keep_per_angle=[round(x, 2) for x in keep],
        coupling_penalty=round(cpl, 4),
        J=round(max(psll) + cpl, 3),
        min_gap_um=round(float(d.min()), 3),
        aperture_um=round(float(d.sum()), 1))


# 후보 목록: 챔피언은 results/ 루트, 비교용 실험 기록은 results/experiments/ 에 둔다.
# 시간·세대·수식평가는 각 실행의 보고값을 그대로 옮겨 적는다 (재계산하지 않음).
CANDIDATES = [
    # (라벨, 트랙, CSV, 시간[s], 세대/반복, 수식평가, 비고)
    ('uniform d=3 (기준선)', '-', None, 0.0, 0, 0, '설계 없음'),
    ('GA (대체 대상)', 'GA', 'results/experiments/spacing_ftmlp_ga.csv', 24.4, 400,
     128000, 'pop64×400세대'),
    ('① main Opt4 (배치 Adam)', '1', 'results/experiments/spacing_ftmlp_formula.csv',
     38.1, 400, 128000, '64재시작×400ep'),
    ('① 400ep (GA예산 강제)', '1',
     'results/experiments/final_spacing_track1_champion.csv', 41.12, 400, 137500,
     '예산 과다 — GA 시간 위반'),
    ('① 150ep', '1', 'results/experiments/final_spacing_track1_e150.csv', 17.22, 150,
     57500, ''),
    ('① 30ep PSLL단독 (개선전)', '1', 'results/experiments/final_spacing_track1_e30.csv',
     5.55, 30, 19040, 'w_isl=0'),
    ('① 챔피언 (+ISL항)', '1', 'results/final_spacing_champion_track1.csv',
     6.31, 30, 19040, 'w_isl=1.0, barrier 3.0'),
    ('② IMP1 (Adam탐색+LBFGS)', '2',
     'results/experiments/final_spacing_champion_formula_lbfgs.csv', 13.34, 500, 10520,
     '탐색 256×500'),
    ('② IMP4 CMA 250세대', '2', 'results/experiments/final_spacing_imp4_cmaes.csv',
     6.77, 250, 6240, '8아일랜드×pop32'),
    ('② IMP4 CMA 60세대', '2', 'results/experiments/final_spacing_imp4_g60.csv',
     3.29, 60, 5000, '6아일랜드×pop32'),
    ('② tandem PSLL단독 (개선전)', '2',
     'results/experiments/final_spacing_imp6_tandem_pm15_single.csv', 0.32, 0, 805,
     'w_isl=0, 학습 27.3s 상각'),
    ('① 로버스트 변형 (K=24)', '1',
     'results/final_spacing_champion_track1_robust.csv', 68.9, 30, 19040,
     '연마만 로버스트, MC 선택'),
    ('② 챔피언 tandem (+ISL항)', '2',
     'results/final_spacing_champion_tandem_pm15.csv', 0.38, 0, 950,
     'w_isl=0.4, 학습 27.7s 상각, 추론 0.17ms'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='champion_evaluation')
    ap.add_argument('--mc', type=int, default=0, metavar='N',
                    help='제조·구동 오차 하 PSLL 분포를 N회 몬테카를로로 병기')
    a = ap.parse_args()
    oracle = PhysicsOracle()
    jc = oracle.jcfg
    rows = []
    for label, track, csv, t_s, gens, n_ev, note in CANDIDATES:
        if csv is None:
            d = th.full((jc.line_N - 1,), jc.d_init, dtype=th.float64)
        else:
            p = os.path.join(HERE, csv)
            if not os.path.exists(p):
                continue
            d = th.tensor(np.loadtxt(p, skiprows=1), dtype=th.float64)
        m = metrics(oracle, d)
        if a.mc:                       # 공칭이 아닌 '실제 기대 성능'
            v = mc_psll(oracle, d, u0_vector(), n=a.mc)
            m.update(mc_mean_db=round(float(v.mean()), 2),
                     mc_p90_db=round(float(v.quantile(0.9)), 2),
                     mc_worst_db=round(float(v.max()), 2))
        rows.append(dict(label=label, track=track, time_s=t_s, gens=gens,
                         n_formula_eval=n_ev, note=note, **m))

    out = dict(protocol=dict(design_angles_deg=list(DESIGN_ANGLES_DEG),
                             guard='kappa=2·lambda/L_ap',
                             isl='Δu 가중 적분, 가드밴드 밖 전부',
                             eta_main='I(u0)/EF(u0)² (동상합 상한 대비)',
                             keep='I(u0)/I(0°) [dB]'),
               rows=rows)
    jp = os.path.join(HERE, 'results', f'{a.tag}.json')
    with open(jp, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    mc_h = f" {'오차p90':>8s}" if a.mc else ''
    hdr = (f"{'후보':30s} {'trk':>3s} {'PSLL':>8s} {'ISL':>8s} {'HPBW':>6s} "
           f"{'η':>6s} {'유지':>7s} {'J':>8s}{mc_h} {'시간':>8s} {'세대':>6s} "
           f"{'수식평가':>9s}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['label']:30s} {r['track']:>3s} "
              f"{r['psll_worst_db']:8.2f} {r['isl_worst_db']:8.2f} "
              f"{r['hpbw_deg']:6.2f} "
              f"{r['eta_main_worst']:6.3f} {r['keep_outer_db']:7.2f} "
              f"{r['J']:8.2f}"
              f"{(' %8.2f' % r['mc_p90_db']) if a.mc else ''} "
              f"{r['time_s']:8.2f} {r['gens']:6d} {r['n_formula_eval']:9d}")
    print('→', jp)


if __name__ == '__main__':
    main()
