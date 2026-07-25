# 트랙 ① 챔피언 — 수식 직접(모사 없음) 최선 구성
#
# 구성 = 수식 배치 Adam 멀티스타트 탐색 → 수식 재순위(무비용, 이미 수식) →
#        다양성 top-K → L-BFGS 연마.  전 단계가 수식 — 모사 미사용.
#
# 기본값 근거:
#   epochs 30  — GA 예산(400ep)은 과잉이었다. 13× 축소에 PSLL 손실 0.34 dB, 시간 7.4× 단축.
#   w_isl 1.0  — 목적함수에 사이드로브 총에너지(ISL)를 추가. PSLL 단독 대비
#                ISL 0.90→0.56, PSLL −13.32→−13.35, 개구 94.5→90.7µm 동시 개선.
#   w_barrier 3.0 — ISL 항은 개구를 좁히려 간격을 하한으로 민다. barrier 를 올려
#                   최소 간격 2.206→2.289µm 로 회복 (안전 마진과 품질 양립).
import os
import json
import time
import argparse
import numpy as np
import torch as th

from physics_oracle import PhysicsOracle
from opt_core import (u0_vector, formula_hard_per_restart, run_gradient,
                      diverse_topk, d_to_logit, polish_lbfgs,
                      coupling_penalty_d, mc_psll)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--restarts', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=30)   # 예산 교정: 400→30 (7.4× 단축)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--top-k', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--w-isl', type=float, default=1.0,
                    help='사이드로브 총에너지(ISL) 항 가중 — 0이면 PSLL 단독')
    ap.add_argument('--w-barrier', type=float, default=3.0,
                    help='커플링 barrier 가중 (ISL 항의 하한 압력 상쇄)')
    ap.add_argument('--robust', type=int, default=0, metavar='K',
                    help='연마 단계를 오차 실현 K개의 기대 손실로 수행 (0=공칭)')
    ap.add_argument('--robust-explore', action='store_true',
                    help='탐색 단계까지 로버스트화 — 실측상 역효과(공칭 0.54 dB '
                         '손실이 이득을 상쇄)라 기본 비활성. 재현용으로만 남김')
    ap.add_argument('--tag', default='champion_track1')
    a = ap.parse_args()

    oracle = PhysicsOracle()
    jc = oracle.jcfg
    # barrier 상향은 최적화 압력 조절용 — 판정은 항상 규약 기본값으로 되돌려 수행한다
    w_barrier_judge = jc.cpl_w_barrier
    if a.w_barrier is not None:
        jc.cpl_w_barrier = a.w_barrier
    u0_vec = u0_vector()

    def eval_I_f(dRA, phi):
        return oracle.intensity(dRA, phi)

    t0 = time.perf_counter()
    # 탐색은 공칭으로 — 노이즈를 넣으면 경사가 흐려져 좋은 분지를 놓친다(실측)
    d_x, t_x, n_x = run_gradient(eval_I_f, th.float64, 'cpu', oracle,
                                 a.restarts, a.epochs, a.lr, True, a.seed,
                                 oracle.u, w_isl=a.w_isl,
                                 robust_k=a.robust if a.robust_explore else 0)
    t1 = time.perf_counter()
    p_rank = formula_hard_per_restart(oracle, d_x, u0_vec)
    top = diverse_topk(d_x, p_rank, a.top_k)
    t_rank = time.perf_counter() - t1
    n_rank = d_x.shape[0] * u0_vec.shape[0]
    d_p, t_p, n_p = polish_lbfgs(oracle, d_to_logit(d_x[top], jc), jc, u0_vec,
                                 w_isl=a.w_isl, robust_k=a.robust,
                                 seed=a.seed)
    p_fin = formula_hard_per_restart(oracle, d_p, u0_vec)
    n_judge = d_p.shape[0] * u0_vec.shape[0]
    jc.cpl_w_barrier = w_barrier_judge          # 판정 기준 원복
    J = [float(p_fin[i]) + float(coupling_penalty_d(d_p[i], jc))
         for i in range(d_p.shape[0])]
    if a.robust:
        # 로버스트 모드에서는 선택 기준도 오차 하 성능이어야 일관된다
        # (공칭 J 로 고르면 로버스트 연마의 이득이 선택 단계에서 버려진다)
        score = [float(mc_psll(oracle, d_p[i], u0_vec, n=120).quantile(0.9))
                 + float(coupling_penalty_d(d_p[i], jc))
                 for i in range(d_p.shape[0])]
        pick = int(np.argmin(score))
    else:
        pick = int(np.argmin(J))      # J 기준 선택 (IMP5 회계)
    d_c = d_p[pick]
    t_total = time.perf_counter() - t0

    mc = mc_psll(oracle, d_c, u0_vec)          # 오차 하 실제 기대 성능
    out = dict(tag=a.tag, track='1_formula_direct',
               recipe=dict(restarts=a.restarts, epochs=a.epochs, lr=a.lr,
                           top_k=a.top_k, polish='lbfgs', judge='J',
                           w_isl=a.w_isl, robust_k=a.robust,
                           robust_explore=a.robust_explore),
               psll_formula_db=float(p_fin[pick]),
               mc=dict(mean=round(float(mc.mean()), 3),
                       p90=round(float(mc.quantile(0.9)), 3),
                       worst=round(float(mc.max()), 3)),
               coupling_penalty=round(float(coupling_penalty_d(d_c, jc)), 4),
               J=round(J[pick], 3),
               elapsed_s=round(t_total, 2),
               breakdown_s=dict(explore=round(t_x, 2), rank=round(t_rank, 2),
                                polish=round(t_p, 2)),
               n_eval_formula=n_x + n_rank + n_p + n_judge,
               min_gap_um=float(d_c.min()), aperture_um=float(d_c.sum()),
               d=d_c.tolist())
    jp = os.path.join(HERE, 'results', f'{a.tag}.json')
    with open(jp, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    np.savetxt(os.path.join(HERE, 'results', f'final_spacing_{a.tag}.csv'),
               np.array(d_c.tolist()), header='d_um', comments='')
    print(json.dumps({k: v for k, v in out.items() if k != 'd'},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
