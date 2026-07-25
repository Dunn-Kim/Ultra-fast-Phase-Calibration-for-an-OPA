# IMP4: CMA-ES-on-surrogate 탐색 — 동결 모사 위 무경사 전역탐색
#
# 탐색 엔진 교체 실험: Adam 멀티스타트(경사) ↔ CMA-ES K-아일랜드(무경사, 공분산 적응).
#   - 적합도 = 모사 hard PSLL worst-angle + 커플링 페널티 (무경사라 soft 근사 불필요)
#   - K개 독립 CMA 인스턴스를 락스텝으로 돌리며 세대당 전 아일랜드 개체를
#     한 번의 MPS 배치로 평가 (모사 저비용 활용)
#   - 이후 champion 규약 그대로: 수식 재순위 + 다양성 top-K + L-BFGS 연마
# 비교 기준: IMP1 champion (Adam 탐색 256×500, −13.402 dB / 13.4s)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import os
import json
import math
import time
import argparse
import numpy as np
import torch as th
import cma

from config import SurrogateConfig
from physics_oracle import PhysicsOracle
from train import psll_db, resolve_device
from optimize_spacing import DESIGN_ANGLES_DEG, load_frozen, coupling_penalty_d
from benchmark import (u0_vector, batched_phi_star, formula_hard_per_restart,
                       coupling_penalty_batched)
from champion_design import diverse_topk, d_to_logit, polish_lbfgs

HERE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))   # 산출물은 상위 results/ 로 통일


def make_fitness(model, oracle, jc, dev):
    u0v = u0_vector(th.float32).to(dev)
    A = u0v.shape[0]

    def fitness(S_np):
        # S_np: (B, N-1) s-로짓 → 적합도 (B,) numpy (낮을수록 좋음)
        s = th.tensor(np.asarray(S_np), dtype=th.float32, device=dev)
        with th.no_grad():
            d = jc.d_min + (jc.d_max - jc.d_min) * th.sigmoid(s)
            dRA, phi, u0RA = batched_phi_star(oracle.k, d, u0v)
            pred = model(dRA, phi)
            I = (pred[:, 0] ** 2 + pred[:, 1] ** 2).cpu().double()
            p = psll_db(I, oracle.u, u0RA.cpu().double(),
                        dRA.sum(dim=1).cpu().double())
            p = p.reshape(s.shape[0], A).amax(dim=1)
            f = p + coupling_penalty_batched(d.cpu().double(), jc)
        return f.numpy(), d.cpu().double()

    return fitness


def run_cmaes_islands(fitness, x0_list, sigma0, popsize, gens, seed, n_angles):
    # K-아일랜드 락스텝: 세대당 전 아일랜드 개체를 한 번의 배치로 평가
    #   x0_list 를 구조적 초기해로 주면 IMP5 (탐색 바닥 상승) 실험이 됨
    islands = []
    for kx, x0 in enumerate(x0_list):
        es = cma.CMAEvolutionStrategy(
            np.asarray(x0, dtype=float), sigma0,
            dict(popsize=popsize, seed=int(seed + 1000 + kx), verbose=-9))
        islands.append(es)
    n_surr = 0
    for _ in range(gens):
        asks = [es.ask() for es in islands]
        S = np.concatenate([np.asarray(xs) for xs in asks], axis=0)
        f, _ = fitness(S)
        n_surr += S.shape[0] * n_angles
        off = 0
        for es, xs in zip(islands, asks):
            es.tell(xs, f[off:off + len(xs)].tolist())
            off += len(xs)
    S_best = np.stack([es.result.xbest for es in islands])
    return S_best, n_surr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='mlp',
                    choices=['mlp', 'element', 'siren', 'ffmlp'])
    ap.add_argument('--ckpt', default='checkpoints/mlp_full.pt')
    ap.add_argument('--islands', type=int, default=8)
    ap.add_argument('--popsize', type=int, default=32)
    ap.add_argument('--gens', type=int, default=250)
    ap.add_argument('--sigma0', type=float, default=0.6)
    ap.add_argument('--top-k', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--tag', default='imp4_cmaes')
    a = ap.parse_args()

    cfg = SurrogateConfig(arch=a.arch)
    oracle = PhysicsOracle()
    jc = oracle.jcfg
    dev = th.device(resolve_device(a.device))
    u0_vec = u0_vector()
    model, _ = load_frozen(cfg, a.ckpt)
    model = model.to(dev)
    fitness = make_fitness(model, oracle, jc, dev)

    p0 = (jc.d_init - jc.d_min) / (jc.d_max - jc.d_min)
    s_center = math.log(p0 / (1.0 - p0))
    rng = np.random.default_rng(a.seed)

    t_total0 = time.perf_counter()
    # [1] 탐색: K-아일랜드 CMA-ES, 세대당 전 개체 일괄 평가
    t0 = time.perf_counter()
    x0s = [s_center + 0.3 * rng.standard_normal(jc.line_N - 1)
           for _ in range(a.islands)]
    S_best, n_surr = run_cmaes_islands(fitness, x0s, a.sigma0, a.popsize,
                                       a.gens, a.seed,
                                       len(DESIGN_ANGLES_DEG))
    _, d_best = fitness(S_best)
    t_x = time.perf_counter() - t0

    # [2] 수식 재순위 + 다양성 top-K (champion 규약)
    t0 = time.perf_counter()
    p_rank = formula_hard_per_restart(oracle, d_best, u0_vec)
    top = diverse_topk(d_best, p_rank, min(a.top_k, d_best.shape[0]))
    s0 = d_to_logit(d_best[top], jc)
    t_rank = time.perf_counter() - t0
    n_rank = d_best.shape[0] * u0_vec.shape[0]

    # [3] L-BFGS 연마 (IMP1)
    d_p, t_p, n_p = polish_lbfgs(oracle, s0, jc, u0_vec)
    p_fin = formula_hard_per_restart(oracle, d_p, u0_vec)
    n_judge = d_p.shape[0] * u0_vec.shape[0]
    pick = int(p_fin.argmin())
    d_c = d_p[pick]
    t_total = time.perf_counter() - t_total0

    # 순수 ② gap: 탐색 최종 자기판정 최적점의 수식 판정과의 괴리
    f_self, _ = fitness(S_best)
    b = int(np.argmin(f_self))
    cpl_b = float(coupling_penalty_d(d_best[b], jc))
    pure2 = dict(psll_self_db=round(float(f_self[b]) - cpl_b, 3),
                 psll_formula_db=round(float(p_rank[b]), 3),
                 gap_db=round(float(p_rank[b]) - (float(f_self[b]) - cpl_b), 3))

    out = dict(
        tag=a.tag, ckpt=a.ckpt,
        recipe=dict(islands=a.islands, popsize=a.popsize, gens=a.gens,
                    sigma0=a.sigma0, top_k=a.top_k, polish='lbfgs'),
        psll_formula_db=float(p_fin[pick]),
        elapsed_s=round(t_total, 2),
        breakdown_s=dict(explore=round(t_x, 2), rank=round(t_rank, 2),
                         polish=round(t_p, 2)),
        n_eval=dict(surrogate=n_surr,
                    formula=n_rank + n_p + n_judge),
        min_gap_um=float(d_c.min()), aperture_um=float(d_c.sum()),
        coupling_penalty=round(float(coupling_penalty_d(d_c, jc)), 4),
        pure2_gap=pure2,
        d=d_c.tolist())
    jp = os.path.join(HERE, 'results', f'{a.tag}.json')
    with open(jp, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    np.savetxt(os.path.join(HERE, 'results', f'final_spacing_{a.tag}.csv'),
               np.array(d_c.tolist()), header='d_um', comments='')
    print(json.dumps({k: v for k, v in out.items() if k != 'd'},
                     indent=2, ensure_ascii=False))
    print('→', jp)


if __name__ == '__main__':
    main()
