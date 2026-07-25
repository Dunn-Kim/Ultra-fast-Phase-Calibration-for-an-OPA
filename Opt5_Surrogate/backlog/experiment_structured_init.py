# IMP5: 구조적 초기해 생성기 — 밀도 테이퍼링(Doyle/Skolnik) + 처프 + 등간격 가족
#
# 원리: 저부엽 창함수 w(x)의 누적분포 역함수로 소자 위치를 배치하면 (밀도 테이퍼링)
#   진폭 테이퍼를 위치 밀도로 흉내내는 비등간격 배열이 나옴 — 무작위 재시작 대비
#   탐색 바닥(floor)을 끌어올리는 결정론 초기해.
# 레인:
#   A) 구조 초기해 → 수식 재순위 → 다양성 top-K → L-BFGS 연마 (탐색 0회!)
#   B) 구조 초기해를 CMA-ES 아일랜드 x0로 → IMP4 파이프라인 (탐색 바닥 상승)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import os
import json
import time
import argparse
import numpy as np
import torch as th
from scipy.signal import windows

from config import SurrogateConfig
from physics_oracle import PhysicsOracle
from train import resolve_device
from optimize_spacing import DESIGN_ANGLES_DEG, load_frozen
from benchmark import u0_vector, formula_hard_per_restart
from champion_design import diverse_topk, d_to_logit, polish_lbfgs
from experiment_cmaes_explore import make_fitness, run_cmaes_islands

HERE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))   # 산출물은 상위 results/ 로 통일


def taper_gaps(w, N, L, lo, hi):
    # 밀도 테이퍼링: 위치 = 창함수 누적분포의 역함수 (연속 격자 4096)
    w = np.clip(np.asarray(w, dtype=float), 1e-6, None)
    grid = np.linspace(0.0, 1.0, w.size)
    cum = np.cumsum(w)
    cum = (cum - cum[0]) / (cum[-1] - cum[0])
    levels = (np.arange(N) + 0.5) / N
    xs = np.interp(levels, cum, grid)
    xs = (xs - xs[0]) * (L / (xs[-1] - xs[0]))
    return np.clip(np.diff(xs), lo, hi)


def build_inits(N, lo, hi):
    # (이름, 간격벡터) 목록 — 전 가족 결정론
    M = 4096
    fams = {
        'taylor_sll20': windows.taylor(M, nbar=4, sll=20),
        'taylor_sll25': windows.taylor(M, nbar=4, sll=25),
        'taylor_sll30': windows.taylor(M, nbar=4, sll=30),
        'hamming': windows.hamming(M),
        'blackman': windows.blackman(M),
        'hann': windows.hann(M),
        'tukey05': windows.tukey(M, alpha=0.5),
        'boxcar': windows.boxcar(M),
    }
    out = []
    for L in (88.0, 94.0, 100.0, 106.0, 112.0):
        for name, w in fams.items():
            out.append((f'{name}_L{int(L)}', taper_gaps(w, N, L, lo, hi)))
    mid = (N - 2) / 2.0
    for L in (94.0, 100.0, 106.0):
        m = L / (N - 1)
        for b in (0.8, -0.8, 1.5, -1.5):
            idx = np.arange(N - 1)
            d = m + b * ((np.abs(idx - mid) / mid) - 0.5) * 2.0
            out.append((f'chirp{b:+.1f}_L{int(L)}', np.clip(d, lo, hi)))
    for d0 in (2.4, 2.8, 3.0, 3.2, 3.6, 4.0):
        out.append((f'uniform_{d0}', np.full(N - 1, d0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='mlp',
                    choices=['mlp', 'element', 'siren', 'ffmlp'])
    ap.add_argument('--ckpt', default='checkpoints/mlp_full.pt')
    ap.add_argument('--top-k', type=int, default=12)
    ap.add_argument('--islands', type=int, default=8)
    ap.add_argument('--popsize', type=int, default=32)
    ap.add_argument('--gens', type=int, default=250)
    ap.add_argument('--sigma0', type=float, default=0.6)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 7])
    ap.add_argument('--device', default='auto')
    ap.add_argument('--tag', default='imp5_structured')
    a = ap.parse_args()

    cfg = SurrogateConfig(arch=a.arch)
    oracle = PhysicsOracle()
    jc = oracle.jcfg
    dev = th.device(resolve_device(a.device))
    u0_vec = u0_vector()

    inits = build_inits(jc.line_N, jc.d_min, jc.d_max)
    names = [n for n, _ in inits]
    D0 = th.tensor(np.stack([d for _, d in inits]), dtype=th.float64)

    # 초기해 자체의 수식 성적 (floor 비교: 무작위 64 최선 ≈ −9.6 dB)
    t0 = time.perf_counter()
    p0 = formula_hard_per_restart(oracle, D0, u0_vec)
    t_rank0 = time.perf_counter() - t0
    fam_best = {}
    for n, p in zip(names, p0.tolist()):
        fam = n.split('_')[0]
        if fam not in fam_best or p < fam_best[fam][1]:
            fam_best[fam] = (n, round(p, 2))
    floor = dict(best_init=names[int(p0.argmin())],
                 best_init_psll_db=round(float(p0.min()), 3),
                 n_inits=len(inits))

    # [레인 A] 탐색 0회: 구조 초기해 → top-K → L-BFGS 연마
    tA0 = time.perf_counter()
    top = diverse_topk(D0, p0, a.top_k)
    d_pA, t_pA, n_pA = polish_lbfgs(oracle, d_to_logit(D0[top], jc), jc, u0_vec)
    pA = formula_hard_per_restart(oracle, d_pA, u0_vec)
    tA = time.perf_counter() - tA0 + t_rank0
    pickA = int(pA.argmin())
    laneA = dict(psll_formula_db=round(float(pA[pickA]), 3),
                 elapsed_s=round(tA, 2),
                 n_eval_formula=len(inits) * 5 + n_pA + a.top_k * 5,
                 min_gap_um=round(float(d_pA[pickA].min()), 3),
                 d=d_pA[pickA].tolist())

    # [레인 B] 구조 초기해를 CMA-ES 아일랜드 x0로 (시드별)
    model, _ = load_frozen(cfg, a.ckpt)
    model = model.to(dev)
    fitness = make_fitness(model, oracle, jc, dev)
    topB = diverse_topk(D0, p0, a.islands)
    x0s = [d_to_logit(D0[topB][i:i + 1], jc).squeeze(0).numpy()
           for i in range(a.islands)]
    laneB = {}
    for sd in a.seeds:
        tB0 = time.perf_counter()
        S_best, n_surr = run_cmaes_islands(fitness, x0s, a.sigma0, a.popsize,
                                           a.gens, sd, len(DESIGN_ANGLES_DEG))
        _, d_best = fitness(S_best)
        p_rank = formula_hard_per_restart(oracle, d_best, u0_vec)
        topP = diverse_topk(d_best, p_rank, min(a.top_k, d_best.shape[0]))
        d_pB, t_pB, n_pB = polish_lbfgs(oracle, d_to_logit(d_best[topP], jc),
                                        jc, u0_vec)
        pB = formula_hard_per_restart(oracle, d_pB, u0_vec)
        tB = time.perf_counter() - tB0
        pickB = int(pB.argmin())
        laneB[f'seed{sd}'] = dict(
            psll_formula_db=round(float(pB[pickB]), 3),
            elapsed_s=round(tB, 2),
            n_eval=dict(surrogate=n_surr,
                        formula=(d_best.shape[0] + d_pB.shape[0]) * 5 + n_pB),
            min_gap_um=round(float(d_pB[pickB].min()), 3),
            d=d_pB[pickB].tolist())

    out = dict(tag=a.tag, ckpt=a.ckpt,
               recipe=dict(top_k=a.top_k, islands=a.islands,
                           popsize=a.popsize, gens=a.gens, sigma0=a.sigma0,
                           polish='lbfgs'),
               init_floor=floor, family_best=fam_best,
               laneA_no_explore=laneA, laneB_cmaes_structured=laneB,
               reference=dict(imp4_random_x0={'s42': -13.480, 's7': -13.208},
                              imp1_adam=-13.402, ga=-11.17))
    jp = os.path.join(HERE, 'results', f'{a.tag}.json')
    with open(jp, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    best_lane = min(laneB.values(), key=lambda r: r['psll_formula_db'])
    src = best_lane if best_lane['psll_formula_db'] < laneA['psll_formula_db'] \
        else laneA
    np.savetxt(os.path.join(HERE, 'results', f'final_spacing_{a.tag}.csv'),
               np.array(src['d']), header='d_um', comments='')
    slim = json.loads(json.dumps(out))
    slim['laneA_no_explore'].pop('d')
    for v in slim['laneB_cmaes_structured'].values():
        v.pop('d')
    print(json.dumps(slim, indent=2, ensure_ascii=False))
    print('→', jp)


if __name__ == '__main__':
    main()
