# 챔피언 설계 — 목표: GA 대비 더 빠르고(벽시계) 더 낮은(loss·PSLL) 최종 간격 산출
#
# 브랜치 정의(②): 수식을 모사한 동결 모델을 경유해 간격 변수만 Adam으로 조작.
#   (main 브랜치의 ①은 수식에 Adam 직결 — 여기서는 lane 비교의 기준선으로만 등장)
#   모델 가중치는 전 과정 불변(동결 게이트). 수식 직접 호출은 [2] 재순위와
#   [3] 연마에 한정 — 전체 수식 평가의 ~15%.
#
# 레시피 (hybrid-XL):
#   [1] 탐색  — FT-서러게이트 MPS 배치 멀티스타트 256×500ep (배치라 시간 증가 미미)
#   [2] 재순위 — 전 재시작을 수식 hard PSLL로 판정 + 다양성 필터 top-K
#              (같은 분지로 수렴한 중복 설계 제거 → 연마 예산을 서로 다른 분지에 배분)
#   [3] 연마  — top-K만 수식 직접 Adam 2단 (β 1.5 → 3.0, lr 3e-3 → 1e-3)
#              β 상향 = soft-PSLL이 hard max에 근접 → 최악 사이드로브 직접 압박
#   [4] 판정  — 수식 hard PSLL worst-angle (±15° 5각 프로토콜, 커플링 페널티 준수 확인)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import os
import json
import time
import argparse
import numpy as np
import torch as th

from config import SurrogateConfig
from physics_oracle import PhysicsOracle
from train import psll_db, resolve_device
from optimize_spacing import DESIGN_ANGLES_DEG, load_frozen, coupling_penalty_d
from benchmark import (u0_vector, batched_phi_star, formula_hard_per_restart,
                       run_gradient, coupling_penalty_batched,
                       soft_psll_per_restart)

HERE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))   # 산출물은 상위 results/ 로 통일


def diverse_topk(d, scores, k, min_dist=0.5):
    # 점수순 그리디 선택, 기선택 설계와 L2 거리 min_dist[µm] 미만이면 건너뜀
    order = scores.argsort().tolist()
    picked = []
    for i in order:
        if all(float((d[i] - d[j]).norm()) > min_dist for j in picked):
            picked.append(i)
        if len(picked) == k:
            break
    for i in order:                     # 부족분은 점수순 보충
        if len(picked) == k:
            break
        if i not in picked:
            picked.append(i)
    return th.tensor(picked, dtype=th.long)


def d_to_logit(d, jc, eps=1e-4):
    p = ((d - jc.d_min) / (jc.d_max - jc.d_min)).clamp(eps, 1.0 - eps)
    return th.log(p / (1.0 - p))


def explore_ensemble(models, oracle, jc, restarts, epochs, lr, seed, dev,
                     lam, u_dev):
    # IMP2: 앙상블 탐색 — 손실 = 모델별 soft-PSLL 평균 + λ·표준편차(불일치 페널티)
    #   모델 간 불일치가 큰 영역 = 모사 신뢰 밖 → 악용(off-manifold) 지대 회피 유도
    g = th.Generator().manual_seed(seed)
    from benchmark import s_init
    s = s_init(jc, restarts, g).float().to(dev).requires_grad_(True)
    opt = th.optim.Adam([s], lr=lr)
    u0_vec = u0_vector(th.float32).to(dev)
    A = u0_vec.shape[0]
    K = len(models)
    n_eval = 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        beta = 0.2 + (2.0 - 0.2) * min(1.0, ep / (0.6 * epochs))
        d = jc.d_min + (jc.d_max - jc.d_min) * th.sigmoid(s)
        dRA, phi, u0RA = batched_phi_star(oracle.k, d, u0_vec)
        L_RA = dRA.sum(dim=1)
        pls = []
        for m in models:
            pred = m(dRA, phi)
            I = pred[:, 0] ** 2 + pred[:, 1] ** 2
            pls.append(soft_psll_per_restart(I, u_dev, u0RA, L_RA, beta,
                                             restarts, A))
        P = th.stack(pls)                                  # (K, R)
        loss_r = (P.mean(dim=0) + lam * P.std(dim=0)
                  + coupling_penalty_batched(d, jc))
        loss_r.sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        n_eval += K * restarts * A
    elapsed = time.perf_counter() - t0
    with th.no_grad():
        d = (jc.d_min + (jc.d_max - jc.d_min) * th.sigmoid(s)).cpu().double()
    return d, elapsed, n_eval


def ensemble_self_psll(models, oracle, d, dev):
    # 앙상블 자기판정 hard PSLL (평균 강도 기준) — 순수 ② gap 지표용
    u0v = u0_vector(th.float32).to(dev)
    with th.no_grad():
        dRA, phi, u0RA = batched_phi_star(oracle.k, d.float().to(dev), u0v)
        I = None
        for m in models:
            pred = m(dRA, phi)
            Ik = pred[:, 0] ** 2 + pred[:, 1] ** 2
            I = Ik if I is None else I + Ik
        I = (I / len(models)).cpu().double()
        p = psll_db(I, oracle.u, u0RA.cpu().double(),
                    dRA.sum(dim=1).cpu().double())
    return p.reshape(d.shape[0], -1).amax(dim=1)


def polish_lbfgs(oracle, s0, jc, u0_vec, beta=3.0, max_iter=150):
    # IMP1: 준뉴턴(L-BFGS) 연마 — strong Wolfe 선탐색, β 고정
    #   연마는 국소 수렴 단계 → 2차 곡률로 Adam 수백 에포크를 수십 반복으로 대체
    R, A = s0.shape[0], u0_vec.shape[0]
    s = s0.clone().double().requires_grad_(True)
    opt = th.optim.LBFGS([s], max_iter=max_iter, history_size=25,
                         line_search_fn='strong_wolfe',
                         tolerance_grad=1e-10, tolerance_change=1e-12)
    n_closure = 0

    def closure():
        nonlocal n_closure
        n_closure += 1
        opt.zero_grad()
        d = jc.d_min + (jc.d_max - jc.d_min) * th.sigmoid(s)
        dRA, phi, u0RA = batched_phi_star(oracle.k, d, u0_vec)
        I = oracle.intensity(dRA, phi)
        loss_r = soft_psll_per_restart(I, oracle.u, u0RA, dRA.sum(dim=1),
                                       beta, R, A)
        loss = (loss_r + coupling_penalty_batched(d, jc)).sum()
        loss.backward()
        return loss

    t0 = time.perf_counter()
    opt.step(closure)
    elapsed = time.perf_counter() - t0
    with th.no_grad():
        d = (jc.d_min + (jc.d_max - jc.d_min) * th.sigmoid(s)).double()
    return d, elapsed, n_closure * R * A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='mlp',
                    choices=['mlp', 'element', 'siren', 'ffmlp'])
    ap.add_argument('--ckpt', default='checkpoints/mlp_replay_ft.pt')
    ap.add_argument('--restarts', type=int, default=256)
    ap.add_argument('--explore-epochs', type=int, default=500)
    ap.add_argument('--top-k', type=int, default=12)
    ap.add_argument('--polish-epochs', type=int, default=150)   # 단당 (2단 = ×2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--tag', default='champion_hybrid')
    ap.add_argument('--trace', action='store_true',
                    help='단계별 loss·수식PSLL 궤적 기록 (타이밍 제외)')
    ap.add_argument('--polish', default='adam2', choices=['adam2', 'lbfgs'],
                    help='연마기: adam2=β 2단 Adam(기존), lbfgs=준뉴턴(IMP1)')
    ap.add_argument('--ens-lambda', type=float, default=1.0,
                    help='IMP2 앙상블 불일치 페널티 계수 (ckpt 2개 이상일 때)')
    a = ap.parse_args()

    cfg = SurrogateConfig(arch=a.arch)
    oracle = PhysicsOracle()
    jc = oracle.jcfg
    dev = th.device(resolve_device(a.device))
    u0_vec = u0_vector()

    ckpts = [c.strip() for c in a.ckpt.split(',')]
    models = []
    for cpath in ckpts:
        m, _ = load_frozen(cfg, cpath)
        models.append(m.to(dev))
    model = models[0]
    u_dev = oracle.u.float().to(dev)

    def eval_I_s(dRA, phi):
        pred = model(dRA, phi)
        return pred[:, 0] ** 2 + pred[:, 1] ** 2

    def eval_I_f(dRA, phi):
        return oracle.intensity(dRA, phi)

    tr_x, tr_p1, tr_p2 = [], [], []

    def make_tf(store):
        def tf(ep, t, d_now, loss_r):
            p = formula_hard_per_restart(oracle, d_now, u0_vec)
            store.append(dict(ep=ep, t=round(t, 4),
                              loss_best=float(loss_r.min()),
                              psll_formula_best=float(p.min())))
        return tf

    t_total0 = time.perf_counter()
    # [1] 탐색 (서러게이트, MPS) — ckpt 복수면 IMP2 앙상블 불일치 탐색
    if len(models) > 1:
        d_x, t_x, n_x = explore_ensemble(models, oracle, jc, a.restarts,
                                         a.explore_epochs, 1e-2, a.seed, dev,
                                         a.ens_lambda, u_dev)
    else:
        d_x, t_x, n_x = run_gradient(eval_I_s, th.float32, dev, oracle,
                                     a.restarts, a.explore_epochs, 1e-2,
                                     True, a.seed, u_dev,
                                     trace_every=25 if a.trace else 0,
                                     trace_fn=make_tf(tr_x) if a.trace else None)
    # [2] 수식 재순위 + 다양성 top-K
    t0 = time.perf_counter()
    p_rank = formula_hard_per_restart(oracle, d_x, u0_vec)
    top = diverse_topk(d_x, p_rank, a.top_k)
    s0 = d_to_logit(d_x[top], jc)
    t_rank = time.perf_counter() - t0
    n_rank = d_x.shape[0] * u0_vec.shape[0]
    # 순수 ② gap 지표 — 탐색 종료 설계에서 자기판정(모사) vs 수식판정
    self_p = ensemble_self_psll(models, oracle, d_x, dev)
    b_self = int(self_p.argmin())
    pure2 = dict(psll_self_db=round(float(self_p[b_self]), 3),
                 psll_formula_db=round(float(p_rank[b_self]), 3),
                 gap_db=round(float(p_rank[b_self] - self_p[b_self]), 3),
                 n_models=len(models))
    # [3] 연마 (수식 직접, CPU f64) — adam2: β 2단 Adam / lbfgs: 준뉴턴(IMP1)
    if a.polish == 'lbfgs':
        d_p2, t_p2, n_p2 = polish_lbfgs(oracle, s0, jc, u0_vec)
        t_p1, n_p1 = 0.0, 0
    else:
        d_p1, t_p1, n_p1 = run_gradient(eval_I_f, th.float64, 'cpu', oracle,
                                        a.top_k, a.polish_epochs, 3e-3, True,
                                        a.seed + 1, oracle.u,
                                        s_start=s0, beta_fix=1.5,
                                        trace_every=5 if a.trace else 0,
                                        trace_fn=make_tf(tr_p1)
                                        if a.trace else None)
        d_p2, t_p2, n_p2 = run_gradient(eval_I_f, th.float64, 'cpu', oracle,
                                        a.top_k, a.polish_epochs, 1e-3, True,
                                        a.seed + 2, oracle.u,
                                        s_start=d_to_logit(d_p1, jc),
                                        beta_fix=3.0,
                                        trace_every=5 if a.trace else 0,
                                        trace_fn=make_tf(tr_p2)
                                        if a.trace else None)
    # [4] 판정
    p_fin = formula_hard_per_restart(oracle, d_p2, u0_vec)
    n_judge = d_p2.shape[0] * u0_vec.shape[0]
    pick = int(p_fin.argmin())
    d_c = d_p2[pick]
    t_total = time.perf_counter() - t_total0

    dRA, phi, u0RA = batched_phi_star(oracle.k, d_c.unsqueeze(0), u0_vec)
    I = oracle.intensity(dRA, phi)
    per_angle = psll_db(I, oracle.u, u0RA, dRA.sum(dim=1)).tolist()
    cpl = float(coupling_penalty_d(d_c, jc))

    prev = {}
    bm_path = os.path.join(HERE, 'results', 'benchmark_ftmlp.json')
    if os.path.exists(bm_path):
        bm = json.load(open(bm_path))
        prev = {k: dict(psll_db=round(bm[k]['psll_formula_db'], 2),
                        time_s=round(bm[k]['elapsed_s'], 1))
                for k in ('ga', 'formula', 'hybrid') if k in bm}

    out = dict(
        tag=a.tag, ckpt=a.ckpt,
        recipe=dict(restarts=a.restarts, explore_epochs=a.explore_epochs,
                    top_k=a.top_k, polish=a.polish,
                    polish_epochs_per_stage=a.polish_epochs,
                    beta_stages=(1.5, 3.0) if a.polish == 'adam2'
                    else ('lbfgs', 3.0),
                    diversity_min_dist_um=0.5),
        psll_formula_db=float(p_fin[pick]),
        psll_per_angle_db={f'{ang:+.1f}deg': round(v, 2)
                           for ang, v in zip(DESIGN_ANGLES_DEG, per_angle)},
        elapsed_s=round(t_total, 2),
        breakdown_s=dict(explore=round(t_x, 2), rank=round(t_rank, 2),
                         polish1=round(t_p1, 2), polish2=round(t_p2, 2)),
        n_eval=dict(surrogate=n_x, formula=n_rank + n_p1 + n_p2 + n_judge),
        min_gap_um=float(d_c.min()), max_gap_um=float(d_c.max()),
        aperture_um=float(d_c.sum()), coupling_penalty=round(cpl, 4),
        pure2_gap=pure2,
        ens_lambda=a.ens_lambda if len(models) > 1 else None,
        d=d_c.tolist(),
        baseline=prev,
        vs_ga=dict(
            psll_gain_db=round(prev['ga']['psll_db'] - float(p_fin[pick]), 2)
            if 'ga' in prev else None,
            speedup=round(prev['ga']['time_s'] / t_total, 1)
            if 'ga' in prev else None))
    if a.trace:
        out['trace'] = dict(explore=tr_x, polish1=tr_p1, polish2=tr_p2)

    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    jp = os.path.join(HERE, 'results', f'{a.tag}.json')
    with open(jp, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    np.savetxt(os.path.join(HERE, 'results', f'final_spacing_{a.tag}.csv'),
               np.array(d_c.tolist()), header='d_um', comments='')
    hide = {'d'}
    print(json.dumps({k: v for k, v in out.items() if k not in hide},
                     indent=2, ensure_ascii=False))
    print('→', jp)


if __name__ == '__main__':
    main()
