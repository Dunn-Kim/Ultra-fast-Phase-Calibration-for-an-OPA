# 3자 대결 — 동결 서러게이트 경유 vs 수식 직접 vs GA (동일 평가예산, 성능·시간)
#
# lane 의미: lane_surrogate/lane_hybrid = ② (모사 경유, 현 브랜치 방식),
#   lane_formula = ① (수식+Adam 직결, main 브랜치 방식의 배치판 기준선),
#   lane_ga = 대체 대상 기준선.
#
# 공통 프로토콜 (optimize_spacing.py 규약 승계):
#   설계각 ±15° 5각 {0, ±7.5, ±15}, φ = φ*(d,u0) closed-form, 박스 [2,5] sigmoid,
#   커플링 페널티 동일 적용, 판정 = 수식(오라클) hard PSLL worst-angle (최종 심판)
# 예산 등식: restarts·epochs·A == pop·gens·A  (패턴 평가 횟수 기준)
#
# 방법별 실행 장치 (정직 고지):
#   surrogate — MPS f32 배치 멀티스타트 (전 재시작 동시 Adam — 속도 주장 본체)
#   formula   — CPU f64 배치 (현행 Opt4 생산 경로 그대로; MPS 이식 시 격차 축소 여지)
#   GA        — CPU f64 배치 평가 (무경사 — 포스터 시절 대체 대상 기준선)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import os
import json
import math
import time
import argparse
import numpy as np
import torch as th

from config import SurrogateConfig
from physics_oracle import PhysicsOracle
from train import psll_db, resolve_device
from optimize_spacing import DESIGN_ANGLES_DEG, load_frozen

HERE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))   # 산출물은 상위 results/ 로 통일


def u0_vector(dtype=th.float64):
    return th.tensor([math.sin(math.radians(a)) for a in DESIGN_ANGLES_DEG],
                     dtype=dtype)


def batched_phi_star(k, d, u0_vec):
    # d(R,N-1) → 재시작×각도 평탄화: dRA(R·A,N-1), φ*(R·A,N), u0RA(R·A,)
    R, A = d.shape[0], u0_vec.shape[0]
    zero = th.zeros(R, 1, dtype=d.dtype, device=d.device)
    x = th.cat([zero, th.cumsum(d, dim=1)], dim=1)
    dRA = d.repeat_interleave(A, dim=0)
    xRA = x.repeat_interleave(A, dim=0)
    u0RA = u0_vec.to(d.dtype).to(d.device).repeat(R)
    return dRA, k * xRA * u0RA.unsqueeze(1), u0RA


def soft_psll_per_restart(I, u, u0RA, L_RA, beta, R, A, gamma=1.0, eps=1e-12):
    # I(R·A,G) → 재시작별 soft worst-angle 손실 (R,) — optimize_spacing.soft_psll 배치판
    du = 2.0 * 1.55 / L_RA
    m = (u.reshape(1, -1) - u0RA.reshape(-1, 1)).abs() < du.reshape(-1, 1)
    main = th.where(m, I, th.zeros_like(I)).amax(dim=1)
    side_db = 10.0 * th.log10(th.where(m, th.full_like(I, eps), I).clamp(min=eps)
                              / main.clamp(min=eps).unsqueeze(1))
    per = (th.logsumexp(beta * side_db, dim=1) / beta).reshape(R, A)
    return th.logsumexp(gamma * per, dim=1) / gamma


def coupling_penalty_batched(d, jc):
    # optimize_spacing.coupling_penalty_d 의 재시작별 벡터판 (R,)
    phys = th.exp(-jc.cpl_gamma * (d - jc.element_width)).mean(dim=1)
    barrier = th.nn.functional.softplus((jc.cpl_d_safe - d) / jc.cpl_tau
                                        ).pow(2).mean(dim=1)
    return jc.cpl_w_phys * phys + jc.cpl_w_barrier * barrier


def s_init(jc, restarts, gen):
    p0 = (jc.d_init - jc.d_min) / (jc.d_max - jc.d_min)
    s0 = math.log(p0 / (1.0 - p0))
    return s0 + 0.1 * th.randn(restarts, jc.line_N - 1, generator=gen,
                               dtype=th.float64)


@th.no_grad()
def formula_hard_per_restart(oracle, d, u0_vec):
    # 각 재시작 설계의 수식 hard PSLL worst-angle (R,) — 최종 심판
    dRA, phi, u0RA = batched_phi_star(oracle.k, d.double().cpu(), u0_vec)
    I = oracle.intensity(dRA, phi)
    L_RA = dRA.sum(dim=1)
    p = psll_db(I, oracle.u, u0RA, L_RA)
    return p.reshape(d.shape[0], u0_vec.shape[0]).amax(dim=1)


def run_gradient(eval_I, params_dtype, dev, oracle, restarts, epochs, lr,
                 use_cpl, seed, u_grid, s_start=None, beta_fix=None,
                 trace_every=0, trace_fn=None):
    # 공통 경사 루프 — eval_I(dRA, phi) 만 갈아끼움 (서러게이트/수식/연마)
    # trace: trace_every 에포크마다 trace_fn(ep, t_train, d, loss_r) 호출.
    #   계측용 평가 시간은 t_train 누적에서 제외 → 보고 타이밍 공정성 유지
    jc = oracle.jcfg
    g = th.Generator().manual_seed(seed)
    if s_start is None:
        s = s_init(jc, restarts, g)
    else:
        s = s_start.clone()          # 하이브리드 연마: top-K 로짓에서 재출발
    s = s.to(params_dtype).to(dev).requires_grad_(True)
    opt = th.optim.Adam([s], lr=lr)
    u0_vec = u0_vector(params_dtype).to(dev)
    A = u0_vec.shape[0]
    n_eval = 0
    t_train = 0.0
    for ep in range(epochs):
        t0 = time.perf_counter()
        beta = beta_fix if beta_fix is not None else \
            0.2 + (2.0 - 0.2) * min(1.0, ep / (0.6 * epochs))
        d = jc.d_min + (jc.d_max - jc.d_min) * th.sigmoid(s)
        dRA, phi, u0RA = batched_phi_star(oracle.k, d, u0_vec)
        I = eval_I(dRA, phi)
        loss_r = soft_psll_per_restart(I, u_grid, u0RA, dRA.sum(dim=1),
                                       beta, restarts, A)
        if use_cpl:
            loss_r = loss_r + coupling_penalty_batched(d, jc)
        loss_r.sum().backward()      # 재시작별 독립 (s 행 분리 + Adam 원소별)
        opt.step()
        opt.zero_grad(set_to_none=True)
        n_eval += restarts * A
        t_train += time.perf_counter() - t0
        if trace_fn is not None and trace_every \
                and (ep % trace_every == 0 or ep == epochs - 1):
            with th.no_grad():
                d_tr = (jc.d_min + (jc.d_max - jc.d_min)
                        * th.sigmoid(s)).cpu().double()
            trace_fn(ep, t_train, d_tr, loss_r.detach().cpu())
    with th.no_grad():
        # MPS는 f64 미지원 → CPU 이동 후 캐스팅
        d = (jc.d_min + (jc.d_max - jc.d_min) * th.sigmoid(s)).cpu().double()
    return d, t_train, n_eval


def lane_surrogate(cfg, ckpt, oracle, restarts, epochs, lr, use_cpl, seed, dev,
                   trace_every=0):
    model, _ = load_frozen(cfg, ckpt)
    model = model.to(dev)
    u_grid = oracle.u.float().to(dev)

    def eval_I(dRA, phi):
        pred = model(dRA, phi)
        return pred[:, 0] ** 2 + pred[:, 1] ** 2

    def surrogate_hard_best(d_now):
        # 서러게이트 자기판정 hard PSLL (R,) — gap 궤적용
        with th.no_grad():
            dRA, phi, u0RA = batched_phi_star(oracle.k, d_now.float().to(dev),
                                              u0_vector(th.float32).to(dev))
            pred = model(dRA, phi)
            I_s = (pred[:, 0] ** 2 + pred[:, 1] ** 2).cpu().double()
            p = psll_db(I_s, oracle.u, u0RA.cpu().double(),
                        dRA.sum(dim=1).cpu().double())
        return p.reshape(d_now.shape[0], -1).amax(dim=1)

    trace = []

    def tf(ep, t, d_now, loss_r):
        p_f = formula_hard_per_restart(oracle, d_now, u0_vector())
        p_s = surrogate_hard_best(d_now)
        trace.append(dict(ep=ep, t=round(t, 4),
                          loss_best=float(loss_r.min()),
                          loss_mean=float(loss_r.mean()),
                          psll_formula_best=float(p_f.min()),
                          psll_surrogate_best=float(p_s.min())))

    d, elapsed, n_eval = run_gradient(eval_I, th.float32, dev, oracle,
                                      restarts, epochs, lr, use_cpl, seed,
                                      u_grid, trace_every=trace_every,
                                      trace_fn=tf if trace_every else None)
    # 선택은 서러게이트 심판(자기 기준), 보고는 수식 심판 병기 → 괴리 = 악용 신호
    u0_vec = u0_vector()
    with th.no_grad():
        dRA, phi, u0RA = batched_phi_star(oracle.k, d.float().to(dev),
                                          u0_vector(th.float32).to(dev))
        pred = model(dRA, phi)
        I_s = (pred[:, 0] ** 2 + pred[:, 1] ** 2).cpu().double()
        p_s = psll_db(I_s, oracle.u, u0RA.cpu().double(),
                      dRA.sum(dim=1).cpu().double())
        psll_s = p_s.reshape(restarts, -1).amax(dim=1)
    psll_f = formula_hard_per_restart(oracle, d, u0_vec)
    pick = int(psll_s.argmin())
    out = dict(d=d[pick].tolist(), elapsed_s=elapsed, n_eval=n_eval,
               psll_surrogate_db=float(psll_s[pick]),
               psll_formula_db=float(psll_f[pick]),
               psll_formula_best_any_db=float(psll_f.min()),
               gap_db=float(psll_f[pick] - psll_s[pick]),
               device=str(dev))
    if trace_every:
        out['trace'] = trace
    return out


def lane_hybrid(cfg, ckpt, oracle, restarts, epochs, lr, use_cpl, seed, dev,
                top_k=8, polish_epochs=100, trace_every=0):
    # 처방: 서러게이트(MPS) 전역 탐색 → 수식 심판 재순위 → top-K만 수식 연마
    #   off-manifold 악용 제거 (최종해는 수식 경사로 수렴) + 속도 대부분 보존
    model, _ = load_frozen(cfg, ckpt)
    model = model.to(dev)
    u_grid = oracle.u.float().to(dev)

    def eval_I_s(dRA, phi):
        pred = model(dRA, phi)
        return pred[:, 0] ** 2 + pred[:, 1] ** 2

    tr_s, tr_p = [], []

    def make_tf(store):
        def tf(ep, t, d_now, loss_r):
            p_f = formula_hard_per_restart(oracle, d_now, u0_vector())
            store.append(dict(ep=ep, t=round(t, 4),
                              loss_best=float(loss_r.min()),
                              psll_formula_best=float(p_f.min())))
        return tf

    d_s, t_s, n_s = run_gradient(eval_I_s, th.float32, dev, oracle,
                                 restarts, epochs, lr, use_cpl, seed, u_grid,
                                 trace_every=trace_every,
                                 trace_fn=make_tf(tr_s) if trace_every else None)
    t0 = time.perf_counter()
    psll_rank = formula_hard_per_restart(oracle, d_s, u0_vector())
    top = psll_rank.argsort()[:top_k]
    jc = oracle.jcfg
    p = ((d_s[top] - jc.d_min) / (jc.d_max - jc.d_min)).clamp(1e-4, 1.0 - 1e-4)
    s0 = th.log(p / (1.0 - p))
    t_rank = time.perf_counter() - t0

    def eval_I_f(dRA, phi):
        return oracle.intensity(dRA, phi)

    d_p, t_p, n_p = run_gradient(eval_I_f, th.float64, 'cpu', oracle,
                                 top_k, polish_epochs, 0.3 * lr, use_cpl,
                                 seed + 1, oracle.u, s_start=s0, beta_fix=2.0,
                                 trace_every=max(1, trace_every // 2)
                                 if trace_every else 0,
                                 trace_fn=make_tf(tr_p) if trace_every else None)
    psll_f = formula_hard_per_restart(oracle, d_p, u0_vector())
    pick = int(psll_f.argmin())
    n_rank = d_s.shape[0] * len(DESIGN_ANGLES_DEG)
    out = dict(d=d_p[pick].tolist(),
               elapsed_s=t_s + t_rank + t_p,
               breakdown_s=dict(surrogate=round(t_s, 2), rank=round(t_rank, 2),
                                polish=round(t_p, 2)),
               n_eval=dict(surrogate=n_s, formula=n_p + n_rank),
               top_k=top_k, polish_epochs=polish_epochs,
               psll_formula_db=float(psll_f[pick]),
               device=f'{dev}+cpu')
    if trace_every:
        # 연마 궤적의 t 는 탐색+재순위 종료 시점 기준으로 이어붙임
        out['trace_surrogate'] = tr_s
        out['trace_polish'] = [dict(r, t=round(r['t'] + t_s + t_rank, 4),
                                    ep=epochs + r['ep']) for r in tr_p]
    return out


def lane_formula(oracle, restarts, epochs, lr, use_cpl, seed, trace_every=0):
    u_grid = oracle.u

    def eval_I(dRA, phi):
        return oracle.intensity(dRA, phi)

    trace = []

    def tf(ep, t, d_now, loss_r):
        p_f = formula_hard_per_restart(oracle, d_now, u0_vector())
        trace.append(dict(ep=ep, t=round(t, 4),
                          loss_best=float(loss_r.min()),
                          loss_mean=float(loss_r.mean()),
                          psll_formula_best=float(p_f.min())))

    d, elapsed, n_eval = run_gradient(eval_I, th.float64, 'cpu', oracle,
                                      restarts, epochs, lr, use_cpl, seed,
                                      u_grid, trace_every=trace_every,
                                      trace_fn=tf if trace_every else None)
    psll_f = formula_hard_per_restart(oracle, d, u0_vector())
    pick = int(psll_f.argmin())
    out = dict(d=d[pick].tolist(), elapsed_s=elapsed, n_eval=n_eval,
               psll_formula_db=float(psll_f[pick]), device='cpu')
    if trace_every:
        out['trace'] = trace
    return out


def lane_ga(oracle, pop, gens, use_cpl, seed, sigma=0.15, tour=2, elite=2,
            trace_every=0):
    # 실수코딩 GA: 토너먼트 선발, 균등 교차, 가우시안 돌연변이, 엘리트 보존
    jc = oracle.jcfg
    g = th.Generator().manual_seed(seed)
    u0_vec = u0_vector()
    lo, hi = jc.d_min, jc.d_max
    P = lo + (hi - lo) * th.rand(pop, jc.line_N - 1, generator=g,
                                 dtype=th.float64)
    n_eval = 0
    trace = []
    t0 = time.perf_counter()

    def fitness(D):
        # (적합도 = PSLL + 커플링페널티, 순수 PSLL) 동시 반환 — 궤적 기록 무비용
        nonlocal n_eval
        psll = formula_hard_per_restart(oracle, D, u0_vec)
        n_eval += D.shape[0] * u0_vec.shape[0]
        f = psll + coupling_penalty_batched(D, jc) if use_cpl else psll
        return f, psll

    def rec(gen, fit, psll):
        if trace_every and (gen % trace_every == 0 or gen == gens - 1):
            trace.append(dict(ep=gen, t=round(time.perf_counter() - t0, 4),
                              loss_best=float(fit.min()),
                              psll_formula_best=float(psll.min())))

    fit, psll = fitness(P)
    rec(0, fit, psll)
    for gen in range(1, gens):
        order = fit.argsort()
        newP = [P[order[:elite]].clone()]
        n_child = pop - elite
        i = th.randint(pop, (n_child, tour), generator=g)
        parents_a = P[i.gather(1, fit[i].argmin(dim=1, keepdim=True)).squeeze(1)]
        j = th.randint(pop, (n_child, tour), generator=g)
        parents_b = P[j.gather(1, fit[j].argmin(dim=1, keepdim=True)).squeeze(1)]
        mask = th.rand(n_child, jc.line_N - 1, generator=g, dtype=th.float64) < 0.5
        child = th.where(mask, parents_a, parents_b)
        mut = sigma * th.randn(n_child, jc.line_N - 1, generator=g,
                               dtype=th.float64)
        child = (child + mut).clamp(lo, hi)
        P = th.cat([newP[0], child], dim=0)
        fit, psll = fitness(P)
        rec(gen, fit, psll)
    elapsed = time.perf_counter() - t0
    # 판정 통일: 페널티 제외 순수 수식 PSLL로 최종 보고
    psll_f = formula_hard_per_restart(oracle, P, u0_vec)
    pick = int(psll_f.argmin())
    out = dict(d=P[pick].tolist(), elapsed_s=elapsed, n_eval=n_eval,
               psll_formula_db=float(psll_f[pick]), device='cpu')
    if trace_every:
        out['trace'] = trace
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='mlp', choices=['mlp', 'element'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--restarts', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--no-cpl', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--tag', default='duel')
    ap.add_argument('--trace', action='store_true',
                    help='에포크/세대별 loss·PSLL 궤적 기록 (타이밍에서 제외)')
    a = ap.parse_args()
    tr = 10 if a.trace else 0
    use_cpl = not a.no_cpl
    cfg = SurrogateConfig(arch=a.arch)
    oracle = PhysicsOracle()
    dev = th.device(resolve_device(a.device))

    out = dict(protocol=dict(design_angles_deg=list(DESIGN_ANGLES_DEG),
                             restarts=a.restarts, epochs=a.epochs, lr=a.lr,
                             use_cpl=use_cpl, seed=a.seed,
                             budget_pattern_evals=a.restarts * a.epochs
                             * len(DESIGN_ANGLES_DEG)))
    print('[1/4] surrogate (batched multistart, %s)…' % dev, flush=True)
    out['surrogate'] = lane_surrogate(cfg, a.ckpt, oracle, a.restarts,
                                      a.epochs, a.lr, use_cpl, a.seed, dev,
                                      trace_every=tr)
    print('     PSLL(수식) %.2f dB | %.1fs' % (
        out['surrogate']['psll_formula_db'],
        out['surrogate']['elapsed_s']), flush=True)
    print('[2/4] hybrid (surrogate 탐색 → top-K 수식 연마)…', flush=True)
    out['hybrid'] = lane_hybrid(cfg, a.ckpt, oracle, a.restarts, a.epochs,
                                a.lr, use_cpl, a.seed, dev, trace_every=tr)
    print('     PSLL(수식) %.2f dB | %.1fs' % (
        out['hybrid']['psll_formula_db'],
        out['hybrid']['elapsed_s']), flush=True)
    print('[3/4] formula direct (CPU f64)…', flush=True)
    out['formula'] = lane_formula(oracle, a.restarts, a.epochs, a.lr,
                                  use_cpl, a.seed, trace_every=tr)
    print('     PSLL(수식) %.2f dB | %.1fs' % (
        out['formula']['psll_formula_db'],
        out['formula']['elapsed_s']), flush=True)
    print('[4/4] GA (CPU f64)…', flush=True)
    out['ga'] = lane_ga(oracle, a.restarts, a.epochs, use_cpl, a.seed,
                        trace_every=max(1, tr // 5) if a.trace else 0)
    print('     PSLL(수식) %.2f dB | %.1fs' % (
        out['ga']['psll_formula_db'], out['ga']['elapsed_s']), flush=True)

    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    jp = os.path.join(HERE, 'results', f'benchmark_{a.tag}.json')
    with open(jp, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for name in ('surrogate', 'hybrid', 'formula', 'ga'):
        d = np.array(out[name]['d'])
        np.savetxt(os.path.join(HERE, 'results', f'spacing_{a.tag}_{name}.csv'),
                   d, header='d_um', comments='')
    hide = {'d', 'trace', 'trace_surrogate', 'trace_polish'}
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in hide}
                      for k, v in out.items() if k != 'protocol'},
                     indent=2, ensure_ascii=False))
    print('→', jp)


if __name__ == '__main__':
    main()
