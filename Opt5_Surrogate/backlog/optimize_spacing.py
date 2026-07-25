# fine-tune 된 서러게이트를 동결하고 간격만 최적화 (2단계 파이프라인의 최종 단계)
#
# 순수 ②형: 최적화 루프에서 수식 호출 0회 — 동결 모사 모델만으로 간격 변수 조작.
#   (champion_design.py 는 여기에 수식 재순위·연마를 더한 실전형)
#
# 변수: 간격 로짓 s → d = d_min + (d_max−d_min)·sigmoid(s)  (박스 [2,5] 하드 보장, Opt4 규약)
# 위상: φ = φ*(d, u0) = k·xₙ·u0 closed-form 유지 (탐색 차원 절반 — 조정 변수는 간격 하나)
# 손실: 다각도(±15° 프로토콜) soft-PSLL worst-case + 커플링 페널티(물리항 — 수식에서 직접)
# 평가: hard PSLL을 (a) 서러게이트 기준, (b) 수식(오라클) 기준 이중 보고
#   — MLP: (a)−(b) 괴리 = off-manifold 악용 신호.  element(FT후): 괴리 = 학습된 MODE 보정(의도).
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
from model import build_model
from train import psll_db

HERE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))   # 산출물은 상위 results/ 로 통일
DESIGN_ANGLES_DEG = (0.0, 7.5, -7.5, 15.0, -15.0)


def load_frozen(cfg, ckpt_path):
    oracle = PhysicsOracle()
    model = build_model(cfg, oracle)
    model.load_state_dict(th.load(ckpt_path, map_location='cpu')['state'])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, oracle


def coupling_penalty_d(d, jc):
    # Opt4 losses.coupling_penalty 와 동일식 (d 직접 인자화)
    phys = th.exp(-jc.cpl_gamma * (d - jc.element_width)).mean()
    barrier = th.nn.functional.softplus((jc.cpl_d_safe - d) / jc.cpl_tau).pow(2).mean()
    return jc.cpl_w_phys * phys + jc.cpl_w_barrier * barrier


def surrogate_intensity(model, oracle, d, u0_vec):
    # d(31,) → 각도별 φ* 적용한 강도 (A, 1801). φ*는 d를 통해 미분 가능.
    A = u0_vec.shape[0]
    dB = d.unsqueeze(0).expand(A, -1)
    zero = th.zeros(A, 1, dtype=d.dtype)
    x = th.cat([zero, th.cumsum(dB, dim=1)], dim=1)
    phi = oracle.k * x * u0_vec.reshape(-1, 1)
    pred = model(dB, phi)
    return (pred[:, 0] ** 2 + pred[:, 1] ** 2).double(), phi


def soft_psll(I, u, u0_vec, L_ap, beta, gamma=1.0, eps=1e-12):
    # 각도별 soft-PSLL[dB] → 각도 간 soft worst-case (LSE)
    du = 2.0 * 1.55 / L_ap
    m = (u.reshape(1, -1) - u0_vec.reshape(-1, 1)).abs() < du
    main = th.where(m, I, th.zeros_like(I)).amax(dim=1)
    side_db = 10.0 * th.log10(th.where(m, th.full_like(I, eps), I).clamp(min=eps)
                              / main.clamp(min=eps).unsqueeze(1))
    per_angle = th.logsumexp(beta * side_db, dim=1) / beta
    return th.logsumexp(gamma * per_angle, dim=0) / gamma


@th.no_grad()
def hard_report(model, oracle, d, u0_vec):
    # 서러게이트/수식 이중 hard PSLL (worst-angle)
    I_s, phi = surrogate_intensity(model, oracle, d, u0_vec)
    dB_ = d.unsqueeze(0).expand(u0_vec.shape[0], -1)
    I_f = oracle.intensity(dB_, phi)
    L = d.sum()
    Ls = th.full_like(u0_vec, float(L))
    p_s = psll_db(I_s, oracle.u, u0_vec, Ls)
    p_f = psll_db(I_f, oracle.u, u0_vec, Ls)
    return p_s.max().item(), p_f.max().item(), p_s.tolist(), p_f.tolist()


def optimize(cfg: SurrogateConfig, ckpt, restarts=8, epochs=400, lr=1e-2,
             use_cpl=True, seed=42, tag=None):
    model, oracle = load_frozen(cfg, ckpt)
    jc = oracle.jcfg
    u0_vec = th.tensor([math.sin(math.radians(a)) for a in DESIGN_ANGLES_DEG],
                       dtype=th.float64)
    g = th.Generator().manual_seed(seed)
    p0 = (jc.d_init - jc.d_min) / (jc.d_max - jc.d_min)
    s0 = math.log(p0 / (1.0 - p0))

    best = None
    t0 = time.time()
    n_eval = 0
    for r in range(restarts):
        s = (s0 + 0.1 * th.randn(jc.line_N - 1, generator=g, dtype=th.float64)
             ).requires_grad_(True)
        opt = th.optim.Adam([s], lr=lr)
        for ep in range(epochs):
            beta = 0.2 + (2.0 - 0.2) * min(1.0, ep / (0.6 * epochs))
            d = jc.d_min + (jc.d_max - jc.d_min) * th.sigmoid(s)
            I_s, _ = surrogate_intensity(model, oracle, d, u0_vec)
            loss = soft_psll(I_s, oracle.u, u0_vec, d.sum(), beta)
            if use_cpl:
                loss = loss + coupling_penalty_d(d, jc)
            opt.zero_grad()
            loss.backward()
            opt.step()
            n_eval += u0_vec.shape[0]
        with th.no_grad():
            d = jc.d_min + (jc.d_max - jc.d_min) * th.sigmoid(s)
            ps, pf, ps_all, pf_all = hard_report(model, oracle, d, u0_vec)
        if best is None or ps < best['psll_surrogate_db']:
            best = dict(restart=r, d=d.tolist(),
                        psll_surrogate_db=ps, psll_formula_db=pf,
                        psll_surrogate_per_angle=ps_all,
                        psll_formula_per_angle=pf_all)
    elapsed = time.time() - t0
    d_best = th.tensor(best['d'], dtype=th.float64)
    tag = tag or os.path.splitext(os.path.basename(ckpt))[0]
    out = dict(tag=tag, ckpt=ckpt, restarts=restarts, epochs=epochs,
               use_cpl=use_cpl, design_angles_deg=list(DESIGN_ANGLES_DEG),
               elapsed_s=round(elapsed, 2), n_surrogate_evals=n_eval,
               min_gap_um=float(d_best.min()), aperture_um=float(d_best.sum()),
               **best)
    os.makedirs(os.path.join(HERE, cfg.results_dir), exist_ok=True)
    jp = os.path.join(HERE, cfg.results_dir, f'optimize_{tag}.json')
    with open(jp, 'w') as f:
        json.dump(out, f, indent=2)
    np.savetxt(os.path.join(HERE, cfg.results_dir, f'spacing_{tag}.csv'),
               np.array(best['d']), header='d_um', comments='')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='element', choices=['mlp', 'element'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--restarts', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--no-cpl', action='store_true')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    cfg = SurrogateConfig(arch=a.arch)
    out = optimize(cfg, a.ckpt, restarts=a.restarts, epochs=a.epochs,
                   use_cpl=not a.no_cpl, tag=a.tag)
    print(json.dumps(out, indent=2))
