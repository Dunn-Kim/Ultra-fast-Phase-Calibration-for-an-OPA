# 등간격(d=3µm) 기준 — phase만 서러게이트 경유로 최적화하는 증빙 실험
#
# 요구 매트릭스의 남은 셀: "등간격 phase × 모사 모델 경유".
# 랜덤 위상에서 출발, 동결 서러게이트의 목표각 강도를 직접 상승시키고
# 판정은 수식으로: (a) 주엽이 u0에 형성되는가, (b) 해석해 φ* 대비 주엽 전력비,
# (c) hard PSLL 이 φ* 수준(등간격 상한)에 도달하는가.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import os
import json
import math
import argparse
import torch as th

from config import SurrogateConfig
from train import psll_db
from optimize_spacing import load_frozen

HERE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))   # 산출물은 상위 results/ 로 통일


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/mlp_full.pt')
    ap.add_argument('--angles', type=float, nargs='+',
                    default=[0.0, 10.0, 20.0, 30.0])
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    cfg = SurrogateConfig(arch='mlp')
    model, oracle = load_frozen(cfg, a.ckpt)
    jc = oracle.jcfg
    N = jc.line_N
    d = th.full((1, N - 1), jc.d_init, dtype=th.float64)
    g = th.Generator().manual_seed(a.seed)

    rows = []
    for ang in a.angles:
        u0 = math.sin(math.radians(ang))
        idx = int((oracle.u - u0).abs().argmin())
        phi = (2.0 * math.pi * th.rand(1, N, generator=g, dtype=th.float64)
               ).float().requires_grad_(True)
        opt = th.optim.Adam([phi], lr=a.lr)
        for _ in range(a.steps):
            pred = model(d.float(), phi)
            I = pred[:, 0] ** 2 + pred[:, 1] ** 2
            loss = -I[0, idx]                     # 서러게이트 목표각 강도 극대화
            opt.zero_grad()
            loss.backward()
            opt.step()
        phi_opt = phi.detach().double()
        phi_star = oracle.steering_phase(d, th.tensor([u0], dtype=th.float64))
        with th.no_grad():
            I_opt = oracle.intensity(d, phi_opt)          # 판정은 수식으로
            I_star = oracle.intensity(d, phi_star)
        u_pk = float(oracle.u[int(I_opt[0].argmax())])
        L = th.tensor([float(d.sum())], dtype=th.float64)
        u0t = th.tensor([u0], dtype=th.float64)
        rows.append(dict(
            angle_deg=ang,
            peak_err_deg=round(math.degrees(math.asin(u_pk) - math.asin(u0)), 3),
            main_power_ratio=round(float(I_opt[0, idx] / I_star[0, idx]), 4),
            psll_opt_db=round(float(psll_db(I_opt, oracle.u, u0t, L)), 2),
            psll_star_db=round(float(psll_db(I_star, oracle.u, u0t, L)), 2)))
        print(rows[-1], flush=True)

    out = dict(ckpt=a.ckpt, steps=a.steps, lr=a.lr,
               note='등간격 d=3, 랜덤 위상 출발 → 서러게이트 경유 φ 최적화, 수식 판정',
               rows=rows)
    with open(os.path.join(HERE, 'results', 'phase_via_surrogate.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
