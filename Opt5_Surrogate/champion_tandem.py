# IMP6: tandem 역설계망 — 사양(조향범위) → 간격을 1회 forward로 산출
#
# 구조 (tandem NN 패러다임): 사양 → InverseNet → 간격 로짓 → 동결 모사(forward) →
#   soft-PSLL + 커플링 페널티. 역전파는 동결 모사를 통과해 InverseNet만 갱신한다.
#   (모사 가중치 불변 — 브랜치 ② 규약. champion 과 동일한 동결 계약)
#
# 기존 champion 과의 차이: champion 은 사양 1건당 전 파이프라인 재실행(수 초).
#   tandem 은 학습 1회를 상각하고, 이후 임의 사양에 ms 급 응답. 품질은 연마로 보정.
#
# 판정 회계 (IMP5 교훈): raw hard PSLL 과 J = PSLL + 커플링 페널티를 반드시 병기.
#
# 기본값 근거:
#   w_isl 0.4 / w_barrier 2.0 — ① 트랙과 달리 tandem 은 한 벌의 가중치로 여러 사양을
#     동시에 만족해야 해 자유도가 적다. w_isl 1.0 은 PSLL 을 0.31 dB 깎았고, 0.4 에서
#     PSLL 무손실로 ISL 1.26→0.98, 최소 간격 2.258→2.312µm, 개구 101→98µm,
#     연마 전 raw J −12.45→−12.62 개선.
import os
import json
import math
import time
import argparse
import numpy as np
import torch as th
import torch.nn as nn

from config import SurrogateConfig
from physics_oracle import PhysicsOracle
from opt_core import (psll_db, resolve_device, load_frozen, coupling_penalty,
                      soft_psll_per_restart, soft_isl_per_restart, du_weights,
                      batched_phi_star, d_to_logit, logit_to_d, polish_lbfgs)

HERE = os.path.dirname(os.path.abspath(__file__))
ANGLE_FRACS = (0.0, 0.5, -0.5, 1.0, -1.0)   # 사양 u_max 에 대한 설계각 배치


def spec_features(u_max):
    # (B,) u_max → (B,4) 입력 특징 (스칼라 사양의 저차원 인코딩)
    u = u_max.reshape(-1, 1)
    return th.cat([u, u ** 2, th.sin(math.pi * u), th.cos(math.pi * u)], dim=1)


class InverseNet(nn.Module):
    def __init__(self, n_gap, hidden=(256, 256, 256)):
        super().__init__()
        dims = [4] + list(hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], n_gap)
        nn.init.zeros_(self.head.weight)          # 등간격 근방에서 출발
        nn.init.zeros_(self.head.bias)

    def forward(self, u_max, s_center):
        return s_center + self.head(self.body(spec_features(u_max)))


def angles_for(u_max):
    # (B,) → (B,A) 설계각 u0 배치
    return u_max.reshape(-1, 1) * th.tensor(ANGLE_FRACS, dtype=u_max.dtype,
                                            device=u_max.device).reshape(1, -1)


def train_inverse(model_s, oracle, jc, dev, steps, batch, lr, seed,
                  u_lo, u_hi, log_every=250, w_isl=0.0):
    th.manual_seed(seed)
    net = InverseNet(jc.line_N - 1).to(dev)
    opt = th.optim.Adam(net.parameters(), lr=lr)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                    eta_min=lr * 0.02)
    s_center = float(d_to_logit(th.tensor(jc.d_init, dtype=th.float64), jc))
    u_grid = oracle.u.float().to(dev)
    dw = du_weights(u_grid) if w_isl else None
    A = len(ANGLE_FRACS)
    hist = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        beta = 0.5 + (3.0 - 0.5) * min(1.0, step / (0.6 * steps))
        u_max = (u_lo + (u_hi - u_lo)
                 * th.rand(batch, device=dev, dtype=th.float32))
        s = net(u_max, s_center)
        d = logit_to_d(s, jc)
        dRA, phi, u0RA = batched_phi_star(oracle.k, d, angles_for(u_max))
        pred = model_s(dRA, phi)
        I = pred[:, 0] ** 2 + pred[:, 1] ** 2
        L_RA = dRA.sum(dim=1)
        loss_r = soft_psll_per_restart(I, u_grid, u0RA, L_RA, beta, batch, A)
        if w_isl:
            loss_r = loss_r + w_isl * soft_isl_per_restart(
                I, u_grid, u0RA, L_RA, batch, A, dw)
        loss = (loss_r + coupling_penalty(d, jc)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            hist.append(dict(step=step, loss=round(float(loss.detach()), 4),
                             beta=round(beta, 2)))
    return net, s_center, time.perf_counter() - t0, hist


def judge(oracle, jc, d, u0_row):
    # 단일 설계 판정 → (raw PSLL, cpl, J)
    dRA, phi, u0RA = batched_phi_star(oracle.k, d.unsqueeze(0), u0_row)
    I = oracle.intensity(dRA, phi)
    p = float(psll_db(I, oracle.u, u0RA, dRA.sum(dim=1)).max())
    c = float(coupling_penalty(d, jc))
    return p, c, p + c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='mlp', choices=['mlp', 'element'])
    ap.add_argument('--ckpt', default='checkpoints/mlp_full.pt')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--u-lo', type=float, default=0.05)
    ap.add_argument('--u-hi', type=float, default=0.55)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--w-isl', type=float, default=0.4,
                    help='ISL 항 가중 (학습·연마 공통) — 0이면 PSLL 단독')
    ap.add_argument('--w-barrier', type=float, default=2.0,
                    help='커플링 barrier 가중 (ISL 항의 하한 압력 상쇄)')
    ap.add_argument('--tag', default='champion_tandem')
    a = ap.parse_args()

    cfg = SurrogateConfig(arch=a.arch)
    oracle = PhysicsOracle()
    jc = oracle.jcfg
    # barrier 상향은 학습·연마 압력 조절용 — 판정은 규약 기본값으로 되돌려 수행
    w_barrier_judge = jc.cpl_w_barrier
    jc.cpl_w_barrier = a.w_barrier
    dev = th.device(resolve_device(a.device))
    model_s, _ = load_frozen(cfg, a.ckpt)
    model_s = model_s.to(dev)

    net, s_center, t_train, hist = train_inverse(
        model_s, oracle, jc, dev, a.steps, a.batch, a.lr, a.seed,
        a.u_lo, a.u_hi, w_isl=a.w_isl)
    net.eval()

    # 평가 사양: ±10/15/20/30° (±15° = champion 프로토콜과 동일)
    specs_deg = [10.0, 15.0, 20.0, 30.0]
    u_specs = th.tensor([math.sin(math.radians(x)) for x in specs_deg],
                        dtype=th.float32)

    # 추론 시간 (사양 1건, 100회 평균) — amortized 설계 응답 시간
    with th.no_grad():
        one = u_specs[:1].to(dev)
        for _ in range(10):
            net(one, s_center)
        if dev.type == 'mps':
            th.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            net(one, s_center)
        if dev.type == 'mps':
            th.mps.synchronize()
        t_infer_ms = (time.perf_counter() - t0) / 100 * 1e3

    with th.no_grad():
        s_all = net(u_specs.to(dev), s_center).cpu().double()
    d_all = logit_to_d(s_all, jc)

    jc.cpl_w_barrier = w_barrier_judge          # 이하 판정·연마 보고는 규약 기준
    rows = []
    for i, deg in enumerate(specs_deg):
        u_row = (u_specs[i].double()
                 * th.tensor(ANGLE_FRACS, dtype=th.float64))
        d_raw = d_all[i]
        p_r, c_r, J_r = judge(oracle, jc, d_raw, u_row)
        # 연마 1: tandem 출력 단일점 L-BFGS (최소 비용 경로)
        t0 = time.perf_counter()
        d_pol, t_pol, n_pol = polish_lbfgs(oracle, d_to_logit(
            d_raw.unsqueeze(0), jc), jc, u_row, beta=3.0, w_isl=a.w_isl)
        p_p, c_p, J_p = judge(oracle, jc, d_pol[0], u_row)
        t_p1 = time.perf_counter() - t0
        # 연마 2: tandem 출력 주변 지터 12점 (champion 의 top-K 다중시작과 동일 폭)
        t0 = time.perf_counter()
        g = th.Generator().manual_seed(a.seed + i)
        s_base = d_to_logit(d_raw.unsqueeze(0), jc).repeat(12, 1)
        s_base[1:] += 0.25 * th.randn(11, jc.line_N - 1, generator=g,
                                      dtype=th.float64)
        d_multi, t_pm, n_pm = polish_lbfgs(oracle, s_base, jc, u_row, beta=3.0,
                                           w_isl=a.w_isl)
        judged = [judge(oracle, jc, d_multi[m], u_row)
                  for m in range(d_multi.shape[0])]
        bm = int(np.argmin([x[2] for x in judged]))     # J 기준 선택 (IMP5 교훈)
        p_m, c_m, J_m = judged[bm]
        rows.append(dict(
            spec_deg=deg,
            raw=dict(psll_db=round(p_r, 3), cpl=round(c_r, 3),
                     J=round(J_r, 3), min_gap=round(float(d_raw.min()), 3),
                     t_ms=round(t_infer_ms, 3)),
            polished=dict(psll_db=round(p_p, 3), cpl=round(c_p, 3),
                          J=round(J_p, 3),
                          min_gap=round(float(d_pol[0].min()), 3),
                          t_s=round(t_p1, 2), n_formula_eval=n_pol),
            polished_multi12=dict(psll_db=round(p_m, 3), cpl=round(c_m, 3),
                                  J=round(J_m, 3),
                                  min_gap=round(float(d_multi[bm].min()), 3),
                                  t_s=round(time.perf_counter() - t0, 2),
                                  n_formula_eval=n_pm),
            d_raw=d_raw.tolist(), d_polished=d_pol[0].tolist(),
            d_multi=d_multi[bm].tolist()))

    # 사양 반응성 (mode collapse 검사): 사양 간 간격 벡터 L2 거리
    spread = float(th.cdist(d_all, d_all).max())

    out = dict(
        tag=a.tag, ckpt=a.ckpt,
        recipe=dict(steps=a.steps, batch=a.batch, lr=a.lr,
                    u_range=[a.u_lo, a.u_hi], angle_fracs=list(ANGLE_FRACS),
                    inverse_hidden=[256, 256, 256]),
        train=dict(elapsed_s=round(t_train, 1), device=str(dev), history=hist),
        infer_ms_per_spec=round(t_infer_ms, 3),
        spec_response_l2_um=round(spread, 3),
        rows=rows,
        reference=dict(
            note='champion(IMP4/IMP1)은 ±15° 전용 1건 설계 — 타 사양은 재실행 필요',
            imp4_cmaes_pm15=dict(psll_db=-13.480, J=-12.998, time_s=6.8),
            imp1_lbfgs_pm15=dict(psll_db=-13.402, J=-13.083, time_s=13.4),
            ga_pm15=dict(psll_db=-11.17, time_s=24.4)))

    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    jp = os.path.join(HERE, 'results', f'{a.tag}.json')
    with open(jp, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for r in rows:
        np.savetxt(os.path.join(
            HERE, 'results',
            f"final_spacing_{a.tag}_pm{int(r['spec_deg'])}.csv"),
            np.array(r['d_multi']), header='d_um', comments='')
    slim = json.loads(json.dumps(out))
    for r in slim['rows']:
        r.pop('d_raw')
        r.pop('d_polished')
        r.pop('d_multi')
    slim['train'].pop('history')
    print(json.dumps(slim, indent=2, ensure_ascii=False))
    print('→', jp)


if __name__ == '__main__':
    main()
