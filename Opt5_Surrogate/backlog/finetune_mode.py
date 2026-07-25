# MODE 비교 검증 + fine-tune (테스트셋 = MODE 규약)
#
# 데이터: Opt3_Back_Forward/Resultants/{phase,pattern}_tracked_32.csv
#   - phase: 인가위상[deg] 31열 (소자0 = 기준 0° 미기록) → φ = [0, row], 행0 = init
#   - pattern: varFDTD E2 1801점 (θ-균등 −90..90°), 임의 스케일 → 프레임별 max 정규화
#   - 쌍 매핑: pattern 행 j ↔ phase 행 j+1 (검증된 규약)
#   - 배열: d = 3µm 등간격 고정 (MODE 로그의 유일한 간격 — 한계 명시)
#
# 게인 처리: 카메라/시뮬 임의 스케일 → 프레임별 닫힌형 LS 게인 c = (m·o)/(m·m)
#   (calibrate_fast 와 동일 규약, 미분가능 → FT 손실 내부에서 적합)
#
# FT 변형 2종:
#   pure   — MODE 80프레임만 (조기종료: holdout 20)
#   replay — MODE 손실 + λ·수식 스트리밍 필드 MSE (d-일반화 망각 방지)
# 리포트: pre/post 강도 R² (train/holdout), 수식 val 드리프트
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import os
import json
import copy
import time
import argparse
import numpy as np
import torch as th

from config import SurrogateConfig
from physics_oracle import PhysicsOracle
from data_gen import SampleStream, load_val_pack
from model import build_model
from train import evaluate, make_steered_pack

HERE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))   # 산출물은 상위 results/ 로 통일
RES = os.path.join(HERE, '..', 'Opt3_Back_Forward', 'Resultants')


def load_mode_pairs(n_elem=32):
    ph = np.loadtxt(os.path.join(RES, f'phase_tracked_{n_elem}.csv'),
                    delimiter=',', skiprows=1)[:, 1:]          # (101, N-1) deg
    pt = np.loadtxt(os.path.join(RES, f'pattern_tracked_{n_elem}.csv'),
                    delimiter=',', skiprows=1)[:, 1:]          # (100, 1801)
    phi = np.deg2rad(ph[1:1 + pt.shape[0]])                    # 행 j+1 ↔ pattern j
    phi = np.concatenate([np.zeros((phi.shape[0], 1)), phi], axis=1)  # 소자0 = 0
    obs = pt / pt.max(axis=1, keepdims=True)                   # 프레임별 max=1
    return (th.tensor(phi, dtype=th.float64),
            th.tensor(obs, dtype=th.float64))


def ls_gain(m, o):
    # 프레임별 닫힌형 게인 (B,) — m, o: (B, G)
    return (m * o).sum(dim=1) / (m * m).sum(dim=1).clamp(min=1e-30)


def frame_r2(m, o):
    c = ls_gain(m, o)
    res = (c.unsqueeze(1) * m - o).pow(2).sum(dim=1)
    tot = (o - o.mean(dim=1, keepdim=True)).pow(2).sum(dim=1)
    return 1.0 - res / tot.clamp(min=1e-30)


def predict_intensity(model, d, phi):
    pred = model(d, phi)
    return (pred[:, 0] ** 2 + pred[:, 1] ** 2).double()


@th.no_grad()
def eval_mode(model, d, phi, obs, sign=+1):
    return frame_r2(predict_intensity(model, d, sign * phi), obs)


def pick_phase_sign(model, d, phi, obs):
    # 인가위상 부호 규약(deg 로그 ↔ 모델 −φ 지수)은 데이터로 판정
    r_pos = eval_mode(model, d, phi, obs, +1).mean().item()
    r_neg = eval_mode(model, d, phi, obs, -1).mean().item()
    return (+1, r_pos) if r_pos >= r_neg else (-1, r_neg)


def finetune(cfg: SurrogateConfig, ckpt, variant='replay', steps=2000,
             lr=1e-4, replay_w=1.0, holdout=20, seed=0, tag=None):
    oracle = PhysicsOracle()
    model = build_model(cfg, oracle)
    state = th.load(ckpt, map_location='cpu')
    model.load_state_dict(state['state'])

    phi, obs = load_mode_pairs()
    B = phi.shape[0]
    d = th.full((B, oracle.jcfg.line_N - 1), oracle.jcfg.d_init, dtype=th.float64)

    sign, r_pre = pick_phase_sign(model, d, phi, obs)
    phi = sign * phi
    g = th.Generator().manual_seed(seed)
    perm = th.randperm(B, generator=g)
    tr, ho = perm[holdout:], perm[:holdout]
    r2_pre = frame_r2(predict_intensity(model, d, phi), obs)
    pre = dict(sign=sign,
               r2_all=r2_pre.mean().item(),
               r2_train=r2_pre[tr].mean().item(),
               r2_holdout=r2_pre[ho].mean().item())

    val = load_val_pack(cfg)
    steered = make_steered_pack(cfg, oracle)
    fm_pre = evaluate(model, oracle, val, steered)

    stream = SampleStream(cfg, oracle, seed=cfg.seed + 30_000)
    opt = th.optim.Adam(model.parameters(), lr=lr)
    best_ho, best_state, best_step = -1e9, None, 0
    t0 = time.time()
    for step in range(1, steps + 1):
        Im = predict_intensity(model, d[tr], phi[tr])
        c = ls_gain(Im.detach(), obs[tr])          # 게인은 상수 취급 (스케일만 흡수)
        loss = ((c.unsqueeze(1) * Im - obs[tr]).pow(2).sum(dim=1)
                / obs[tr].pow(2).sum(dim=1)).mean()
        if variant == 'replay':
            dr, pr, Er = stream.batch(128)
            tgt = th.stack([Er.real, Er.imag], dim=1).float()
            loss = loss + replay_w * th.nn.functional.mse_loss(model(dr, pr), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps:
            r_ho = eval_mode(model, d[ho], phi[ho], obs[ho]).mean().item()
            if r_ho > best_ho:
                best_ho, best_step = r_ho, step
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)          # 조기종료: holdout 최고점 복원

    r2_post = frame_r2(predict_intensity(model, d, phi), obs)
    fm_post = evaluate(model, oracle, val, steered)
    tag = tag or f"{cfg.arch}_{variant}"
    out = dict(tag=tag, ckpt=ckpt, variant=variant, steps=steps, lr=lr,
               replay_w=replay_w if variant == 'replay' else 0.0,
               best_step=best_step, elapsed_s=round(time.time() - t0, 1),
               pre=pre,
               post=dict(r2_all=r2_post.mean().item(),
                         r2_train=r2_post[tr].mean().item(),
                         r2_holdout=r2_post[ho].mean().item()),
               formula_pre=fm_pre, formula_post=fm_post)
    os.makedirs(os.path.join(HERE, cfg.ckpt_dir), exist_ok=True)
    ft_path = os.path.join(HERE, cfg.ckpt_dir, f'{tag}_ft.pt')
    th.save(dict(state=model.state_dict(), cfg=cfg.to_dict(), report=out), ft_path)
    rp = os.path.join(HERE, cfg.results_dir, f'finetune_{tag}.json')
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, 'w') as f:
        json.dump(out, f, indent=2)
    return out, ft_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='element', choices=['mlp', 'element'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--variant', default='replay', choices=['pure', 'replay'])
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--replay-w', type=float, default=1.0)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    cfg = SurrogateConfig(arch=a.arch)
    out, path = finetune(cfg, a.ckpt, variant=a.variant, steps=a.steps,
                         lr=a.lr, replay_w=a.replay_w, tag=a.tag)
    print(json.dumps(out, indent=2))
    print('FT ckpt:', path)
