# 사전학습: 스트리밍 수식 타깃 → 서러게이트 회귀
#
# 게이트 지표 (val = 수식 홀드아웃 4096):
#   field_rmse  — 복소장 RMSE (타깃 |E|≤1 스케일)
#   int_r2      — 강도 R² (샘플별 R² 평균)
#   psll_err    — 조향 전용 팩(φ=φ*)에서 |PSLL_pred − PSLL_true| 평균 [dB]
import os
import json
import time
import math
import argparse
import torch as th

from config import SurrogateConfig
from physics_oracle import PhysicsOracle
from data_gen import SampleStream, load_val_pack
from model import build_model
from opt_core import psll_db, resolve_device      # 판정 지표는 opt_core 가 단일 진실원

HERE = os.path.dirname(os.path.abspath(__file__))


def make_steered_pack(cfg, oracle, n=512):
    # PSLL 지표 전용: φ=φ* 정확 조향 샘플 (u0 균등 그리드)
    stream = SampleStream(cfg, oracle, seed=cfg.seed + 20_000)
    d = stream.sample_d(n)
    u0 = th.linspace(-cfg.u0_range, cfg.u0_range, n, dtype=th.float64)
    phi = oracle.steering_phase(d, u0)
    with th.no_grad():
        E = oracle.field(d, phi)
    L_ap = oracle.positions(d)[:, -1]
    return d, phi, E, u0, L_ap


@th.no_grad()
def evaluate(model, oracle, val, steered, batch=256):
    model.eval()
    dev = next(model.parameters()).device   # 모델 장치 추종 (지표 계산은 CPU f64)
    dv, pv, Ev = val
    ds, ps, Es, u0s, Ls = steered
    u = oracle.u
    # 필드 RMSE + 강도 R²
    se, n_el = 0.0, 0
    ss_res, ss_tot = 0.0, 0.0
    for i in range(0, dv.shape[0], batch):
        d, p, E = dv[i:i + batch], pv[i:i + batch], Ev[i:i + batch]
        pred = model(d.float().to(dev), p.float().to(dev)).cpu().double()
        Ep = th.complex(pred[:, 0], pred[:, 1])
        se += (Ep - E).abs().pow(2).sum().item()
        n_el += E.numel()
        Ip, It = Ep.abs() ** 2, E.abs() ** 2
        mu = It.mean(dim=1, keepdim=True)
        ss_res += (It - Ip).pow(2).sum(dim=1)
        ss_tot += (It - mu).pow(2).sum(dim=1)
    field_rmse = math.sqrt(se / n_el)
    int_r2 = (1.0 - ss_res / ss_tot.clamp(min=1e-30)).mean().item()
    # PSLL 오차 (조향 팩)
    errs = []
    for i in range(0, ds.shape[0], batch):
        d, p, E = ds[i:i + batch], ps[i:i + batch], Es[i:i + batch]
        u0, L = u0s[i:i + batch], Ls[i:i + batch]
        pred = model(d.float().to(dev), p.float().to(dev)).cpu().double()
        Ip = pred[:, 0] ** 2 + pred[:, 1] ** 2
        It = E.abs() ** 2
        errs.append((psll_db(Ip, u, u0, L) - psll_db(It, u, u0, L)).abs())
    psll_err = th.cat(errs).mean().item()
    model.train()
    return dict(field_rmse=field_rmse, int_r2=int_r2, psll_err_db=psll_err)


def train(cfg: SurrogateConfig, tag=None):
    th.manual_seed(cfg.seed)
    oracle = PhysicsOracle()
    stream = SampleStream(cfg, oracle)
    val = load_val_pack(cfg)
    steered = make_steered_pack(cfg, oracle)
    model = build_model(cfg, oracle)
    dev = th.device(resolve_device(cfg.device))
    model = model.to(dev)
    n_par = sum(p.numel() for p in model.parameters())

    opt = th.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps,
                                                    eta_min=cfg.lr_min)
    tag = tag or cfg.arch
    os.makedirs(os.path.join(HERE, cfg.ckpt_dir), exist_ok=True)
    os.makedirs(os.path.join(HERE, cfg.results_dir), exist_ok=True)
    log_path = os.path.join(HERE, cfg.results_dir, f'train_{tag}.jsonl')
    ckpt_path = os.path.join(HERE, cfg.ckpt_dir, f'{tag}.pt')
    best = float('inf')
    t0 = time.time()
    print(f'[{tag}] params {n_par/1e6:.2f}M | steps {cfg.steps} | batch {cfg.batch} '
          f'| device {dev}', flush=True)

    with open(log_path, 'w') as lf:
        for step in range(1, cfg.steps + 1):
            d, phi, E = stream.batch(cfg.batch)
            tgt = th.stack([E.real, E.imag], dim=1).float().to(dev)
            pred = model(d.float().to(dev), phi.float().to(dev))
            loss = th.nn.functional.mse_loss(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % cfg.val_every == 0 or step == cfg.steps:
                m = evaluate(model, oracle, val, steered)
                m.update(step=step, loss=loss.item(),
                         elapsed_s=round(time.time() - t0, 1))
                lf.write(json.dumps(m) + '\n')
                lf.flush()
                print(f"[{tag}] {step:6d} loss {m['loss']:.3e} | field_rmse "
                      f"{m['field_rmse']:.3e} | int_R2 {m['int_r2']:.5f} | "
                      f"PSLL err {m['psll_err_db']:.3f} dB | {m['elapsed_s']}s",
                      flush=True)
                if m['field_rmse'] < best:
                    best = m['field_rmse']
                    th.save(dict(state=model.state_dict(), cfg=cfg.to_dict(),
                                 metrics=m), ckpt_path)
    print(f'[{tag}] 완료: best field_rmse {best:.3e} → {ckpt_path}', flush=True)
    return ckpt_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='mlp',
                    choices=['mlp', 'element', 'siren', 'ffmlp'])
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--device', default=None, choices=['auto', 'cpu', 'mps'])
    ap.add_argument('--seed', type=int, default=None,
                    help='앙상블용 시드 오버라이드 (기본 42)')
    ap.add_argument('--lr', type=float, default=None,
                    help='학습률 오버라이드 (SIREN은 3e-4 권장)')
    ap.add_argument('--siren-w0', type=float, default=None,
                    help='SIREN w0 오버라이드 (기본 30, 완화 프로브용)')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    cfg = SurrogateConfig(arch=a.arch)
    if a.steps:
        cfg.steps = a.steps
    if a.device:
        cfg.device = a.device
    if a.seed is not None:
        cfg.seed = a.seed
    if a.lr is not None:
        cfg.lr = a.lr
    if a.siren_w0 is not None:
        cfg.siren_w0 = a.siren_w0
    train(cfg, tag=a.tag)
