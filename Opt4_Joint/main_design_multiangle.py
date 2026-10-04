# Stage A' — 다각도(multi-angle) 간격 설계: 조향 전 범위에서 양측 로브 억제
#
# 단일각(θ=0) 설계의 한계: 조향 시 가시창이 u = v + u0 로 이동하며 설계 밖 사이드로브가
# 노출되고 EF 비율(EF(u_side)/EF(u0))이 불리해져 PSLL 열화 (−15.5 → −11.1 dB @ ±30°).
# 해법: 조향각 집합 {0, ±10, ±20, ±30}° 전체의 soft-PSLL을 worst-case soft-max로 집계해
# 간격만 최적화. 위상은 각도별 해석 조향해(런타임 재설정 대상)로 두므로 손실은 순수 간격 담당
# — '간격 = 전 조향 범위 로브 억제 / 위상 = 각도별 주엽 조향' 역할 분담의 완성형.
#
# 간격은 완전 비등간격: 각 d_n 독립 로짓 + 초기 섭동 확대(std 0.3) — 모든 간격 상호 상이.
#
# 실행: venv/bin/python main_design_multiangle.py
import json
import math
import os
import time

import pandas as pd
import torch as th
import torch.nn.functional as F

from config import JointConfig
from losses import beta_schedule, multi_angle_worst_psll
from main_design import RESULTS, steering_sweep
from metrics import hard_psll_db, summarize
from model import OPAModel


def run_multiangle_single(cfg, restart_seed, track_logs=False):
    th.manual_seed(restart_seed)
    model = OPAModel(cfg)
    n_gap = cfg.line_N - 1
    s = th.full((n_gap,), model.s_init_uniform(), dtype=cfg.dtype) \
        + cfg.s_init_std * th.randn(n_gap, dtype=cfg.dtype)
    s.requires_grad_(True)
    # 각도별 학습형 위상 잔차 δφ_a (해석 조향해에 더해짐 — 사이드로브 미세 정형 담당)
    dphi = th.zeros(len(cfg.design_angles_deg), cfg.line_N, dtype=cfg.dtype,
                    requires_grad=True)
    opt_s = th.optim.Adam([s], lr=cfg.lr_spacing)
    opt_p = th.optim.Adam([dphi], lr=cfg.lr_phase)

    logs = {'loss': [], 'worst_psll': [], 'gaps': []} if track_logs else None

    def worst_hard(u_grid):
        with th.no_grad():
            x = model.positions(s)
            return max(
                hard_psll_db(model, x,
                             model.steering_phase(x, math.sin(math.radians(a))) + dphi[i],
                             u=u_grid, u0=math.sin(math.radians(a)))
                for i, a in enumerate(cfg.design_angles_deg))

    for t in range(cfg.epochs_joint):
        beta = beta_schedule(t, cfg.epochs_joint, cfg.beta_start, cfg.beta_end)
        opt_s.zero_grad()
        opt_p.zero_grad()
        loss = multi_angle_worst_psll(model, s, beta, cfg.design_angles_deg,
                                      cfg.angle_agg_gamma, dphi=dphi)
        loss.backward()
        opt_s.step()
        opt_p.step()
        if logs is not None:
            logs['loss'].append(loss.item())
            logs['gaps'].append(model.gaps(s.detach()).tolist())
            logs['worst_psll'].append(worst_hard(model.u_train))

    return {'s': s.detach(), 'dphi': dphi.detach(), 'model': model,
            'worst_psll': worst_hard(model.u_val), 'logs': logs, 'seed': restart_seed}


def snap(cfg, model, s_star):
    # 공정 그리드 스냅 + 간격 중복 강제 제거 ('모든 간격 상호 상이' 요구)
    with th.no_grad():
        d = model.gaps(s_star)
        grid = cfg.fab_snap_nm * 1e-3
        d_snap = th.round(d / grid) * grid
        vals = d_snap.tolist()
        used = set()
        for i in sorted(range(len(vals)), key=lambda j: vals[j]):
            v, step = vals[i], 0
            while round(v, 9) in used:
                step += 1
                cand_up = vals[i] + step * grid
                cand_dn = vals[i] - step * grid
                v = cand_up if cand_up <= cfg.d_max else cand_dn
                if v < cfg.d_min:
                    v = cand_up
            vals[i] = round(v, 9)
            used.add(vals[i])
        d_snap = th.tensor(vals, dtype=cfg.dtype, device=cfg.device)
        return d_snap, F.pad(d_snap.cumsum(-1), (1, 0))


@th.no_grad()
def gap_distinctness(d):
    # '모든 간격 상호 상이' 검증: 최소 쌍별 차이
    return th.pdist(d.reshape(-1, 1)).min().item()


def main():
    os.makedirs(RESULTS, exist_ok=True)
    # 비등간격 초기 다양성 확대 + worst-case 집계 강화 + 학습 연장
    cfg = JointConfig(s_init_std=0.5, angle_agg_gamma=2.0,
                      epochs_joint=1500, restarts=24)
    t0 = time.time()
    report = {'config': cfg.to_dict()}

    runs = []
    for r in range(cfg.restarts):
        res = run_multiangle_single(cfg, cfg.seed + r, track_logs=(r == 0))
        runs.append(res)
        print(f"[MA] restart {r:02d}  worst-angle PSLL = {res['worst_psll']:+.2f} dB")
    best = min(runs, key=lambda z: z['worst_psll'])
    if best['logs'] is None:
        best = run_multiangle_single(cfg, best['seed'], track_logs=True)
    model = best['model']
    report['MA_all_restarts_worst_psll'] = sorted(round(z['worst_psll'], 3) for z in runs)

    # 스냅 + 지표
    d_snap, x_snap = snap(cfg, model, best['s'])
    phi0 = model.steering_phase(x_snap, 0.0)
    report['MA_best'] = summarize(model, x_snap, phi0, s=best['s'])
    report['MA_best']['seed'] = best['seed']
    report['MA_best']['min_pairwise_gap_diff_um'] = gap_distinctness(d_snap)
    report['MA_best']['n_unique_gaps'] = len(set(round(v, 6) for v in d_snap.tolist()))

    # E1' — 조향 스윕 (위상 polish 포함; 설계각 사이 구멍 검출 위해 5° 간격 조밀 평가)
    dense = tuple(a for a in range(-30, 31, 5))
    report['E1_steering_sweep'] = steering_sweep(cfg, model, x_snap, angles_deg=dense)
    worst_row = max(report['E1_steering_sweep'], key=lambda r: r['psll_db'])
    print(f"\n[MA] worst steering: θ={worst_row['theta_deg']:+.0f}°  "
          f"PSLL={worst_row['psll_db']:+.2f} dB  I0(exEF)={worst_row['main_lobe_eff_exEF']:.3f}")

    # 산출물
    pd.DataFrame({'gap_index': range(1, cfg.line_N), 'd_um': d_snap.tolist()}).to_csv(
        os.path.join(RESULTS, 'final_spacing_multiangle.csv'), index=False)
    pd.DataFrame({'element': range(cfg.line_N), 'x_um': x_snap.tolist()}).to_csv(
        os.path.join(RESULTS, 'final_layout_multiangle.csv'), index=False)
    pd.DataFrame(best['logs']['gaps']).to_csv(
        os.path.join(RESULTS, 'spacing_log_multiangle.csv'), index_label='epoch')
    pd.DataFrame({'loss': best['logs']['loss'],
                  'worst_psll_db': best['logs']['worst_psll']}).to_csv(
        os.path.join(RESULTS, 'design_log_multiangle.csv'), index_label='epoch')

    report['runtime_sec'] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, 'design_report_multiangle.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    import plotting
    plotting.make_all_multiangle(cfg, model, x_snap, report, RESULTS)
    print(f"총 {report['runtime_sec']} s. 산출물 → {RESULTS}/")
    return report


if __name__ == '__main__':
    main()
