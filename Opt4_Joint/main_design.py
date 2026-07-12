# Stage A — 오프라인 설계: 간격(s) + 위상(φ) 공동 최적화
#
# 역할 분담 (물리로 보장):
#   위상  = 주엽 형성/조향/캘리브레이션 (임의 x에서 φ=k·x·u0로 L_main 전역 최소 도달)
#   간격  = 비주기화로 grating lobe/PSLL 억제 (위상 정렬 평형에서 사이드로브는 순수 x의 함수)
# 두-시간척도(lr_phase > lr_spacing) + 위상 워밍업으로 위상이 항상 φ*(x)를 추적 → bilevel 근사
#
# 3-phase 스케줄: warmup(φ만, w_sll=0) → joint(동시, w_sll 램프 + β 어닐링) → polish(s 동결)
# 멀티스타트 16회 → 검증 격자 하드 PSLL 최소 run 채택 → 5nm 스냅 → 재검증
#
# 실행: venv/bin/python main_design.py            (전체: A/B/C + E1 조향 + E3 MC + E4 스냅)
import json
import math
import os
import time

import pandas as pd
import torch as th

from config import JointConfig
from losses import beta_schedule, total_loss, w_sll_schedule
from metrics import hard_psll_db, summarize
from model import OPAModel

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def run_single(cfg, restart_seed, mode='joint', track_logs=False):
    """mode: 'joint' | 'phase_only'(baseline A) | 'spacing_only'(ablation C)"""
    th.manual_seed(restart_seed)
    model = OPAModel(cfg)
    n_gap = cfg.line_N - 1
    s0 = model.s_init_uniform()

    learn_spacing = mode in ('joint', 'spacing_only')
    if learn_spacing:
        s = th.full((n_gap,), s0, dtype=cfg.dtype) \
            + cfg.s_init_std * th.randn(n_gap, dtype=cfg.dtype)
        s.requires_grad_(True)
    else:
        s = th.full((n_gap,), s0, dtype=cfg.dtype)  # 등간격 d_init 동결

    phi = model.steering_phase(model.positions(s.detach())).clone()
    phi.requires_grad_(True)

    opt_phase = th.optim.Adam([phi], lr=cfg.lr_phase)
    opt_spacing = th.optim.Adam([s], lr=cfg.lr_spacing) if learn_spacing else None

    logs = {'loss': [], 'psll': [], 'i0': [], 'gaps': []} if track_logs else None

    def step(update_phase, update_spacing, beta, w_sll):
        opt_phase.zero_grad()
        if opt_spacing is not None:
            opt_spacing.zero_grad()
        x = model.positions(s)
        phi_eff = model.steering_phase(x) if mode == 'spacing_only' else phi
        loss, i0 = total_loss(model, x, phi_eff, beta, w_sll)
        loss.backward()
        if update_phase and mode != 'spacing_only':
            opt_phase.step()
        if update_spacing and opt_spacing is not None:
            opt_spacing.step()
        return loss.item(), i0.item()

    def log_epoch(loss_v, i0_v):
        if logs is None:
            return
        with th.no_grad():
            x = model.positions(s)
            phi_eff = model.steering_phase(x) if mode == 'spacing_only' else phi
            logs['psll'].append(hard_psll_db(model, x, phi_eff, u=model.u_train))
            logs['gaps'].append(model.gaps(s).tolist())
        logs['loss'].append(loss_v)
        logs['i0'].append(i0_v)

    # Phase 1 — 위상 워밍업 (spacing_only는 위상이 해석해라 생략)
    if mode != 'spacing_only':
        for _ in range(cfg.epochs_warmup):
            lv, iv = step(True, False, cfg.beta_start, 0.0)
            log_epoch(lv, iv)

    # Phase 2 — joint (β 어닐링 + w_sll 램프; phase_only도 동일 손실로 위상만 갱신)
    for t in range(cfg.epochs_joint):
        beta = beta_schedule(t, cfg.epochs_joint, cfg.beta_start, cfg.beta_end)
        w = w_sll_schedule(t, cfg.w_sll_ramp_epochs)
        lv, iv = step(True, True, beta, w)
        log_epoch(lv, iv)

    # Phase 3 — polish (간격 동결, 위상 재수렴, 조기 종료)
    opt_phase.param_groups[0]['lr'] = cfg.lr_phase_polish
    window = []
    for _ in range(cfg.epochs_polish):
        if mode == 'spacing_only':
            break
        lv, iv = step(True, False, cfg.beta_end, 1.0)
        log_epoch(lv, iv)
        window.append(lv)
        if len(window) > 20:
            window.pop(0)
            if abs(window[0] - window[-1]) / max(abs(window[0]), 1e-30) < 1e-6:
                break

    with th.no_grad():
        x = model.positions(s)
        phi_final = model.steering_phase(x) if mode == 'spacing_only' else phi.detach()
        val_psll = hard_psll_db(model, x, phi_final, u=model.u_val)
    return {'s': s.detach(), 'phi': phi_final.clone(), 'model': model,
            'val_psll': val_psll, 'logs': logs, 'seed': restart_seed}


def snap_and_recalibrate(cfg, model, s_star):
    # E4 — 5nm 공정 그리드 스냅 후 위상만 재캘리브레이션 (Stage B 모사)
    with th.no_grad():
        d = model.gaps(s_star)
        grid = cfg.fab_snap_nm * 1e-3  # nm → µm
        d_snap = th.round(d / grid) * grid
        zero = th.zeros(1, dtype=cfg.dtype, device=cfg.device)
        x_snap = th.cat([zero, th.cumsum(d_snap, dim=0)])
    phi = model.steering_phase(x_snap).clone().requires_grad_(True)
    opt = th.optim.Adam([phi], lr=cfg.lr_phase_polish)
    for _ in range(cfg.epochs_polish):
        opt.zero_grad()
        loss, _ = total_loss(model, x_snap, phi, cfg.beta_end, 1.0)
        loss.backward()
        opt.step()
    return d_snap, x_snap, phi.detach()


def steering_sweep(cfg, model, x_frozen, angles_deg=(0, 10, -10, 20, -20, 30, -30)):
    # E1 — 간격 동결, 각도별 위상만 재조향 → '간격=1회 설계 / 위상=런타임 조향' 검증
    rows = []
    for a in angles_deg:
        u0 = math.sin(math.radians(a))
        phi = model.steering_phase(x_frozen, u0).clone().requires_grad_(True)
        opt = th.optim.Adam([phi], lr=cfg.lr_phase_polish)
        for _ in range(cfg.epochs_polish):
            opt.zero_grad()
            loss, _ = total_loss(model, x_frozen, phi, cfg.beta_end, 1.0, u0=u0)
            loss.backward()
            opt.step()
        m = summarize(model, x_frozen, phi.detach(), u0=u0)
        rows.append({'theta_deg': a, **m})
    return rows


@th.no_grad()
def robustness_mc(cfg, model, s_star, phi_star, n_draw=1000,
                  sigma_gap_nm=20.0, sigma_phi_rad=0.05):
    # E3 — fab/DAC 공차 몬테카를로: PSLL 분포 95백분위
    th.manual_seed(cfg.seed + 9999)
    d_star = model.gaps(s_star)
    psll = []
    for _ in range(n_draw):
        d = d_star + (sigma_gap_nm * 1e-3) * th.randn_like(d_star)
        zero = th.zeros(1, dtype=cfg.dtype, device=cfg.device)
        x = th.cat([zero, th.cumsum(d, dim=0)])
        phi = phi_star + sigma_phi_rad * th.randn_like(phi_star)
        psll.append(hard_psll_db(model, x, phi, u=model.u_train))
    t = th.tensor(psll)
    return {'p50': t.median().item(),
            'p95': t.kthvalue(int(0.95 * n_draw)).values.item(),
            'worst': t.max().item()}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    cfg = JointConfig()
    t0 = time.time()
    report = {'config': cfg.to_dict()}

    # --- A: baseline (등간격 3µm + 위상만, 새 손실/수정 Adam) ---
    base = run_single(cfg, cfg.seed, mode='phase_only', track_logs=True)
    report['A_baseline_uniform'] = summarize(base['model'], base['model'].positions(base['s']),
                                             base['phi'], s=base['s'])
    print(f"[A] 등간격 baseline  PSLL = {report['A_baseline_uniform']['psll_db']:+.2f} dB")

    # --- B: joint 멀티스타트 ---
    runs = []
    for r in range(cfg.restarts):
        res = run_single(cfg, cfg.seed + r, mode='joint', track_logs=(r == 0))
        runs.append(res)
        print(f"[B] restart {r:02d}  val PSLL = {res['val_psll']:+.2f} dB")
    best = min(runs, key=lambda z: z['val_psll'])
    # 로그는 restart 0에서만 수집 → best의 로그가 없으면 재실행으로 확보
    if best['logs'] is None:
        best = run_single(cfg, best['seed'], mode='joint', track_logs=True)
    model = best['model']
    x_best = model.positions(best['s'])
    report['B_joint_best'] = summarize(model, x_best, best['phi'], s=best['s'])
    report['B_joint_best']['seed'] = best['seed']
    report['B_all_restarts_psll'] = sorted(round(z['val_psll'], 3) for z in runs)
    print(f"[B] best (seed {best['seed']})  PSLL = {report['B_joint_best']['psll_db']:+.2f} dB")

    # --- C: spacing_only ablation (위상 = 해석 조향해 고정) ---
    abl = run_single(cfg, cfg.seed, mode='spacing_only')
    report['C_spacing_only'] = summarize(abl['model'], abl['model'].positions(abl['s']),
                                         abl['phi'], s=abl['s'])
    print(f"[C] spacing-only     PSLL = {report['C_spacing_only']['psll_db']:+.2f} dB")

    # --- E4: fab 스냅 + 위상 재캘리브레이션 ---
    d_snap, x_snap, phi_snap = snap_and_recalibrate(cfg, model, best['s'])
    report['E4_fab_snap'] = summarize(model, x_snap, phi_snap)
    degrade = report['E4_fab_snap']['psll_db'] - report['B_joint_best']['psll_db']
    report['E4_fab_snap']['psll_degrade_db'] = degrade
    if degrade > 0.5:
        print(f"[E4] 경고: 스냅 열화 {degrade:.2f} dB > 0.5 dB")

    # --- E1: 조향 스윕 (스냅된 최종 간격 사용) ---
    report['E1_steering_sweep'] = steering_sweep(cfg, model, x_snap)

    # --- E3: 강건성 몬테카를로 ---
    report['E3_robustness_mc'] = robustness_mc(cfg, model, best['s'], best['phi'])

    # --- 산출물 ---
    pd.DataFrame({'gap_index': range(1, cfg.line_N),
                  'd_um': d_snap.tolist()}).to_csv(
        os.path.join(RESULTS, 'final_spacing.csv'), index=False)
    pd.DataFrame({'element': range(cfg.line_N),
                  'x_um': x_snap.tolist(),
                  'phase_rad_mod2pi': th.remainder(phi_snap, 2 * math.pi).tolist()}).to_csv(
        os.path.join(RESULTS, 'final_layout.csv'), index=False)
    pd.DataFrame(best['logs']['gaps']).to_csv(
        os.path.join(RESULTS, 'spacing_log.csv'), index_label='epoch')
    pd.DataFrame({'loss': best['logs']['loss'], 'psll_db': best['logs']['psll'],
                  'main_lobe_I': best['logs']['i0']}).to_csv(
        os.path.join(RESULTS, 'design_log.csv'), index_label='epoch')
    pd.DataFrame({'loss': base['logs']['loss'], 'psll_db': base['logs']['psll'],
                  'main_lobe_I': base['logs']['i0']}).to_csv(
        os.path.join(RESULTS, 'baseline_log.csv'), index_label='epoch')

    report['runtime_sec'] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, 'design_report.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # --- 플롯 ---
    import plotting
    plotting.make_all(cfg, model, base, best, x_snap, phi_snap, RESULTS)

    print(f"\n총 {report['runtime_sec']} s. 산출물 → {RESULTS}/")
    return report


if __name__ == '__main__':
    main()
