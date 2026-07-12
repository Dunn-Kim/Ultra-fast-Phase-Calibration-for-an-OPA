# 다각도 설계 플롯: (1) 조향각별 PSLL 비교 (단일각 vs 다각도 설계), (2) 조향 패턴 오버레이
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch as th


def _pattern_db(model, x, phi, u0):
    with th.no_grad():
        intens = model.intensity(x, phi, model.u_val)
        i0 = model.intensity_at_u0(x, phi, u0)
        db = 10.0 * np.log10(np.maximum(intens.numpy() / max(i0.item(), 1e-30), 1e-15))
    return np.degrees(np.arcsin(model.u_val.numpy())), db


def make_all(cfg, model, x_snap, report, outdir):
    # (1) 조향각별 PSLL — 단일각 설계(design_report.json)와 비교
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = sorted(report['E1_steering_sweep'], key=lambda r: r['theta_deg'])
    ax.plot([r['theta_deg'] for r in rows], [r['psll_db'] for r in rows],
            'o-', color='tab:blue', label='multi-angle design')
    single_path = os.path.join(outdir, 'design_report.json')
    if os.path.exists(single_path):
        old = sorted(json.load(open(single_path))['E1_steering_sweep'],
                     key=lambda r: r['theta_deg'])
        ax.plot([r['theta_deg'] for r in old], [r['psll_db'] for r in old],
                's--', color='tab:gray', label='single-angle design (θ=0)')
    ax.axhline(-1.65, color='tab:red', ls=':', lw=0.8)
    ax.annotate('uniform 3µm limit (−1.65 dB)', xy=(-28, -1.4), color='tab:red', fontsize=8)
    ax.set(xlabel='steering angle [deg]', ylabel='PSLL [dB]',
           title=f'N={cfg.line_N} steering robustness (phase-only re-steer, spacing frozen)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'psll_vs_angle.png'), dpi=150)
    plt.close(fig)

    # (2) 조향 패턴 — θ = 0°, +30° (다각도 설계 간격, 해석 조향위상)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, a in zip(axes, (0.0, 30.0)):
        u0 = math.sin(math.radians(a))
        phi = model.steering_phase(x_snap, u0)
        theta, db = _pattern_db(model, x_snap, phi, u0)
        ax.plot(theta, db, lw=0.8, color='tab:blue')
        ax.axvline(a, color='tab:green', ls='--', lw=0.8)
        gl = cfg.wavelength / cfg.d_init
        for m in (1, -1):
            ug = u0 + m * gl
            if abs(ug) <= 1:
                ax.axvline(math.degrees(math.asin(ug)), color='tab:red', ls=':', lw=0.8)
        ax.set(ylabel='rel. intensity [dB]', ylim=(-40, 2),
               title=f'steered to {a:+.0f}° (red dotted = uniform-array grating lobe positions)')
        ax.grid(alpha=0.3)
    axes[1].set(xlabel='theta [deg]', xlim=(-90, 90))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'pattern_steered.png'), dpi=150)
    plt.close(fig)
