# 비교 플롯 3종: (1) 등간격 vs 최적 패턴 dB 오버레이, (2) 간격 진화, (3) 수렴 곡선
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch as th


def _pattern_db(model, x, phi):
    with th.no_grad():
        intens = model.intensity(x, phi, model.u_val)
        i0 = model.intensity_at_u0(x, phi)
        db = 10.0 * np.log10(np.maximum(intens.numpy() / max(i0.item(), 1e-30), 1e-15))
    theta = np.degrees(np.arcsin(model.u_val.numpy()))
    return theta, db


def make_all(cfg, model, base, best, x_snap, phi_snap, outdir):
    # (1) 패턴 오버레이 — ±31.1° = 등간격 3µm의 grating lobe 위치 마킹
    x_base = base['model'].positions(base['s'])
    th_b, db_b = _pattern_db(base['model'], x_base, base['phi'])
    th_o, db_o = _pattern_db(model, x_snap, phi_snap)
    gl = math.degrees(math.asin(cfg.wavelength / cfg.d_init))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(th_b, db_b, lw=0.8, color='tab:gray', label=f'uniform {cfg.d_init}µm + phase-only')
    ax.plot(th_o, db_o, lw=0.8, color='tab:blue', label='Opt4 joint (spacing+phase, snapped)')
    for g in (gl, -gl):
        ax.axvline(g, color='tab:red', ls='--', lw=0.8, alpha=0.7)
    ax.annotate(f'grating lobe ±{gl:.1f}°', xy=(gl, -2), color='tab:red', fontsize=8)
    ax.set(xlabel='theta [deg]', ylabel='relative intensity [dB]',
           ylim=(-40, 2), xlim=(-90, 90),
           title=f'N={cfg.line_N} far-field: uniform vs joint-optimized (val grid {cfg.n_grid_val})')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'pattern_overlay.png'), dpi=150)
    plt.close(fig)

    # (2) 간격 진화 히트맵 + 최종 간격 분포
    gaps = np.array(best['logs']['gaps'])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4),
                                 gridspec_kw={'width_ratios': [2, 1]})
    im = a1.imshow(gaps.T, aspect='auto', cmap='viridis', origin='lower',
                   extent=[0, gaps.shape[0], 0.5, gaps.shape[1] + 0.5])
    a1.set(xlabel='epoch', ylabel='gap index', title='gap evolution [µm]')
    fig.colorbar(im, ax=a1)
    a2.stem(range(1, gaps.shape[1] + 1), gaps[-1])
    a2.axhline(cfg.d_init, color='tab:gray', ls='--', lw=0.8)
    a2.axhline(cfg.d_min, color='tab:red', ls=':', lw=0.8)
    a2.axhline(cfg.d_max, color='tab:red', ls=':', lw=0.8)
    a2.set(xlabel='gap index', ylabel='d [µm]', title='final gaps')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'spacing_evolution.png'), dpi=150)
    plt.close(fig)

    # (3) 수렴 곡선 — loss / 하드 PSLL / 주엽 강도
    lg = best['logs']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(lg['loss'], lw=0.9)
    axes[0].set(xlabel='epoch', ylabel='total loss', title='loss')
    axes[1].plot(lg['psll'], lw=0.9, label='joint')
    axes[1].plot(base['logs']['psll'], lw=0.9, color='tab:gray', label='baseline')
    axes[1].set(xlabel='epoch', ylabel='hard PSLL [dB]', title='PSLL')
    axes[1].legend(fontsize=8)
    axes[2].plot(lg['i0'], lw=0.9)
    axes[2].set(xlabel='epoch', ylabel='I(u0)', title='main lobe intensity')
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'convergence.png'), dpi=150)
    plt.close(fig)
