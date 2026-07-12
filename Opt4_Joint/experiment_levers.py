# 진짜 지렛대 식별: 진폭이 0.8 dB밖에 못 준 이유 = 성긴 배열(d_min=2µm ≫ λ/2).
# PSLL를 실제로 움직이는 변수 = 소자 수 N + 충전율(d_min). 아포다이제이션 아님.
#
# 스윕: (1) N ∈ {32,64,128} 비등간격+진폭, (2) d_min ∈ {2.0,1.0,0.775=λ/2} at N=32.
# 실행: venv/bin/python experiment_levers.py
import json
import math
import os

import torch as th

import experiment_redesign as E

RESULTS = E.RESULTS
DT = E.DT


def run(N, dmin, learn_amp=True, epochs=1200, best=6):
    E.N, E.DMIN = N, dmin
    return E.best_of(best, learn_spacing=True, learn_amp=learn_amp, w_eff=0.0)


def main():
    floor = lambda n: 10 * math.log10(1.0 / n)
    print('=== 지렛대 1: 소자 수 N (d_min=2µm, 비등간격+진폭) ===')
    lv_N = []
    for N in (32, 64, 128):
        r = run(N, 2.0, epochs=1000, best=5)
        lv_N.append({'N': N, **{k: r[k] for k in ('psll', 'eff', 'fwhm', 'aperture')}})
        print(f"  N={N:3d}  PSLL {r['psll']:+6.2f} dB (랜덤바닥 {floor(N):+.1f}) "
              f"| FWHM {r['fwhm']:.2f}° | 개구 {r['aperture']:.0f}µm")

    print('\n=== 지렛대 2: 충전율 d_min (N=32, 비등간격+진폭) ===')
    lv_d = []
    for dmin in (2.0, 1.0, 0.775):
        r = run(32, dmin, epochs=1000, best=5)
        lv_d.append({'d_min': dmin, **{k: r[k] for k in ('psll', 'eff', 'fwhm', 'aperture')}})
        tag = ' (=λ/2)' if abs(dmin - 0.775) < 1e-3 else ''
        print(f"  d_min={dmin:.3f}µm{tag:7s}  PSLL {r['psll']:+6.2f} dB "
              f"| FWHM {r['fwhm']:.2f}° | 개구 {r['aperture']:.0f}µm")

    E.N, E.DMIN = 32, 2.0
    json.dump({'lever_N': lv_N, 'lever_dmin': lv_d, 'floor_ref': {n: round(floor(n), 2) for n in (32, 64, 128)}},
              open(os.path.join(RESULTS, 'lever_experiment.json'), 'w'), indent=2, ensure_ascii=False)
    print(f"\n산출물 → {RESULTS}/lever_experiment.json")


if __name__ == '__main__':
    main()
