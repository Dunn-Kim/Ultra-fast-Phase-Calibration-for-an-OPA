# 클린시트 리디자인 실험 — "모든 로브 변수" = 위치 + 위상 + 진폭(apodization) 공동 최적화
#
# 물음: 비등간격이 우수한데, 처음부터 중앙 로브 최대화 + 양측 로브 억제로
#       모든 자유도를 변수화하면 성능이 더 나올까?
#
# 물리 답:
#  - N=32·균일진폭에서 peak sidelobe는 정보이론 바닥 ≈ 10log10(1/N) = −15 dB 근방.
#    현재 간격최적화(−13.3 dB)는 이미 이 바닥에 근접 → 간격만으론 한계.
#  - 바닥을 깨는 유일한 추가 변수 = 진폭 가중 a_n (element excitation, apodization).
#    Chebyshev/Taylor 원리: 에너지를 중앙으로 모아 사이드로브를 임의 깊이로 낮춤.
#    대가 = 주엽 지향성(개구효율 η) 하락 + 빔폭 증가. 단일 최적해 아닌 Pareto 전선.
#
# 모델: I(u) = EF(u)²·|Σ_n a_n e^{j(k x_n u − φ_n)}|² / (Σ a_n)²
#   개구효율 η = (Σa_n)² / (N Σa_n²)  (균일=1, taper<1 → 주엽 directivity 손실 지표)
#
# 실행: venv/bin/python experiment_redesign.py
import json
import math
import os

import torch as th

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
LAM, W, N = 1.55, 1.0, 32
DMIN, DMAX, DINIT = 2.0, 5.0, 3.0
K = 2 * math.pi / LAM
DT = th.float64


def positions(s):
    d = DMIN + (DMAX - DMIN) * th.sigmoid(s)
    return th.cat([th.zeros(1, dtype=DT), th.cumsum(d, 0)])


def amps(la, learn):
    return th.nn.functional.softplus(la) if learn else th.ones(N, dtype=DT)


def intensity(x, phi, a, u):
    u_ = u.reshape(-1, 1)
    af = (a.reshape(1, -1) * th.exp(1j * (K * u_ * x.reshape(1, -1) - phi.reshape(1, -1)))).sum(1)
    ef = th.sinc(th.as_tensor(W, dtype=DT) * u.reshape(-1) / LAM)
    return (ef.abs() * af.abs() / a.sum()) ** 2


def i_at(x, phi, a, u0):
    return intensity(x, phi, a, th.tensor([u0], dtype=DT))[0]


def taper_eff(a):
    return (a.sum() ** 2 / (N * (a ** 2).sum())).item()


def hard_psll(x, phi, a, u0, ug):
    I = intensity(x, phi, a, ug)
    i0 = i_at(x, phi, a, u0)
    L = x[-1].detach() - x[0].detach()
    mask = (ug - u0).abs() > 2 * LAM / L
    return (10 * th.log10(I[mask] / (i0 + 1e-12) + 1e-12)).max().item()


def fwhm(x, phi, a, u0, ug):
    I = intensity(x, phi, a, ug).detach()
    half = i_at(x, phi, a, u0).item() / 2
    i0 = int(th.argmin((ug - u0).abs()))
    U, Iv = ug.tolist(), I.tolist()

    def cr(dr):
        i = i0
        while 0 < i < len(Iv) - 1:
            j = i + dr
            if Iv[j] < half <= Iv[i]:
                t = (Iv[i] - half) / (Iv[i] - Iv[j])
                return U[i] + t * (U[j] - U[i])
            i = j
        return U[i]
    ul, ur = max(cr(-1), -1), min(cr(1), 1)
    return math.degrees(math.asin(ur)) - math.degrees(math.asin(ul))


def soft_psll(x, phi, a, u0, ug, beta):
    I = intensity(x, phi, a, ug)
    i0 = i_at(x, phi, a, u0)
    L = x[-1].detach() - x[0].detach()
    mask = (ug - u0).abs() > 2 * LAM / L
    D = 10 * th.log10(I[mask] / (i0 + 1e-12) + 1e-12)
    return th.logsumexp(beta * D, 0) / beta


def optimize(learn_spacing, learn_amp, w_eff=0.0, seed=0, epochs=1200, u0=0.0):
    # w_eff: 개구효율 유지 벌점 가중 (0이면 순수 최저 PSLL 추구 → taper 깊어짐)
    th.manual_seed(seed)
    s = th.full((N - 1,), math.log((DINIT - DMIN) / (DMAX - DINIT)), dtype=DT)
    if learn_spacing:
        s = (s + 0.3 * th.randn(N - 1, dtype=DT)).requires_grad_(True)
    la = th.zeros(N, dtype=DT)  # softplus(0)=0.693 ~ 균일 출발
    if learn_amp:
        la = (la + 0.05 * th.randn(N, dtype=DT)).requires_grad_(True)
    phi = th.zeros(N, dtype=DT, requires_grad=True)
    params = [phi] + ([s] if learn_spacing else []) + ([la] if learn_amp else [])
    opt = th.optim.Adam([{'params': [phi], 'lr': 3e-2},
                         {'params': [s] if learn_spacing else [], 'lr': 1e-2},
                         {'params': [la] if learn_amp else [], 'lr': 2e-2}])
    ug = th.linspace(-1, 1, 4001, dtype=DT)
    for t in range(epochs):
        beta = 0.2 * (2.0 / 0.2) ** (t / epochs)
        opt.zero_grad()
        x = positions(s)
        a = amps(la, learn_amp)
        # 위상은 해석 조향해로 고정(주엽 정렬) → 손실은 순수 위치/진폭 담당
        phi_eff = th.remainder(K * x.detach() * u0, 2 * math.pi) if learn_spacing or learn_amp else phi
        loss = soft_psll(x, phi_eff, a, u0, ug, beta)
        if learn_amp and w_eff > 0:
            loss = loss - w_eff * 10 * th.log10(th.tensor(1.0, dtype=DT) * (a.sum() ** 2 / (N * (a ** 2).sum())))
        loss.backward()
        opt.step()
    with th.no_grad():
        x = positions(s)
        a = amps(la, learn_amp)
        phi_f = th.remainder(K * x * u0, 2 * math.pi)
        uv = th.linspace(-1, 1, 40001, dtype=DT)
        return {'psll': hard_psll(x, phi_f, a, u0, uv), 'eff': taper_eff(a),
                'fwhm': fwhm(x, phi_f, a, u0, uv),
                'x': x.tolist(), 'a': (a / a.max()).tolist(),
                'aperture': (x[-1] - x[0]).item()}


def best_of(n, **kw):
    return min((optimize(seed=i, **kw) for i in range(n)), key=lambda r: r['psll'])


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print('=== 4개 구성 비교 (θ=0, N=32, best-of-8) ===')
    configs = {
        'A 등간격+균일진폭 (기존 Opt1)': dict(learn_spacing=False, learn_amp=False),
        'B 비등간격+균일진폭 (현재 사양)': dict(learn_spacing=True, learn_amp=False),
        'C 등간격+진폭taper': dict(learn_spacing=False, learn_amp=True, w_eff=0.0),
        'D 비등간격+진폭taper (리디자인)': dict(learn_spacing=True, learn_amp=True, w_eff=0.0),
    }
    out = {}
    for name, kw in configs.items():
        r = best_of(8, **kw)
        out[name] = r
        print(f"  {name:32s}  PSLL {r['psll']:+6.2f} dB | 개구효율 {r['eff']:.3f} "
              f"| FWHM {r['fwhm']:.2f}° | 개구 {r['aperture']:.1f}µm")

    print('\n=== Pareto: 리디자인(D) 진폭효율 벌점 스윕 — 사이드로브 vs 주엽 지향성 ===')
    pareto = []
    for w in [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2]:
        r = best_of(6, learn_spacing=True, learn_amp=True, w_eff=w)
        pareto.append({'w_eff': w, 'psll': r['psll'], 'eff': r['eff'], 'fwhm': r['fwhm']})
        print(f"  w_eff={w:4.2f}  PSLL {r['psll']:+6.2f} dB | 개구효율 {r['eff']:.3f} "
              f"(주엽손실 {10*math.log10(r['eff']):+.2f} dB) | FWHM {r['fwhm']:.2f}°")

    json.dump({'configs': out, 'pareto': pareto},
              open(os.path.join(RESULTS, 'redesign_experiment.json'), 'w'),
              indent=2, ensure_ascii=False)
    print(f"\n산출물 → {RESULTS}/redesign_experiment.json")


if __name__ == '__main__':
    main()
