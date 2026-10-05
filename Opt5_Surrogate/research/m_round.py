# 회차 최소화 파이프라인 — 탐색(Adam) → 배치 saddle-free Newton(λ 사다리) → 정확 minimax SLP
#
# 회차 = 순차 배치 평가 수. 장비가 있으면 한 회차 안의 병렬 폭은 공짜에 가깝다 →
#   · Newton: 31차원 정확 헤시안 1회차 + λ 사다리(K 스텝 동시 평가) 1회차 = 반복당 2회차
#   · SLP: 후보 간 독립 → 같은 반복을 병렬로 진행한 것으로 계상 (회차 = 반복 수)
import math

import numpy as np
import torch as th
import torch.nn.functional as F
from scipy.optimize import linprog

from opt_core import coupling_penalty


# ---------- 판정 θ-격자 정확 평가 + 해석 야코비안 ----------

def theta_eval(ctx, d, jac=True):
    # d(31,) f64 → S(A,G) 사이드로브 dB(가드 밖, 메인 대비; 가드 안 −inf), J, dS(A,G,31)
    o, jc = ctx.o, ctx.jc
    u, ef = o.u, o.ef
    N = jc.line_N
    x = F.pad(d.cumsum(0), (1, 0))
    w = u.reshape(1, -1) - ctx.u0v.reshape(-1, 1)                 # (A,G) = u − u0
    E = th.exp(1j * o.k * w.unsqueeze(-1) * x)                       # (A,G,N)
    AF = E.sum(-1)
    I = (ef ** 2).reshape(1, -1) * AF.abs() ** 2 / N ** 2
    guard = w.abs() < 2.0 * 1.55 / d.sum()                          # opt_core.guard_mask 규약
    im = th.where(guard, I, th.zeros_like(I)).argmax(-1)              # (A,)
    a = th.arange(I.shape[0])
    logI = th.log10(I.clamp(min=1e-30)) * 10.0
    S = th.where(guard, th.full_like(I, -math.inf), logI - logI[a, im].unsqueeze(-1))
    cpl = coupling_penalty(d, jc)
    J = float(S.max() + cpl)
    ctx.meter.add((3 if jac else 1) * u.numel() / o.u.numel())
    if not jac:
        return S, J, None
    # d logI/dx_n = −2k·w·Im(conj(AF)·E_n)/|AF|²  → dS = (10/ln10)(dlogI(u) − dlogI(u_main))
    dl = -2.0 * o.k * w.unsqueeze(-1) * (AF.conj().unsqueeze(-1) * E).imag \
        / (AF.abs() ** 2).clamp(min=1e-30).unsqueeze(-1)
    dS = (10.0 / math.log(10.0)) * (dl - dl[a, im].unsqueeze(1))      # (A,G,N) wrt x
    dS = dS.flip(-1).cumsum(-1).flip(-1)[..., 1:]                     # x_n = Σ_{m<n} d_m
    return S, J, dS


def cpl_grad(ctx, d):
    d = d.detach().clone().requires_grad_(True)
    c = coupling_penalty(d, ctx.jc)
    g, = th.autograd.grad(c, d)
    return float(c), g


def slp(ctx, d, iters=10, delta=1.5, rho=0.02, rho_max=0.2, tol=1e-6):
    # epigraph 신뢰영역 SLP: min t + ∇cpl·Δ  s.t. S_i + ∇S_i·Δ ≤ t (S_i ≥ max S − δ), |Δ|∞ ≤ ρ, 박스
    # 공로(merit) = 판정 J 그 자체 → 수락된 스텝은 J 를 단조 감소시킨다.
    jc = ctx.jc
    d = d.double().cpu().clamp(jc.d_min, jc.d_max)
    S, J, dS = theta_eval(ctx, d)
    n = d.numel()
    for _ in range(iters):
        act = S >= S.max() - delta
        Gk, sk = dS[act].numpy(), S[act].numpy()
        c0, gc = cpl_grad(ctx, d)
        dn = d.numpy()
        lo = np.maximum(-rho, jc.d_min - dn)
        hi = np.minimum(rho, jc.d_max - dn)
        res = linprog(np.r_[gc.numpy(), 1.0],
                      A_ub=np.c_[Gk, -np.ones(len(sk))], b_ub=-sk,
                      bounds=list(zip(lo, hi)) + [(None, None)], method='highs')
        if res.status != 0:
            break
        step = th.from_numpy(res.x[:n])
        pred = J - (res.x[n] + c0 + float(gc @ step))
        if pred < tol:
            break
        d_new = (d + step).clamp(jc.d_min, jc.d_max)
        S_new, J_new, dS_new = theta_eval(ctx, d_new)
        r = (J - J_new) / pred
        if J_new < J:
            d, S, J, dS = d_new, S_new, J_new, dS_new
        rho = min(2.0 * rho, rho_max) if r > 0.75 else (0.25 * rho if r < 0.25 else rho)
        if rho < 1e-6:
            break
    return d, J


# ---------- 배치 saddle-free Newton ----------

def grad_hess(f, s):
    # 행별 독립 목적 f:(R,n)→(R,) 의 기울기 (R,n) 와 헤시안 (R,n,n)
    s = s.detach().requires_grad_(True)
    L = f(s)
    g, = th.autograd.grad(L.sum(), s, create_graph=True)
    n = s.shape[1]
    eye = th.eye(n, dtype=s.dtype, device=s.device).unsqueeze(1).expand(n, *s.shape)
    H, = th.autograd.grad(g, s, grad_outputs=eye, is_grads_batched=True)   # (n,R,n)
    return L.detach(), g.detach(), H.permute(1, 0, 2).detach()


def newton(ctx, f, s, iters, K=8, e_lo=-4.0, e_hi=1.0, step_max=2.0, frac=1.0, fused=False,
           chunk=1024):
    # saddle-free Newton: p(λ) = −Q (|Λ| + λ)⁻¹ Qᵀ g, λ = λ_max·10^e — K개 스텝을 동시 평가해 행마다 최선 채택.
    # fused: 사다리 K점 전부에서 (L, g, H) 를 같은 회차에 계산해 다음 반복에 재사용
    #   → 반복당 1회차 (비융합 2회차). 궤적은 비융합과 같고 연산만 K배.
    R, n = s.shape
    ex = th.linspace(e_lo, e_hi, K, dtype=th.float64)
    r = th.arange(R)
    L, g, H = grad_hess(f, s)
    ctx.meter.add(R * frac * 2 * n)                         # 헤시안 = n 개 HVP (≈2단위씩)
    for it in range(iters):
        if it and not fused:
            L, g, H = grad_hess(f, s)
            ctx.meter.add(R * frac * 2 * n)
        lam, Q = th.linalg.eigh(0.5 * (H + H.transpose(1, 2)).cpu().double())
        a = lam.abs()
        gq = (Q.transpose(1, 2) @ g.cpu().double().unsqueeze(-1)).squeeze(-1)    # (R,n)
        lk = a.amax(-1, keepdim=True) * 10.0 ** ex                                  # (R,K)
        p = -(Q.unsqueeze(1) @ (gq.unsqueeze(1) / (a.unsqueeze(1) + lk.unsqueeze(-1))
                                ).unsqueeze(-1)).squeeze(-1)                      # (R,K,n)
        p = p * (step_max / p.abs().amax(-1, keepdim=True).clamp(min=step_max))   # ‖p‖∞ ≤ step_max
        cand = (s.unsqueeze(1).cpu().double() + p).to(s.device, s.dtype)
        if fused:
            # R·K 행 헤시안을 한 번에 만들면 중간 텐서가 수 GB → MPS 에서 결과가 깨진다 (R 1024 실측).
            # 같은 회차 안의 연산을 메모리만 나눠 계산하고, 청크마다 늘어난 회차 계측은 되돌린다.
            parts = [grad_hess(f, c) for c in cand.reshape(R * K, n).split(chunk)]
            ctx.meter.round(1 - len(parts))
            Lc, gc, Hc = (th.cat(x) for x in zip(*parts))
            ctx.meter.add(R * K * frac * 2 * n)
            Lc, gc, Hc = Lc.reshape(R, K), gc.reshape(R, K, n), Hc.reshape(R, K, n, n)
        else:
            with th.no_grad():
                Lc = f(cand.reshape(R * K, n)).reshape(R, K)
        j = Lc.argmin(-1)
        better = Lc[r, j] < L
        s = th.where(better.unsqueeze(-1), cand[r, j], s)
        if fused:                                           # 채택 안 된 행은 현재점 (L, g, H) 유지
            L = th.where(better, Lc[r, j], L)
            g = th.where(better.unsqueeze(-1), gc[r, j], g)
            H = th.where(better.reshape(R, 1, 1), Hc[r, j], H)
    return s


def run(ctx, seed, budget, R=1024, s_std=0.3, epochs=10, lr=0.08, ex_dv=2.0, ex_isl=0.3,
        ex_bar=3.0, nt_iters=25, nt_dv=2.0, nt_beta=2.0, nt_isl=0.3, nt_bar=3.0, K=8,
        step_max=2.0, fused=False, slp_k=64, slp_iters=10, slp_delta=1.5, n_out=64):
    # 기본값 = 챔피언 (8시드 J −13.552, 71회차 / fused 47회차) — 근거는 research/README.md
    g = th.Generator().manual_seed(seed)
    n = ctx.jc.line_N - 1

    # 1) 탐색: Adam, β 0.2→2.0 어닐링 (연속화로 거친 지형을 매끈하게 시작)
    vx = ctx.vdomain(ex_dv)
    s = (ctx.s_center + s_std * th.randn(R, n, generator=g, dtype=th.float64))
    s = s.to(ctx.device, th.float32).requires_grad_(True)
    opt = th.optim.Adam([s], lr=lr)
    for ep in range(epochs):
        b = 0.2 + 1.8 * min(1.0, ep / max(1.0, 0.6 * epochs))
        d = ctx.logit_to_d(s)
        (vx.soft_loss(d, b, ex_isl) + ctx.penalty(d, ex_bar)).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    # 2) 전 후보 배치 Newton (선별 없음 — 연마 전 순위는 연마 후 품질을 예측하지 못한다)
    vn = ctx.vdomain(nt_dv)

    def f(z):
        d = ctx.logit_to_d(z)
        return vn.soft_loss(d, nt_beta, nt_isl) + ctx.penalty(d, nt_bar)
    s = newton(ctx, f, s.detach(), nt_iters, K=K, step_max=step_max, frac=vn.frac, fused=fused)

    # 3) 정밀 v-격자 순위 → 상위 slp_k 를 판정 θ-격자 hard minimax 로 마무리
    d = ctx.logit_to_d(s.cpu().double())
    vd64 = ctx.vdomain(1.0, th.float64, 'cpu')
    d = d[(vd64.hard_psll(d) + ctx.penalty(d)).argsort()]
    if slp_k:
        d = th.cat([th.stack([slp(ctx, x, slp_iters, slp_delta)[0] for x in d[:slp_k]]),
                    d[slp_k:]])
        ctx.meter.round(slp_iters)                          # 후보 병렬 → 회차 = 반복 수
    return d[:n_out]
