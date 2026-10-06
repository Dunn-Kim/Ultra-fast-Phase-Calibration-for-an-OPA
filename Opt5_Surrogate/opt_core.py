# 최적화 공용 코어 — 챔피언 경로와 백로그 실험이 공유하는 단일 진실원
#
# 여기 모인 것: 설계 프로토콜(각도/판정), 배치 필드 조립, soft/hard PSLL,
#   커플링 페널티, 경사 탐색 루프, 다양성 선택, L-BFGS 연마, 동결 로더.
# 물리 상수·EF 는 Opt4 JointConfig 가 단일 진실원 (physics_oracle 경유) — 여기서 중복 금지.
import math
import time
import numpy as np
import torch as th
import torch.nn.functional as F

from physics_oracle import PhysicsOracle
from model import build_model

# ±15° 설계 프로토콜 (0, ±max/2, ±max) — 챔피언·백로그 공통
DESIGN_ANGLES_DEG = (0.0, 7.5, -7.5, 15.0, -15.0)


# ---------- 장치 ----------

def resolve_device(name):
    # 'auto' → M3 Pro GPU(MPS) 가용 시 mps, 아니면 cpu
    if name == 'auto':
        return 'mps' if th.backends.mps.is_available() else 'cpu'
    return name


def sync(dev):
    # MPS 는 비동기 — 시간 측정 전에 큐를 비워야 실제 소요가 잡힌다
    if str(dev).startswith('mps'):
        th.mps.synchronize()


# ---------- 프로토콜 / 필드 조립 ----------

def u0_vector(dtype=th.float64, angles_deg=None):
    angles_deg = DESIGN_ANGLES_DEG if angles_deg is None else angles_deg
    return th.tensor([math.sin(math.radians(a)) for a in angles_deg],
                     dtype=dtype)


def batched_phi_star(k, d, u0_vec):
    # d(R,N-1), u0 (A,) 공통 또는 (R,A) 재시작별 → 재시작×각도 평탄화:
    #   dRA(R·A,N-1), φ*(R·A,N), u0RA(R·A,)
    R, A = d.shape[0], u0_vec.shape[-1]
    x = F.pad(d.cumsum(-1), (1, 0))
    dRA = d.repeat_interleave(A, dim=0)
    xRA = x.repeat_interleave(A, dim=0)
    u0RA = u0_vec.to(d).expand(R, A).reshape(-1)
    return dRA, k * xRA * u0RA.unsqueeze(1), u0RA


# ---------- 판정 지표 ----------

def guard_mask(u, u0, L_ap, kappa=2.0, wavelength=1.55):
    # 주엽 가드밴드: |u − u0| < κ·λ/L_ap (Opt4 규약)
    du = kappa * wavelength / L_ap
    return (u.reshape(1, -1) - u0.reshape(-1, 1)).abs() < du.reshape(-1, 1)


def psll_db(I, u, u0, L_ap, eps=1e-12):
    # I(B,G), u0(B,), L_ap(B,) → hard PSLL[dB] (B,)
    m = guard_mask(u, u0, L_ap)
    main = th.where(m, I, th.zeros_like(I)).amax(dim=1)
    side = th.where(m, th.full_like(I, -1.0), I).amax(dim=1)
    return 10.0 * th.log10(side.clamp(min=eps) / main.clamp(min=eps))


def soft_psll_per_restart(I, u, u0RA, L_RA, beta, R, A, gamma=1.0, eps=1e-12):
    # I(R·A,G) → 재시작별 soft worst-angle 손실 (R,) — 미분가능 대리 목적
    m = guard_mask(u, u0RA, L_RA)
    main = th.where(m, I, th.zeros_like(I)).amax(dim=1)
    side_db = 10.0 * th.log10(th.where(m, th.full_like(I, eps), I).clamp(min=eps)
                              / main.clamp(min=eps).unsqueeze(1))
    per = (th.logsumexp(beta * side_db, dim=1) / beta).reshape(R, A)
    return th.logsumexp(gamma * per, dim=1) / gamma


def du_weights(u):
    # θ-균등 격자의 Δu 가중 (u=sinθ 라 간격이 변함) — 에너지 적분용 상수
    return th.tensor(np.gradient(u.detach().cpu().numpy()), dtype=u.dtype,
                     device=u.device)


def soft_isl_per_restart(I, u, u0RA, L_RA, R, A, dw=None, eps=1e-12):
    # 사이드로브 총에너지 / 메인로브 에너지 [dB] — "메인로브 제외 전부" 축의 미분가능판
    #   PSLL 은 최대 한 점만 보지만 ISL 은 잡광 전체를 본다. 두 지표는 상충할 수 있다.
    dw = du_weights(u) if dw is None else dw
    m = guard_mask(u, u0RA, L_RA)
    Iw = I * dw.reshape(1, -1)
    e_main = th.where(m, Iw, th.zeros_like(Iw)).sum(dim=1)
    e_side = th.where(m, th.zeros_like(Iw), Iw).sum(dim=1)
    isl_db = 10.0 * th.log10(e_side.clamp(min=eps) / e_main.clamp(min=eps))
    return isl_db.reshape(R, A).mean(dim=1)      # 각도 평균 (worst 는 과도하게 뾰족)


# 제조·구동 오차 규약 — 로버스트 최적화와 평가가 공유하는 기본값
NOISE_POS_UM = 0.05          # 소자 위치 표준편차 [µm] (리소·에칭 편차)
NOISE_PHASE_RAD = math.radians(5.0)   # 인가 위상 표준편차 [rad] (DAC·열드리프트 잔차)


def mc_psll(oracle, d, u0_vec, n=300, sig_d=NOISE_POS_UM, sig_p=NOISE_PHASE_RAD,
            seed=99):
    # 오차 하 worst-angle PSLL 분포 (n회) — 공칭값이 아닌 '실제 기대 성능'
    jc = oracle.jcfg
    g = th.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        dd = (d + sig_d * th.randn(d.shape, generator=g, dtype=d.dtype)
              ).clamp(jc.d_min, jc.d_max)
        dRA, phi, u0RA = batched_phi_star(oracle.k, dd.unsqueeze(0), u0_vec)
        phi = phi + sig_p * th.randn(phi.shape, generator=g, dtype=phi.dtype)
        I = oracle.intensity(dRA, phi)
        out.append(float(psll_db(I, oracle.u, u0RA, dRA.sum(dim=1)).max()))
    return th.tensor(out, dtype=th.float64)


def formula_hard_per_restart(oracle, d, u0_vec):
    # 각 설계의 수식 hard PSLL worst-angle (R,) — 최종 심판
    with th.no_grad():
        dRA, phi, u0RA = batched_phi_star(oracle.k, d.double().cpu(), u0_vec)
        I = oracle.intensity(dRA, phi)
        p = psll_db(I, oracle.u, u0RA, dRA.sum(dim=1))
    return p.reshape(d.shape[0], u0_vec.shape[0]).amax(dim=1)


# ---------- 커플링 페널티 (d>2µm 규약) ----------

def coupling_penalty(d, jc):
    # d(N-1,) → 스칼라, d(R,N-1) → (R,)  (Opt4 losses.coupling_penalty 와 동일식)
    phys = th.exp(-jc.cpl_gamma * (d - jc.element_width)).mean(-1)
    barrier = F.softplus((jc.cpl_d_safe - d) / jc.cpl_tau).pow(2).mean(-1)
    return jc.cpl_w_phys * phys + jc.cpl_w_barrier * barrier


# ---------- 파라미터화 (박스 [d_min, d_max] 하드 보장) ----------

def d_to_logit(d, jc, eps=1e-4):
    p = ((d - jc.d_min) / (jc.d_max - jc.d_min)).clamp(eps, 1.0 - eps)
    return th.log(p / (1.0 - p))


def logit_to_d(s, jc):
    return jc.d_min + (jc.d_max - jc.d_min) * th.sigmoid(s)


def s_init(jc, restarts, gen):
    s0 = d_to_logit(th.tensor(jc.d_init, dtype=th.float64), jc)
    return s0 + 0.1 * th.randn(restarts, jc.line_N - 1, generator=gen,
                               dtype=th.float64)


# ---------- 탐색 (경사) ----------

def run_gradient(eval_I, params_dtype, dev, oracle, restarts, epochs, lr,
                 use_cpl, seed, u_grid, s_start=None, beta_fix=None,
                 trace_every=0, trace_fn=None, u0_vec=None, w_isl=0.0,
                 robust_k=0, sig_d=NOISE_POS_UM, sig_p=NOISE_PHASE_RAD):
    # 공통 경사 루프 — eval_I(dRA, phi) 만 갈아끼움 (서러게이트/수식)
    # trace 계측 평가 시간은 반환 타이밍에서 제외 → 보고 공정성 유지
    jc = oracle.jcfg
    g = th.Generator().manual_seed(seed)
    s = s_init(jc, restarts, g) if s_start is None else s_start.clone()
    s = s.to(params_dtype).to(dev).requires_grad_(True)
    opt = th.optim.Adam([s], lr=lr)
    u0_vec = (u0_vector(params_dtype) if u0_vec is None
              else u0_vec.to(params_dtype)).to(dev)
    A = u0_vec.shape[0]
    dw = du_weights(u_grid) if w_isl else None
    n_eval = 0
    t_train = 0.0
    for ep in range(epochs):
        t0 = time.perf_counter()
        beta = beta_fix if beta_fix is not None else \
            0.2 + (2.0 - 0.2) * min(1.0, ep / (0.6 * epochs))
        d = logit_to_d(s, jc)
        # robust_k>0 이면 재시작마다 K개 오차 실현을 만들어 기대 손실을 최소화한다
        d_eval = d if not robust_k else d.repeat_interleave(robust_k, dim=0)
        R_eval = restarts * max(1, robust_k)
        dRA, phi, u0RA = batched_phi_star(oracle.k, d_eval, u0_vec)
        if robust_k:
            dRA = dRA + sig_d * th.randn(dRA.shape, generator=g,
                                         dtype=dRA.dtype, device=dRA.device)
            phi = phi + sig_p * th.randn(phi.shape, generator=g,
                                         dtype=phi.dtype, device=phi.device)
        I = eval_I(dRA, phi)
        L_RA = dRA.sum(dim=1)
        loss_r = soft_psll_per_restart(I, u_grid, u0RA, L_RA, beta, R_eval, A)
        if w_isl:
            loss_r = loss_r + w_isl * soft_isl_per_restart(
                I, u_grid, u0RA, L_RA, R_eval, A, dw)
        if robust_k:                      # K개 실현의 평균 = 오차 하 기대 손실
            loss_r = loss_r.reshape(restarts, robust_k).mean(dim=1)
        if use_cpl:
            loss_r = loss_r + coupling_penalty(d, jc)
        loss_r.sum().backward()      # 재시작별 독립 (s 행 분리 + Adam 원소별)
        opt.step()
        opt.zero_grad(set_to_none=True)
        n_eval += R_eval * A
        sync(dev)
        t_train += time.perf_counter() - t0
        if trace_fn is not None and trace_every \
                and (ep % trace_every == 0 or ep == epochs - 1):
            with th.no_grad():
                d_tr = logit_to_d(s, jc).cpu().double()
            trace_fn(ep, t_train, d_tr, loss_r.detach().cpu())
    with th.no_grad():
        d = logit_to_d(s, jc).cpu().double()      # MPS 는 f64 미지원 → CPU 후 캐스팅
    return d, t_train, n_eval


# ---------- 선택 / 연마 ----------

def diverse_topk(d, scores, k, min_dist=0.5):
    # 점수순 그리디 선택, 기선택 설계와 L2 거리 min_dist[µm] 미만이면 건너뜀
    order = scores.argsort().tolist()
    picked = []
    for i in order:
        if all(float((d[i] - d[j]).norm()) > min_dist for j in picked):
            picked.append(i)
        if len(picked) == k:
            break
    for i in order:                     # 부족분은 점수순 보충
        if len(picked) == k:
            break
        if i not in picked:
            picked.append(i)
    return th.tensor(picked, dtype=th.long)


def polish_lbfgs(oracle, s0, jc, u0_vec, beta=3.0, max_iter=150, w_isl=0.0,
                 robust_k=0, sig_d=NOISE_POS_UM, sig_p=NOISE_PHASE_RAD,
                 seed=0):
    # 준뉴턴 연마 (strong Wolfe) — 국소 수렴 단계에서 Adam 수백 에포크를 대체
    # robust_k>0: 고정된 K개 오차 실현(공통 난수)으로 기대 손실을 연마 —
    #   L-BFGS 는 결정론적 목적함수를 요구하므로 매 클로저마다 새 샘플을 뽑지 않는다
    R, A = s0.shape[0], u0_vec.shape[0]
    dw = du_weights(oracle.u) if w_isl else None
    if robust_k:
        gn = th.Generator().manual_seed(seed)
        eps_d = sig_d * th.randn(R * robust_k, jc.line_N - 1, generator=gn,
                                 dtype=th.float64)
        eps_p = sig_p * th.randn(R * robust_k * A, jc.line_N, generator=gn,
                                 dtype=th.float64)
    s = s0.clone().double().requires_grad_(True)
    opt = th.optim.LBFGS([s], max_iter=max_iter, history_size=25,
                         line_search_fn='strong_wolfe',
                         tolerance_grad=1e-10, tolerance_change=1e-12)
    n_closure = 0

    def closure():
        nonlocal n_closure
        n_closure += 1
        opt.zero_grad()
        d = logit_to_d(s, jc)
        d_eval = d if not robust_k else d.repeat_interleave(robust_k, dim=0)
        R_eval = R * max(1, robust_k)
        dRA, phi, u0RA = batched_phi_star(oracle.k, d_eval, u0_vec)
        if robust_k:                       # 고정 오차 실현 (결정론적 목적함수)
            dRA = dRA + eps_d.repeat_interleave(A, dim=0)
            phi = phi + eps_p
        I = oracle.intensity(dRA, phi)
        L_RA = dRA.sum(dim=1)
        loss_r = soft_psll_per_restart(I, oracle.u, u0RA, L_RA, beta,
                                       R_eval, A)
        if w_isl:
            loss_r = loss_r + w_isl * soft_isl_per_restart(
                I, oracle.u, u0RA, L_RA, R_eval, A, dw)
        if robust_k:
            loss_r = loss_r.reshape(R, robust_k).mean(dim=1)
        loss = (loss_r + coupling_penalty(d, jc)).sum()
        loss.backward()
        return loss

    t0 = time.perf_counter()
    opt.step(closure)
    elapsed = time.perf_counter() - t0
    with th.no_grad():
        d = logit_to_d(s, jc).double()
    return d, elapsed, n_closure * R * max(1, robust_k) * A


# ---------- v-도메인 엔진 (조향해 φ* 전용, GPU 친화) ----------
#
# φ = φ*(u0) 이면 AF_a(u) = W(u − u0_a),  W(v) = Σ_n exp(j·k·x_n·v)  — 각도마다 다시 풀 필요가 없다.
# 균등 v 격자 v_m = v0 + (p·Q + q)·dv 에서 W[p, q] = Σ_n exp(jkx_n(v0+pQdv))·exp(jkx_n·q·dv)
#   → (R,P,N)@(R,N,Q) 배치 행렬곱. 원소별 (R·A, G, N) 텐서 대비 메모리 병목이 사라진다.
# 격자는 균등 u (θ-균등 판정 격자와 다름) — 탐색·연마용. 최종 판정은 formula_hard_per_restart.

DV_FINE = math.sin(math.radians(0.1))      # θ-격자 정면 간격과 동일


class VDomain:
    def __init__(self, oracle, u0_vec, dv=DV_FINE, device='cpu', dtype=th.float32):
        self.jc, self.k = oracle.jcfg, oracle.k
        half = int(math.ceil((1.0 + float(u0_vec.abs().max())) / dv))
        self.Q = int(math.ceil(math.sqrt(2 * half + 1)))
        self.P = int(math.ceil((2 * half + 1) / self.Q))
        v = (th.arange(self.P * self.Q, dtype=th.float64) - half) * dv   # v=0 이 격자점
        u = v.reshape(1, -1) + u0_vec.double().reshape(-1, 1)           # (A, M)
        valid = u.abs() <= 1.0
        ef2 = oracle.opa.element_factor_amp(u.clamp(-1.0, 1.0)) ** 2
        self.ef2 = (th.where(valid, ef2, th.zeros_like(ef2))
                    / self.jc.line_N ** 2).to(device, dtype)
        self.valid = valid.to(device)
        self.v = v.to(device, dtype)
        self.vp = (v[0] + th.arange(self.P, dtype=th.float64) * self.Q * dv).to(device, dtype)
        self.vq = (th.arange(self.Q, dtype=th.float64) * dv).to(device, dtype)

    def W2(self, d):
        # d(R,N-1) → |W(v)|² (R, M)
        kx = F.pad(d.cumsum(-1), (1, 0)).unsqueeze(2) * self.k          # (R,N,1)
        ap, bq = kx * self.vp, kx * self.vq
        Ar, Ai = th.cos(ap).transpose(1, 2), th.sin(ap).transpose(1, 2)
        Br, Bi = th.cos(bq), th.sin(bq)
        Wr, Wi = Ar @ Br - Ai @ Bi, Ar @ Bi + Ai @ Br
        return (Wr ** 2 + Wi ** 2).reshape(d.shape[0], -1)

    def parts(self, d):
        # → I(R,A,M), 메인로브 가드 마스크, 사이드로브 마스크
        I = self.ef2.unsqueeze(0) * self.W2(d).unsqueeze(1)
        guard = (self.v.reshape(1, 1, -1).abs()
                 < (2.0 * self.jc.wavelength / d.sum(-1)).reshape(-1, 1, 1))
        return I, guard, self.valid.unsqueeze(0) & ~guard

    def soft_loss(self, d, beta, w_isl=0.0, gamma=1.0, eps=1e-12):
        # 재시작별 soft worst-angle PSLL [+ w_isl·각도평균 ISL] (R,)
        I, guard, side = self.parts(d)
        main = th.where(guard, I, th.zeros_like(I)).amax(-1, keepdim=True)
        sdb = 10.0 * th.log10(th.where(side, I, th.full_like(I, eps)).clamp(min=eps)
                              / main.clamp(min=eps))
        per = th.logsumexp(beta * sdb, dim=-1) / beta
        loss = th.logsumexp(gamma * per, dim=-1) / gamma
        if w_isl:
            e_main = th.where(guard, I, th.zeros_like(I)).sum(-1)
            e_side = th.where(side, I, th.zeros_like(I)).sum(-1)
            loss = loss + w_isl * (10.0 * th.log10(e_side.clamp(min=eps)
                                                   / e_main.clamp(min=eps))).mean(-1)
        return loss

    @th.no_grad()
    def hard_psll(self, d, eps=1e-12):
        # v-격자 hard worst-angle PSLL [dB] (R,) — 순위용 근사 (판정은 θ-격자)
        I, guard, side = self.parts(d)
        main = th.where(guard, I, th.zeros_like(I)).amax(-1)
        sd = th.where(side, I, th.zeros_like(I)).amax(-1)
        return (10.0 * th.log10(sd.clamp(min=eps) / main.clamp(min=eps))).amax(-1)


# ---------- 동결 로더 ----------

def load_frozen(cfg, ckpt_path):
    # 모사 모델을 동결 상태로 적재 — 최적화가 가중치를 건드리지 못하게 한다
    oracle = PhysicsOracle()
    model = build_model(cfg, oracle)
    model.load_state_dict(th.load(ckpt_path, map_location='cpu')['state'])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, oracle
