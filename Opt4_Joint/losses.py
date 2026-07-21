# 손실 설계 — 역할 분담이 구조에 내장됨:
#   L_main  = −10·log10(I(u0))         : 위상 담당 (임의 x에서 φ=k·x·u0로 전역 최소 도달 가능)
#   SoftPSLL = (1/β)·LSE(β·D_i), i∈SL  : 간격 담당 (위상 정렬 평형에서 사이드로브는 순수 x의 함수)
#   D_i = 10·log10(I_i / I(u0))        : 주엽으로 상대화 → "주엽 낮춰 PSLL 개선" 퇴화 차단
# 기존 하드코딩 grating lobe 인덱스(589/1211) 전면 폐기 — 마스크를 매 epoch u-공간에서 재계산
import math

import torch as th


def main_lobe_loss(model, x, phi, u0=None):
    i0 = model.intensity_at_u0(x, phi, u0)
    return -10.0 * th.log10(i0 + model.cfg.eps), i0


def sidelobe_mask(model, x, u, u0=None):
    # SL = { u_i : |u_i − u0| > Δ },  Δ = κ·λ/L_ap (L_ap = 현재 개구, detach)
    c = model.cfg
    u0 = model.u0 if u0 is None else u0
    l_ap = x[-1].detach() - x[0].detach()
    delta = c.guard_kappa * c.wavelength / l_ap
    return (u - u0).abs() > delta


def soft_psll(model, x, phi, beta, u=None, u0=None):
    c = model.cfg
    u = model.u_train if u is None else u
    intens = model.intensity(x, phi, u)
    i0 = model.intensity_at_u0(x, phi, u0)
    mask = sidelobe_mask(model, x, u, u0)
    d_db = 10.0 * th.log10(intens[mask] / (i0 + c.eps) + c.eps)
    return th.logsumexp(beta * d_db, dim=0) / beta


def total_loss(model, x, phi, beta, w_sll, u=None, u0=None):
    lm, i0 = main_lobe_loss(model, x, phi, u0)
    if w_sll > 0.0:
        ls = soft_psll(model, x, phi, beta, u, u0)
        return lm + w_sll * ls, i0
    return lm, i0


def multi_angle_worst_psll(model, s, beta, angles_deg, gamma=1.0, dphi=None):
    # 다각도 설계 손실: 조향각 집합 전체의 soft-PSLL을 worst-case soft-max로 집계.
    #   각도별 위상 = 해석 조향해 k·x·u0 (주엽 정렬 정확 유지) + 학습형 잔차 δφ_a (사이드로브
    #   미세 정형 — 각도별 위상의 역할을 설계 단계까지 확장한 bilevel).
    #   해석 항은 x.detach() 기반이라 간격 gradient는 AF의 k·x_n·u 항으로만 흐름.
    #   soft-PSLL의 D_i = I_i/I(u0) 상대화가 주엽 하락을 자동 벌점화하므로 δφ가 주엽을
    #   희생하는 퇴화는 차단됨. EF(u)/EF(u0) 비율이 각도별로 정확히 포함(평가 지표와 동일).
    x = model.positions(s)
    per_angle = []
    for i, a in enumerate(angles_deg):
        u0 = math.sin(math.radians(a))
        phi = model.steering_phase(x, u0)
        if dphi is not None:
            phi = phi + dphi[i]
        per_angle.append(soft_psll(model, x, phi, beta, u0=u0))
    stacked = th.stack(per_angle)
    return th.logsumexp(gamma * stacked, dim=0) / gamma


def coupling_penalty(model, d, weight=1.0):
    # 커플링 페널티 (config: cpl_*) — 간격 분포의 "바닥 몰림" 벌점. d>d_min 하드 제약과 별개.
    #   (a) 물리형: 전력 결합 ∝ exp(−γ·엣지갭) 의 총량 근사 (coupled-mode, γ는 포스터 하드웨어 유도)
    #   (b) 장벽형: EF 적합 영역 밖(d<d_safe)에서만 켜지는 이차 softplus 장벽
    # weight = 스윕용 전역 배율 (기본 1 = config 가중 그대로). 반환 단위 = dB 등가.
    c = model.cfg
    phys = th.exp(-c.cpl_gamma * (d - c.element_width)).mean()
    barrier = th.nn.functional.softplus((c.cpl_d_safe - d) / c.cpl_tau).pow(2).mean()
    return weight * (c.cpl_w_phys * phys + c.cpl_w_barrier * barrier)


def beta_schedule(t, total, beta_start, beta_end):
    # 지수 어닐링: 초기엔 전 사이드로브 균등 억제, 후기엔 peak 집중
    frac = min(max(t / max(total, 1), 0.0), 1.0)
    return beta_start * (beta_end / beta_start) ** frac


def w_sll_schedule(t, ramp_epochs):
    return min(1.0, t / max(ramp_epochs, 1))
