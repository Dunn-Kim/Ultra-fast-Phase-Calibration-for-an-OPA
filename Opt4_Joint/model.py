# 미분가능 OPA 해석 far-field 모델 (비주기 배열 일반화)
#
# 기존 Opt1 (Optimizer.py): AF = Σ exp(j(2αn − φ_n)), 2αn = k·(n·3µm)·sinθ  — 등간격 특수형
# 일반화:                    AF(u) = Σ_n exp(j(k·x_n·u − φ_n)),  u = sinθ, x_n = 소자 위치
#   x_n = n·3µm 대입 시 기존 식과 항등 (재현성 브리지 → test_opt4.py 항등성 회귀)
#
# 간격 파라미터화 (박스 제약 내장, 투영/페널티 불필요):
#   d_n = d_min + (d_max − d_min)·sigmoid(s_n),  x = [0, cumsum(d)]
#
# 강도: I(u) = [ EF_amp(u)·|AF(u)| / N ]²,  EF_amp(u) = sinc(w·u/λ)  (진폭 소자인자)
#   → 강도 소자인자 = sinc² (물리 규약, 등간격 3µm의 grating lobe 감쇠 −1.65 dB 재현)
#   → 정규화 /N 으로 이상적 주엽 강도 = 1 (dB가 절대 의미)
import math
import torch as th
import torch.nn.functional as F


class OPAModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.k = 2.0 * math.pi / cfg.wavelength          # rad/µm
        self.u0 = math.sin(math.radians(cfg.theta_target_deg))
        kw = dict(dtype=cfg.dtype, device=cfg.device)
        self.u_train = th.linspace(-1.0, 1.0, cfg.n_grid_train, **kw)
        self.u_val = th.linspace(-1.0, 1.0, cfg.n_grid_val, **kw)
        # 회귀 테스트 전용: 기존 θ-균등 1801점 격자 (-90:0.1:90 deg)
        theta = th.linspace(-90.0, 90.0, 1801, **kw)
        self.u_theta1801 = th.sin(th.deg2rad(theta))

    # --- 간격/위치 ---
    def gaps(self, s):
        c = self.cfg
        return c.d_min + (c.d_max - c.d_min) * th.sigmoid(s)

    def positions(self, s):
        return F.pad(self.gaps(s).cumsum(-1), (1, 0))      # x_0 = 0, 길이 N [µm]

    def uniform_positions(self, d_fixed=None):
        c = self.cfg
        d = c.d_init if d_fixed is None else d_fixed
        return th.arange(c.line_N, dtype=c.dtype, device=c.device) * d

    def s_init_uniform(self):
        # d_n = d_init 에서 출발하는 로짓 (등간격 baseline과 동일 출발점)
        c = self.cfg
        p = (c.d_init - c.d_min) / (c.d_max - c.d_min)
        return math.log(p / (1.0 - p))

    # --- 패턴 ---
    def element_factor_amp(self, u):
        # EF(u) = sinc(w_eff·u/λ)·(1−u²)^(p/2)·exp(g(u)),  g=a1u²+a2u⁴+a3u⁶ (g(0)=0)
        # 구조항: obliquity (1−u²)^(p/2)를 다항에서 분리 → 꼬리 발산 차단(|u|→1에서 EF→0 강제).
        # 잔차 g는 관측회귀로만 학습(gray-box). 기본값(w_eff=None,p=0,g=0)이면 기존 top-hat sinc와 동일.
        c = self.cfg
        w_eff = c.element_width if c.ef_w_eff is None else c.ef_w_eff
        ef = th.sinc(w_eff * u / c.wavelength)
        if c.ef_oblq_p:
            ef = ef * (1.0 - u ** 2).clamp(min=0.0) ** (c.ef_oblq_p / 2.0)
        a1, a2, a3 = c.ef_gcoef
        if a1 or a2 or a3:
            ef = ef * th.exp(a1 * u ** 2 + a2 * u ** 4 + a3 * u ** 6)
        return ef

    def intensity(self, x, phi, u):
        # I(u) = [ EF_amp·|AF|/N ]²  — x[µm], phi[rad], u = sinθ (임의 격자/스칼라)
        c = self.cfg
        u_ = u.reshape(-1, 1) if u.dim() else u.reshape(1, 1)
        af = th.exp(1j * (self.k * u_ * x.reshape(1, -1) - phi.reshape(1, -1))).sum(dim=1)
        ef = self.element_factor_amp(u.reshape(-1) if u.dim() else u.reshape(1))
        out = (ef * af.abs() / c.line_N) ** 2
        return out if u.dim() else out.squeeze(0)

    def intensity_at_u0(self, x, phi, u0=None):
        u0 = self.u0 if u0 is None else u0
        u = th.tensor(u0, dtype=self.cfg.dtype, device=self.cfg.device)
        return self.intensity(x, phi, u)

    def steering_phase(self, x, u0=None):
        # 해석 조향해: φ_n = k·x_n·u0 (mod 2π) — 임의 위치 배열에서 |AF(u0)| = N 달성
        u0 = self.u0 if u0 is None else u0
        return th.remainder(self.k * x.detach() * u0, 2.0 * math.pi)

    def legacy_pattern(self, x, phi, u=None):
        # 기존 Opt1 규약 재현: E(θ) = sinc²·|AF| (강도 아님 — 항등성 회귀 전용)
        u = self.u_theta1801 if u is None else u
        af = th.exp(1j * (self.k * u.reshape(-1, 1) * x.reshape(1, -1)
                          - phi.reshape(1, -1))).sum(dim=1)
        return self.element_factor_amp(u) ** 2 * af.abs()
