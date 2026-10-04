# 수식 오라클 — Opt4 OPAModel 재사용 브리지 + 배치 필드 합성
#
# 서러게이트 타깃 = 정규화 복소장 E(u) = EF_amp(u)·AF(u)/N,
#   AF(u) = Σₙ exp(j(k·xₙ·u − φₙ))  (Opt4 부호 규약 그대로)
# 강도 I = |E|² 는 OPAModel.intensity 와 항등 — test_opt5.py 회귀 게이트로 고정.
import os
import math
import importlib.util
import torch as th
import torch.nn.functional as F

# Opt5에도 config.py/model.py가 있어 바레 임포트는 sys.modules에서 충돌 →
# Opt4 모듈을 고유 이름으로 파일 직접 로드 (Opt4가 단일 진실원)
_OPT4 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Opt4_Joint')


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_OPT4, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


JointConfig = _load('opt4_config', 'config.py').JointConfig
OPAModel = _load('opt4_model', 'model.py').OPAModel


class PhysicsOracle:
    """MODE-적합 EF 기본값의 canonical 수식을 배치로 평가하는 타깃 생성기."""

    def __init__(self, jcfg: JointConfig = None):
        self.jcfg = jcfg or JointConfig()
        self.opa = OPAModel(self.jcfg)
        self.k = self.opa.k                       # rad/µm
        self.u = self.opa.u_theta1801             # (1801,) f64 — MODE 로그 격자
        self.ef = self.opa.element_factor_amp(self.u)   # (1801,) f64, 진폭 EF

    @property
    def n_grid(self):
        return self.u.numel()

    def positions(self, d):
        # d: (B, N-1) 간격[µm] → x: (B, N) 위치[µm], x₀=0
        return F.pad(d.cumsum(-1), (1, 0))

    def field(self, d, phi):
        # d: (B, N-1), phi: (B, N) [rad] → E: (B, 1801) complex128
        x = self.positions(d.to(self.u.dtype))                      # (B, N)
        ph = (self.k * self.u.reshape(1, -1, 1) * x.unsqueeze(1)
              - phi.to(self.u.dtype).unsqueeze(1))                  # (B, G, N)
        af = th.exp(1j * ph).sum(dim=2)                             # (B, G)
        return self.ef.unsqueeze(0) * af / self.jcfg.line_N

    def intensity(self, d, phi):
        return self.field(d, phi).abs() ** 2

    def steering_phase(self, d, u0):
        # 해석 조향해 φₙ = k·xₙ·u₀ (Opt4.steering_phase 배치판)
        x = self.positions(d.to(self.u.dtype))
        u0 = u0 if th.is_tensor(u0) else th.tensor(u0, dtype=x.dtype)
        return th.remainder(self.k * x * u0.reshape(-1, 1), 2.0 * math.pi)
