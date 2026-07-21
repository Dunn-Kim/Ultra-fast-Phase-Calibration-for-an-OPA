# Opt4_Joint 설정
# 내부 단위 규약: 길이 µm, 각도 radian, dtype float64, device cpu 기본
from dataclasses import dataclass, field, asdict
import torch as th


@dataclass
class JointConfig:
    # 배열 물리
    line_N: int = 32           # 소자 수
    wavelength: float = 1.55   # µm
    element_width: float = 1.0 # µm
    d_init: float = 3.0        # µm (등간격 출발점 = 기존 Opt1~3 규약)
    d_min: float = 2.0         # µm (도파로 상호결합/공정 하한, 사양 open question)
    d_max: float = 5.0         # µm (개구 상한 = (N-1)*d_max 구조 보장)
    theta_target_deg: float = 0.0

    # 소자인자 EF 보정 (gray-box model-discrepancy).
    #   EF(u) = sinc(w_eff·u/λ) · (1−u²)^(p/2) · exp(a1·u²+a2·u⁴+a3·u⁶)
    #   ├ w_eff: 유효 개구폭[µm] (None=element_width)  ├ oblq_p: obliquity 지수 (0=없음, 1=cosθ)
    #   └ ef_gcoef: 짝수 다항 잔차 계수 (관측회귀로만 학습 — PSLL과 분리). g(0)=0 구조 보존.
    # 기본값 = 실 MODE(varFDTD) 적합치. Opt3_Back_Forward/Resultants 의 위상↔패턴 로그
    #   (N=32 100쌍)로 적합, N=64/128 홀드아웃 R² 0.996 전이 검증. 강도 배율
    #   g_I(u)=(1−u²)^0.318·exp(−3.27u²+0.561u⁴+0.749u⁶) → 진폭 계수는 그 절반.
    #   순수 sinc(구 기본값)로 되돌리려면 ef_oblq_p=0.0, ef_gcoef=(0,0,0).
    ef_w_eff: float = None
    ef_oblq_p: float = 0.318
    ef_gcoef: tuple = (-1.635, 0.2805, 0.3745)

    # 손실
    guard_kappa: float = 2.0   # 주엽 가드밴드 Δ = κ·λ/L_ap
    beta_start: float = 0.2    # soft-PSLL 온도 [1/dB]
    beta_end: float = 2.0
    w_sll_ramp_epochs: int = 100
    eps: float = 1e-12

    # 최적화 (수정 Adam: torch.optim.Adam, 이중 bias-correction 버그 없음)
    lr_phase: float = 3e-2     # rad 스케일
    lr_spacing: float = 1e-2   # sigmoid 로짓 스케일
    lr_phase_polish: float = 1e-2
    epochs_warmup: int = 50    # 위상만 (w_sll=0)
    epochs_joint: int = 500    # 동시 갱신 + β 어닐링 + w_sll 램프
    epochs_polish: int = 100   # 간격 동결, 위상 재수렴
    restarts: int = 16         # 멀티스타트 (간격 로짓 초기 섭동 시드만 변경)
    s_init_std: float = 0.1    # 등간격 로짓 주변 섭동

    # 격자
    n_grid_train: int = 4001   # u-균등 [-1, 1]
    n_grid_val: int = 40001

    # 다각도 설계 (조향 전 범위에서 양측 로브 억제)
    design_angles_deg: tuple = (0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0)
    angle_agg_gamma: float = 1.0   # 각도 간 worst-case soft-max 온도 [1/dB]

    # 제작
    fab_snap_nm: float = 5.0   # 공정 그리드 스냅 [nm]

    # 재현성
    seed: int = 42
    dtype: object = field(default=th.float64, repr=False)
    device: str = 'cpu'        # MPS는 float64/complex 미지원 → cpu 고정

    def to_dict(self):
        d = asdict(self)
        d['dtype'] = str(self.dtype)
        return d
