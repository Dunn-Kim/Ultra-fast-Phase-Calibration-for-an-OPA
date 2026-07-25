# Opt5_Surrogate 설정 — 물리 상수/EF는 Opt4 JointConfig가 단일 진실원 (여기 중복 금지)
from dataclasses import dataclass, field, asdict
import torch as th


@dataclass
class SurrogateConfig:
    # --- (d, φ) 샘플링 분포 (train 스트리밍 + val 팩 공용) ---
    # d 혼합: 박스 균등 / d_init 주변 섭동 / 챔피언 근방 섭동 (합=1)
    d_mix: tuple = (0.5, 0.3, 0.2)
    d_perturb_std: float = 0.3     # µm, d_init 주변
    d_champ_std: float = 0.15      # µm, 챔피언 간격 주변
    champion_csv: str = '../Opt4_Joint/results/final_spacing_realEF_pm15_cpl.csv'
    # φ 혼합: 조향해 φ*(u0) / φ*+잡음(캘리브레이션형) / 전랜덤 (합=1)
    phi_mix: tuple = (0.5, 0.25, 0.25)
    u0_range: float = 0.5          # 조향 u0 ~ U(−0.5, 0.5) (≈±30°)
    phi_noise_std: float = 0.3     # rad

    # --- 모델 ---
    arch: str = 'mlp'              # 'mlp' | 'element' | 'siren' | 'ffmlp'
    hidden: tuple = (1024, 1024, 1024, 1024)
    elem_hidden: tuple = (256, 256)   # NeuralElement 공유망
    siren_w0: float = 30.0         # SIREN 주파수 스케일 (고차원 입력엔 완화 필요 신호)

    # --- 학습 ---
    batch: int = 512
    steps: int = 20000             # 스트리밍 스텝 (배치당 신선 샘플)
    lr: float = 1e-3
    lr_min: float = 1e-5           # cosine 스케줄 하한
    val_size: int = 4096
    val_every: int = 500
    seed: int = 42
    dtype: object = field(default=th.float32, repr=False)  # NN은 f32 (오라클 f64→캐스팅)
    device: str = 'auto'           # 'auto'(mps 가용 시 mps) | 'cpu' | 'mps' — train 전용
                                   # (finetune/optimize는 f64 리포트 정밀도 위해 CPU 고정)

    # --- 산출 ---
    data_dir: str = 'data'
    ckpt_dir: str = 'checkpoints'
    results_dir: str = 'results'

    def to_dict(self):
        d = asdict(self)
        d['dtype'] = str(self.dtype)
        return d
