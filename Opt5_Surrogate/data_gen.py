# (d, φ) 샘플러 + 스트리밍 배치 + val 팩 생성
#
# train = 스트리밍: 수식이 무한 오라클이므로 배치마다 신선 샘플 (저장·과적합 없음).
# val   = 시드 고정 팩 저장 (검증셋 = 수식 규약). test = MODE 로그 (finetune_mode.py).
import os
import argparse
import numpy as np
import torch as th

from config import SurrogateConfig
from physics_oracle import PhysicsOracle


class SampleStream:
    """혼합 분포에서 (d, φ, E_target) 배치를 생성. gen 시드로 재현성 제어."""

    def __init__(self, cfg: SurrogateConfig, oracle: PhysicsOracle, seed=None):
        self.cfg = cfg
        self.o = oracle
        self.jc = oracle.jcfg
        self.gen = th.Generator().manual_seed(cfg.seed if seed is None else seed)
        self.champ = self._load_champion()

    def _load_champion(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            self.cfg.champion_csv)
        if os.path.exists(path):
            d = np.loadtxt(path, skiprows=1)
            return th.tensor(d, dtype=th.float64)
        return th.full((self.jc.line_N - 1,), self.jc.d_init, dtype=th.float64)

    def _rand(self, *shape):
        return th.rand(*shape, generator=self.gen, dtype=th.float64)

    def _randn(self, *shape):
        return th.randn(*shape, generator=self.gen, dtype=th.float64)

    def sample_d(self, B):
        c, jc = self.cfg, self.jc
        n = jc.line_N - 1
        lo, hi = jc.d_min, jc.d_max
        which = th.multinomial(th.tensor(c.d_mix, dtype=th.float64), B,
                               replacement=True, generator=self.gen)
        d = th.empty(B, n, dtype=th.float64)
        m0, m1, m2 = (which == 0), (which == 1), (which == 2)
        if m0.any():   # 박스 균등
            d[m0] = lo + (hi - lo) * self._rand(int(m0.sum()), n)
        if m1.any():   # d_init 주변 섭동
            d[m1] = jc.d_init + c.d_perturb_std * self._randn(int(m1.sum()), n)
        if m2.any():   # 챔피언 근방
            d[m2] = self.champ.unsqueeze(0) + c.d_champ_std * self._randn(int(m2.sum()), n)
        return d.clamp(lo, hi)

    def sample_phi(self, d):
        c, jc = self.cfg, self.jc
        B, N = d.shape[0], jc.line_N
        u0 = c.u0_range * (2.0 * self._rand(B) - 1.0)
        phi_star = self.o.steering_phase(d, u0)                       # (B, N)
        which = th.multinomial(th.tensor(c.phi_mix, dtype=th.float64), B,
                               replacement=True, generator=self.gen)
        phi = phi_star.clone()
        m1, m2 = (which == 1), (which == 2)
        if m1.any():   # 조향해 + 캘리브레이션형 잡음
            phi[m1] = phi[m1] + c.phi_noise_std * self._randn(int(m1.sum()), N)
        if m2.any():   # 전랜덤
            phi[m2] = 2.0 * th.pi * self._rand(int(m2.sum()), N)
        return th.remainder(phi, 2.0 * th.pi)

    def batch(self, B):
        # → d(B,31) f64, phi(B,32) f64, E(B,1801) complex128
        d = self.sample_d(B)
        phi = self.sample_phi(d)
        with th.no_grad():
            E = self.o.field(d, phi)
        return d, phi, E


def build_val_pack(cfg: SurrogateConfig, path=None):
    o = PhysicsOracle()
    stream = SampleStream(cfg, o, seed=cfg.seed + 10_000)   # train 시드와 분리
    d, phi, E = stream.batch(cfg.val_size)
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                cfg.data_dir, 'val_pack.npz')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path,
                        d=d.numpy().astype(np.float32),
                        phi=phi.numpy().astype(np.float32),
                        E_re=E.real.numpy().astype(np.float32),
                        E_im=E.imag.numpy().astype(np.float32))
    return path


def load_val_pack(cfg: SurrogateConfig, path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                cfg.data_dir, 'val_pack.npz')
    z = np.load(path)
    d = th.tensor(z['d'], dtype=th.float64)
    phi = th.tensor(z['phi'], dtype=th.float64)
    E = th.complex(th.tensor(z['E_re']), th.tensor(z['E_im'])).to(th.complex128)
    return d, phi, E


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', action='store_true', help='val 팩 생성')
    a = ap.parse_args()
    cfg = SurrogateConfig()
    if a.val:
        p = build_val_pack(cfg)
        z = np.load(p)
        print(f'val 팩 저장: {p}')
        print(f"  d {z['d'].shape}  phi {z['phi'].shape}  E {z['E_re'].shape}")
