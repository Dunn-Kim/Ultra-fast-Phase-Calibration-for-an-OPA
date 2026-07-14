# 공정 비교 하네스 — 배치(x)만 받아 동일 프로토콜로 조향 강건성 평가.
# 프로토콜: 조향 스윕 θ0 ∈ {−30..30, 5° 간격}, 각도별 위상 = 해석 조향해 + polish(100ep),
#           PSLL은 40001점 검증 격자 하드 max. 지표 = worst PSLL (+ 각도별 테이블).
# 챔피언 기준선: 다각도 Adam 설계 worst −11.54 dB.
import math

import torch as th

from config import JointConfig
from losses import total_loss
from metrics import fwhm_deg, hard_psll_db
from model import OPAModel

SWEEP_DEG = tuple(range(-30, 31, 5))


def make_model(cfg=None):
    return OPAModel(cfg or JointConfig())


def polish_phase(model, x, u0, epochs=100, lr=1e-2):
    cfg = model.cfg
    phi = model.steering_phase(x, u0).clone().requires_grad_(True)
    opt = th.optim.Adam([phi], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss, _ = total_loss(model, x, phi, cfg.beta_end, 1.0, u0=u0)
        loss.backward()
        opt.step()
    return phi.detach()


@th.no_grad()
def _metrics_at(model, x, phi, u0):
    psll = hard_psll_db(model, x, phi, u=model.u_val, u0=u0)
    i0 = model.intensity_at_u0(x, phi, u0).item()
    return psll, i0


def evaluate_layout(x, model=None, polish=True, sweep=SWEEP_DEG):
    """x: 위치 텐서[µm]. 반환: {'worst_psll', 'sweep': [{theta, psll, i0}], 'aperture'}"""
    model = model or make_model()
    x = x.to(dtype=model.cfg.dtype)
    rows = []
    for a in sweep:
        u0 = math.sin(math.radians(a))
        phi = polish_phase(model, x, u0) if polish else model.steering_phase(x, u0)
        psll, i0 = _metrics_at(model, x, phi, u0)
        rows.append({'theta_deg': a, 'psll_db': round(psll, 2), 'i0': round(i0, 4)})
    worst = max(r['psll_db'] for r in rows)
    return {'worst_psll_db': worst,
            'psll_at_0': next(r['psll_db'] for r in rows if r['theta_deg'] == 0),
            'sweep': rows,
            'aperture_um': round((x[-1] - x[0]).item(), 2),
            'min_gap_um': round((x[1:] - x[:-1]).min().item(), 4),
            'fwhm0_deg': round(fwhm_deg(model, x, model.steering_phase(x, 0.0), u0=0.0), 3)}


def check_constraints(x, d_min=2.0, d_max=5.0, tol=1e-9):
    d = x[1:] - x[:-1]
    ok = bool((d >= d_min - tol).all() and (d <= d_max + tol).all())
    return ok, {'min': d.min().item(), 'max': d.max().item()}


def gaps_to_positions(d):
    zero = th.zeros(1, dtype=d.dtype)
    return th.cat([zero, th.cumsum(d, dim=0)])
