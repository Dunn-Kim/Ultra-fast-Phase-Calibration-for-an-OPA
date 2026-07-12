# no_grad 하드 지표 — 학습은 soft(LSE), 판정은 hard(max)
import math
import torch as th

from losses import sidelobe_mask


@th.no_grad()
def hard_psll_db(model, x, phi, u=None, u0=None):
    # PSLL[dB] = max_{i∈SL} 10·log10(I_i / I(u0))
    u = model.u_val if u is None else u
    intens = model.intensity(x, phi, u)
    i0 = model.intensity_at_u0(x, phi, u0)
    mask = sidelobe_mask(model, x, u, u0)
    return (10.0 * th.log10(intens[mask] / (i0 + model.cfg.eps) + model.cfg.eps)).max().item()


@th.no_grad()
def main_lobe_efficiency(model, x, phi, u0=None):
    # 정규화상 이상값 1 (u0=0). 조향 시 EF 감쇠 포함/제외 둘 다 보고
    i0 = model.intensity_at_u0(x, phi, u0).item()
    u0v = model.u0 if u0 is None else u0
    ef2 = model.element_factor_amp(
        th.tensor(u0v, dtype=model.cfg.dtype, device=model.cfg.device)).item() ** 2
    return i0, i0 / max(ef2, 1e-30)


@th.no_grad()
def fwhm_deg(model, x, phi, u=None, u0=None):
    # −3 dB 교차점 선형 보간 → 도(deg) 단위 빔폭. 어깨(shoulder) 형성 감시 겸용
    u = model.u_val if u is None else u
    u0 = model.u0 if u0 is None else u0
    intens = model.intensity(x, phi, u)
    half = model.intensity_at_u0(x, phi, u0).item() / 2.0
    i0_idx = int(th.argmin((u - u0).abs()).item())
    iv = intens.cpu().numpy()
    uv = u.cpu().numpy()

    def cross(direction):
        i = i0_idx
        while 0 < i < len(iv) - 1:
            j = i + direction
            if iv[j] < half <= iv[i]:
                t = (iv[i] - half) / (iv[i] - iv[j])
                return uv[i] + t * (uv[j] - uv[i])
            i = j
        return uv[i]

    ul, ur = cross(-1), cross(+1)
    ul, ur = max(min(ul, 1.0), -1.0), max(min(ur, 1.0), -1.0)
    return math.degrees(math.asin(ur)) - math.degrees(math.asin(ul))


@th.no_grad()
def gap_stats(model, s):
    d = model.gaps(s)
    return {'min_gap_um': d.min().item(), 'mean_gap_um': d.mean().item(),
            'max_gap_um': d.max().item(), 'aperture_um': d.sum().item()}


@th.no_grad()
def summarize(model, x, phi, s=None, u0=None):
    i0, eff_ef = main_lobe_efficiency(model, x, phi, u0)
    out = {'psll_db': hard_psll_db(model, x, phi, u0=u0),
           'main_lobe_I': i0, 'main_lobe_eff_exEF': eff_ef,
           'fwhm_deg': fwhm_deg(model, x, phi, u0=u0)}
    if s is not None:
        out.update(gap_stats(model, s))
    return out
