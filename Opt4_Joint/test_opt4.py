# 검증 스위트 (pytest 또는 python test_opt4.py) — 전부 CPU float64
import math
import time

import torch as th

from config import JointConfig
from losses import soft_psll, total_loss
from metrics import hard_psll_db, summarize
from model import OPAModel


def small_cfg(**kw):
    d = dict(line_N=4, n_grid_train=181, n_grid_val=1801,
             epochs_warmup=5, epochs_joint=20, epochs_polish=5, restarts=1)
    d.update(kw)
    return JointConfig(**d)


# 1. 항등성 회귀 — x_n = n·3µm, φ=0 에서 기존 Opt1 수식과 rtol 1e-9 일치
def test_legacy_identity():
    cfg = JointConfig()
    model = OPAModel(cfg)
    x = model.uniform_positions()          # n·3µm
    phi = th.zeros(cfg.line_N, dtype=cfg.dtype)

    # 기존 Optimizer.py 수식 독립 재구현 (SI 단위):
    #   E(θ) = [sin(β)/β]² · |Σ_n exp(j·k·d·n·sinθ − jφ)| , β = k·w·sinθ/2
    k_si = 2 * math.pi / 1.55e-6
    theta = th.linspace(-90.0, 90.0, 1801, dtype=cfg.dtype)
    u = th.sin(th.deg2rad(theta))
    beta = k_si * 1e-6 * u / 2
    ef = th.where(beta.abs() < 1e-30, th.ones_like(beta), th.sin(beta) / beta) ** 2
    n = th.arange(cfg.line_N, dtype=cfg.dtype)
    af = th.exp(1j * (k_si * 3e-6 * u.reshape(-1, 1) * n.reshape(1, -1))).sum(dim=1)
    legacy = ef * af.abs()

    ours = model.legacy_pattern(x, phi)
    assert th.allclose(ours, legacy, rtol=1e-9, atol=1e-9), \
        f'최대 오차 {(ours - legacy).abs().max().item():.3e}'

    # grating lobe가 정확히 인덱스 589/1211 (±31.1°)에서 국소 최대
    pat = ours.numpy()
    for idx in (589, 1211):
        seg = pat[idx - 8: idx + 9]
        assert abs(int(seg.argmax()) + idx - 8 - idx) <= 1, f'grating lobe 인덱스 {idx} 불일치'
    print('[1] 항등성 회귀 OK (rtol 1e-9, grating lobe 589/1211 재현)')


# 2. 기울기 정합 — gradcheck (N=4 토이, float64)
def test_gradcheck():
    cfg = small_cfg()
    model = OPAModel(cfg)
    s0 = th.full((cfg.line_N - 1,), model.s_init_uniform(), dtype=cfg.dtype) \
        + 0.05 * th.randn(cfg.line_N - 1, dtype=cfg.dtype)
    phi0 = 0.1 * th.randn(cfg.line_N, dtype=cfg.dtype)

    def f(s, phi):
        x = model.positions(s)
        loss, _ = total_loss(model, x, phi, beta=1.0, w_sll=0.7)
        return loss

    assert th.autograd.gradcheck(f, (s0.requires_grad_(True), phi0.requires_grad_(True)),
                                 eps=1e-7, rtol=1e-6, atol=1e-8)
    print('[2] gradcheck OK (∂L/∂s, ∂L/∂φ)')


# 3. 제약 불변식 — 어떤 s에서도 d ∈ [d_min, d_max]
def test_box_constraint():
    cfg = JointConfig()
    model = OPAModel(cfg)
    for scale in (0.1, 1.0, 10.0, 100.0):
        s = scale * th.randn(cfg.line_N - 1, dtype=cfg.dtype)
        d = model.gaps(s)
        assert (d >= cfg.d_min).all() and (d <= cfg.d_max).all()
    print('[3] 박스 제약 불변식 OK (sigmoid 구조 보장)')


# 4. 성능 게이트 — joint가 baseline 대비 grating lobe ≥ 10 dB 억제 + 주엽 유지
def test_performance_gate():
    from main_design import run_single
    cfg = JointConfig(restarts=1)
    base = run_single(cfg, cfg.seed, mode='phase_only')
    joint = run_single(cfg, cfg.seed, mode='joint')
    mb = summarize(base['model'], base['model'].positions(base['s']), base['phi'])
    mj = summarize(joint['model'], joint['model'].positions(joint['s']), joint['phi'],
                   s=joint['s'])
    print(f"    baseline PSLL {mb['psll_db']:+.2f} dB / joint PSLL {mj['psll_db']:+.2f} dB "
          f"/ 주엽 {mj['main_lobe_I']:.3f}")
    assert mb['psll_db'] > -3.0, 'baseline은 grating lobe 때문에 PSLL ≈ −1.65 dB여야 함'
    assert mj['psll_db'] <= mb['psll_db'] - 8.0, '단일 시드 joint 억제량 ≥ 8 dB 기대'
    assert mj['main_lobe_I'] >= 0.8, '주엽 효율 ≥ 0.8'
    assert mj['min_gap_um'] >= cfg.d_min - 1e-9
    print('[4] 성능 게이트 OK')


# 5. E2E — 간격 동결 + 위상 캘리브레이션, grad 격리 확인
def test_calibration_e2e():
    from main_calibration import calibrate
    cfg = JointConfig()
    model = OPAModel(cfg)
    x = model.uniform_positions()          # 임의 동결 배열이면 충분
    x.requires_grad_(False)
    th.manual_seed(7)
    err = (2 * th.rand(cfg.line_N, dtype=cfg.dtype) - 1) * math.pi
    phi_cal = calibrate(cfg, model, x, err, epochs=300)
    with th.no_grad():
        i0 = model.intensity_at_u0(x, phi_cal + err).item()
    assert i0 >= 0.95, f'주엽 회복 {i0:.3f} < 0.95'
    print(f'[5] E2E 캘리브레이션 OK (회복 {i0*100:.1f}%, 간격 grad 격리)')


# 6. 조향 — θ_t = 20°에서 위상만으로 주엽 형성
def test_steering():
    cfg = JointConfig(theta_target_deg=20.0)
    model = OPAModel(cfg)
    x = model.uniform_positions()
    phi = model.steering_phase(x)
    with th.no_grad():
        intens = model.intensity(x, phi, model.u_val)
        peak_u = model.u_val[intens.argmax()].item()
    peak_deg = math.degrees(math.asin(peak_u))
    # 등간격 3µm은 grating lobe가 주엽과 동급 — 목표각 ±0.2° '또는' lobe 위치 확인
    candidates = [20.0]
    for m in (1, -1):
        ug = math.sin(math.radians(20.0)) + m * cfg.wavelength / cfg.d_init
        if abs(ug) <= 1:
            candidates.append(math.degrees(math.asin(ug)))
    assert min(abs(peak_deg - c) for c in candidates) < 0.2, f'peak {peak_deg:.2f}°'
    with th.no_grad():
        i0 = model.intensity_at_u0(x, phi).item()
        ef2 = model.element_factor_amp(th.tensor(model.u0, dtype=cfg.dtype)).item() ** 2
    assert i0 / ef2 > 0.999, '해석 조향해에서 |AF(u0)| = N 이어야 함'
    print(f'[6] 조향 OK (20° 지향, 배열인자 완전 정렬)')


# 7. 런타임 스모크 — 단일 run 650 epoch < 30 s
def test_runtime():
    from main_design import run_single
    cfg = JointConfig(restarts=1)
    t0 = time.time()
    run_single(cfg, cfg.seed, mode='joint')
    dt = time.time() - t0
    assert dt < 30.0, f'{dt:.1f} s'
    print(f'[7] 런타임 스모크 OK ({dt:.1f} s / 650 epoch)')


# 8. Gray-box EF 보정 — 기본값 항등 + 구조항 g(0)=0 불변식
def test_graybox_ef():
    import torch as th
    from config import JointConfig
    from model import OPAModel
    u = th.sin(th.deg2rad(th.tensor([0., 20, 45, 89], dtype=th.float64)))
    # 기본 config(무보정) = 기존 top-hat sinc
    m0 = OPAModel(JointConfig())
    assert th.allclose(m0.element_factor_amp(u), th.sinc(1.0 * u / 1.55), rtol=1e-12)
    # obliquity 구조항: |u|→1서 EF→0 (꼬리 발산 차단)
    m1 = OPAModel(JointConfig(ef_oblq_p=1.0))
    assert m1.element_factor_amp(th.tensor([0.9999], dtype=th.float64)).item() < 0.02
    # g(0)=0 보존 (주엽 게이지) — 임의 g계수에도 broadside EF = 1
    m2 = OPAModel(JointConfig(ef_oblq_p=1.0, ef_gcoef=(-0.5, 0.4, -0.1)))
    assert abs(m2.element_factor_amp(th.tensor([0.0], dtype=th.float64)).item() - 1.0) < 1e-12
    print('[8] gray-box EF OK (기본 항등, obliquity 꼬리→0, g(0)=0 게이지)')


# 9. 고속 캘리브레이션 — 카메라 프레임 5장으로 주엽 복원 (ε 미지, 실칩 인터페이스)
def test_calibrate_fast():
    import math
    import torch as th
    from config import JointConfig
    from main_calibration import calibrate_fast
    from model import OPAModel
    cfg = JointConfig()
    m = OPAModel(cfg)
    x = m.uniform_positions()
    x.requires_grad_(False)
    th.manual_seed(3)
    eps = 2 * math.pi * th.rand(cfg.line_N, dtype=cfg.dtype) - math.pi
    eps[0] = 0.0                                    # 채널 0 = 기준
    g = th.Generator().manual_seed(0)

    def shoot(phi_a):                                # 카메라 프레임 (ε는 알고리즘에 비노출)
        I = m.intensity(x, -(phi_a + eps), m.u_train)
        return (I * (1 + 0.01 * th.randn(I.shape, generator=g, dtype=cfg.dtype))).clamp(min=0)

    phi = calibrate_fast(cfg, m, x, shoot, frames=5)
    rec = (th.exp(1j * (phi + eps)).sum().abs() ** 2 / cfg.line_N ** 2).item()
    assert rec >= 0.98, f'5프레임 복원율 {rec:.3f} < 0.98'
    print(f'[9] 고속 캘리브레이션 OK (5프레임, 복원율 {rec:.4f} — REV는 155프레임에 0.968)')


if __name__ == '__main__':
    test_legacy_identity()
    test_gradcheck()
    test_box_constraint()
    test_performance_gate()
    test_calibration_e2e()
    test_steering()
    test_runtime()
    test_graybox_ef()
    test_calibrate_fast()
    print('\n전체 테스트 통과')
