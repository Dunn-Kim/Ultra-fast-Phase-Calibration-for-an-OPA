# Opt5 회귀 게이트 — 오라클/모델/데이터 계약을 고정한다
#
# [1] 오라클 배치 강도 ≡ Opt4 OPAModel.intensity (단일 진실원)
# [2] NeuralElement 초기상태 ≡ 수식 (zero-init 게인 → 항등, f32 허용오차)
# [3] MLP 입출력 형상 + encode_inputs 차원 계약
# [4] 조향해 φ* → 주엽 피크가 u0 에 정확히 형성
# [5] MODE 로더 형상 + 부호 판정 + 물리모델 기준 R² (수식이 MODE를 설명하는 수준)
# [6] 프레임별 LS 게인 → 카메라/시뮬 임의 스케일 불변 (calibrate_fast 규약)
# [7] MPS ↔ CPU forward 패리티 (MPS 가용 시)
import os
import sys
import math
import torch as th
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import SurrogateConfig                     # noqa: E402
from physics_oracle import PhysicsOracle               # noqa: E402
from model import build_model, encode_inputs, input_dim  # noqa: E402
sys.path.insert(0, os.path.join(HERE, 'backlog'))
from finetune_mode import (load_mode_pairs, pick_phase_sign,   # noqa: E402
                           frame_r2, predict_intensity)


@pytest.fixture(scope='module')
def oracle():
    return PhysicsOracle()


def _rand_batch(oracle, B=4, seed=0):
    g = th.Generator().manual_seed(seed)
    jc = oracle.jcfg
    d = jc.d_min + (jc.d_max - jc.d_min) * th.rand(B, jc.line_N - 1,
                                                   generator=g, dtype=th.float64)
    phi = 2.0 * math.pi * th.rand(B, jc.line_N, generator=g, dtype=th.float64)
    return d, phi


def test_oracle_matches_opt4(oracle):
    d, phi = _rand_batch(oracle)
    I_batch = oracle.intensity(d, phi)
    x = oracle.positions(d)
    for b in range(d.shape[0]):
        I_ref = oracle.opa.intensity(x[b], phi[b], oracle.u)
        assert th.allclose(I_batch[b], I_ref, rtol=1e-9, atol=1e-12)


def test_element_init_is_formula(oracle):
    d, phi = _rand_batch(oracle, seed=1)
    model = build_model(SurrogateConfig(arch='element'), oracle).eval()
    with th.no_grad():
        pred = model(d, phi).double()
    E = oracle.field(d, phi)
    err = th.complex(pred[:, 0], pred[:, 1]) - E
    rmse = err.abs().pow(2).mean().sqrt().item()
    assert rmse < 1e-3, f'element zero-init이 수식과 불일치: rmse={rmse:.2e}'


def test_mlp_shapes(oracle):
    d, phi = _rand_batch(oracle, seed=2)
    z = encode_inputs(d, phi, oracle.k)
    assert z.shape == (d.shape[0], input_dim(oracle.jcfg.line_N))
    model = build_model(SurrogateConfig(arch='mlp'), oracle).eval()
    with th.no_grad():
        pred = model(d, phi)
    assert pred.shape == (d.shape[0], 2, oracle.n_grid)


def test_steering_phase_peaks_at_u0(oracle):
    d, _ = _rand_batch(oracle, B=8, seed=3)
    u0 = th.linspace(-0.4, 0.4, 8, dtype=th.float64)
    phi = oracle.steering_phase(d, u0)
    I = oracle.intensity(d, phi)
    u_peak = oracle.u[I.argmax(dim=1)]
    # θ-균등 격자라 du = cosθ·dθ 로 변함 → 해당 지점 국소 격자폭의 2배 이내
    du_local = th.cos(th.asin(u0)) * math.radians(0.1)
    assert ((u_peak - u0).abs() <= 2.0 * du_local).all(), \
        f'조향 피크 이탈: {(u_peak - u0).abs().max():.2e}'


def test_mode_loader_and_physics_r2(oracle):
    phi, obs = load_mode_pairs()
    assert phi.shape == (100, 32) and obs.shape == (100, 1801)
    # element 초기상태 = 수식 그 자체 → 물리모델이 MODE를 설명하는 수준을 게이트로 고정
    model = build_model(SurrogateConfig(arch='element'), oracle).eval()
    d = th.full((phi.shape[0], oracle.jcfg.line_N - 1), oracle.jcfg.d_init,
                dtype=th.float64)
    sign, r2 = pick_phase_sign(model, d, phi, obs)
    assert r2 > 0.95, f'수식↔MODE R² 붕괴: {r2:.4f} (기대 ≈0.996)'


def test_frame_gain_scale_invariance(oracle):
    phi, obs = load_mode_pairs()
    model = build_model(SurrogateConfig(arch='element'), oracle).eval()
    d = th.full((8, oracle.jcfg.line_N - 1), oracle.jcfg.d_init, dtype=th.float64)
    with th.no_grad():
        I = predict_intensity(model, d, phi[:8])
    r_base = frame_r2(I, obs[:8])
    r_scaled = frame_r2(I, obs[:8] * 7371.5)   # 카메라 임의 게인
    assert th.allclose(r_base, r_scaled, atol=1e-9)


def test_optimization_keeps_weights_frozen(oracle):
    # [8] 동결 게이트 — 간격 최적화(run_gradient)가 모델 가중치를 1비트도 바꾸지 않음
    import hashlib
    from opt_core import run_gradient

    model = build_model(SurrogateConfig(arch='mlp'), oracle).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def whash(m):
        h = hashlib.sha256()
        for k, v in sorted(m.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def eval_I(dRA, phi):
        pred = model(dRA, phi)
        return pred[:, 0] ** 2 + pred[:, 1] ** 2

    h0 = whash(model)
    run_gradient(eval_I, th.float32, 'cpu', oracle, 4, 5, 1e-2, True, 0,
                 oracle.u.float())
    assert whash(model) == h0, '최적화가 모델 가중치를 변경함 — 동결 위반'
    assert sum(int(p.requires_grad) for p in model.parameters()) == 0


@pytest.mark.skipif(not th.backends.mps.is_available(), reason='MPS 없음')
@pytest.mark.parametrize('arch', ['mlp', 'element'])
def test_mps_cpu_parity(oracle, arch):
    d, phi = _rand_batch(oracle, seed=4)
    d32, p32 = d.float(), phi.float()
    model = build_model(SurrogateConfig(arch=arch), oracle).eval()
    with th.no_grad():
        out_cpu = model(d32, p32)
        m_mps = model.to('mps')
        out_mps = m_mps(d32.to('mps'), p32.to('mps')).cpu()
    rmse = (out_cpu - out_mps).pow(2).mean().sqrt().item()
    assert rmse < 1e-3, f'{arch} MPS/CPU 불일치: rmse={rmse:.2e}'


def test_vdomain_matches_formula(oracle):
    # [10] v-도메인 bmm 강도 ≡ Opt4 OPAModel.intensity (조향해 φ*, 유효 격자점 전부)
    from opt_core import VDomain, u0_vector
    d, _ = _rand_batch(oracle, B=2, seed=5)
    u0v = u0_vector()
    vd = VDomain(oracle, u0v, dtype=th.float64)
    I, _, _ = vd.parts(d)
    x = oracle.positions(d)
    for b in range(d.shape[0]):
        for a in range(u0v.shape[0]):
            m = vd.valid[a]
            u = vd.v[m] + u0v[a]
            phi = oracle.opa.steering_phase(x[b], float(u0v[a]))
            I_ref = oracle.opa.intensity(x[b], phi, u)
            assert th.allclose(I[b, a, m], I_ref, rtol=1e-8, atol=1e-12)
