# Opt4_Joint — 간격 + 위상 공동 최적화 (역할 분담형 Adam)

기존 Opt1~3은 소자 간격을 3 µm 상수로 고정하고 **위상만** 최적화했다.
등간격 3 µm / λ=1.55 µm에서는 sinθ = ±λ/d = ±31.1° grating lobe가 물리적으로 불가피하며
(위상으로는 제거 불가), element factor 감쇠분 −1.65 dB가 PSLL의 한계였다.

Opt4는 **간격과 위상을 각각의 물리적 역할에 맞게 분담**시킨다:

| 변수 | 역할 | 근거 |
|---|---|---|
| 위상 φ | 주엽 형성 · 조향 · 캘리브레이션 | 임의 배치 x에서 φ = k·x·sinθ₀로 \|AF(u₀)\| = N 전역 최적 도달 가능 |
| 간격 d | 비주기화로 grating lobe / PSLL 억제 | 위상 정렬 평형에서 사이드로브 구조는 순수 {xₙ}의 함수 |

## 핵심 설계

- **모델**: I(u) = [sinc(w·u/λ)·|Σₙ e^{j(k·xₙ·u − φₙ)}| / N]², u = sinθ.
  xₙ = n·3µm 대입 시 기존 Opt1 수식과 항등 (rtol 1e-9 회귀 테스트).
- **간격 파라미터화**: dₙ = d_min + (d_max−d_min)·sigmoid(sₙ) — 박스 제약 [2, 5] µm 구조 보장
  (투영/페널티 불필요).
- **손실**: L = −10log₁₀ I(u₀) + w_sll · SoftPSLL_β,
  SoftPSLL = (1/β)·logsumexp(β·Dᵢ), Dᵢ = 10log₁₀(Iᵢ/I(u₀)) — 주엽 상대화로
  "주엽 낮춰 PSLL 개선" 퇴화 차단. 기존 grating lobe 인덱스 하드코딩(589/1211) 폐기,
  마스크(|u−u₀| > 2λ/L)를 매 epoch 재계산.
- **최적화**: torch.optim.Adam (기존 수동 Adam의 bias correction 이중 적용 버그 제거).
  두-시간척도(φ lr 3e-2 > s lr 1e-2) + 3-phase 스케줄(warmup 50 → joint 500 → polish 100)
  + β 어닐링(0.2→2.0 [1/dB]) + 멀티스타트 16회.
- **2단계 워크플로 (하드웨어 현실)**: 간격 = 제작 전 1회 설계(Stage A),
  위상 = 런타임 캘리브레이션(Stage B, 간격 동결을 코드로 강제).

## 실행

```bash
python main_design.py             # Stage A : 단일각(θ=0) 설계 + 실험 전체 (~15 s, CPU)
python main_design_multiangle.py  # Stage A': 다각도(±30°) 설계 — 조향 강건 간격 (~5 min)
python main_calibration.py        # Stage B : 위상 캘리브레이션 E2 (20 trials)
python test_opt4.py               # 검증 스위트 7종
```

의존성: torch, numpy, pandas, matplotlib (CPU, float64 — CUDA/Lumerical 불필요)

## 결과 (N=32, seed 42~57)

| 구성 | PSLL | 주엽 I(u₀) |
|---|---|---|
| A. 등간격 3µm + 위상만 (baseline) | **−1.65 dB** (grating lobe) | 1.000 |
| B. joint 공동 최적화 (best, 스냅 후) | **−15.46 dB** | 1.000 |
| C. 간격만 (위상 = 해석 조향해) | −14.03 dB | 1.000 |

- 억제량 **13.8 dB** (N=32 랜덤 배열 물리 한계 ≈ −15 dB에 근접), 주엽 손실 0, FWHM 0.88°(등간격과 동일)
- E1 조향 스윕: 간격 동결, 위상만 재조향 → ±30° 전 구간 PSLL ≤ −11.1 dB (grating lobe 재출현 없음)
- E2 캘리브레이션: 무작위 위상오차 U(−π,π) 20회 → 주엽 회복 평균 100.0% / 최저 99.9%
- E3 강건성 MC (간격 σ=20nm, 위상 σ=0.05rad, 1000회): PSLL p95 = −14.24 dB
- E4 5nm 공정 스냅: 열화 0.06 dB

산출물: `results/final_spacing.csv`(확정 간격), `final_layout.csv`(위치+위상),
`design_report.json`(전 실험 지표), `pattern_overlay.png` 등 플롯 3종.

## 다각도(multi-angle) 설계 — 조향 전 범위 로브 억제

단일각(θ=0) 설계는 조향 시 가시창 이동 + EF 비율 열세로 PSLL이 열화된다
(−15.5 dB @ 0° → −11.1 dB @ ±30°). `main_design_multiangle.py`는 조향각 집합
{0, ±10, ±20, ±30}° 전체의 soft-PSLL을 worst-case soft-max(γ=2)로 집계해 간격을
최적화한다. 각도별 위상 = 해석 조향해 + 학습형 잔차 δφ_a (설계 단계 bilevel).
간격은 완전 비등간격 — 31개 전부 상호 상이(5nm 스냅 후 중복 강제 제거).

결과 (N=32, 5° 조밀 스윕, 간격 동결 + 위상만 재조향):

| θ | 0° | ±10° | ±20° | ±30° |
|---|---|---|---|---|
| PSLL | −13.3 dB | −13.2 dB | −12.9 dB | −11.5 dB |
| 주엽 배열인자 효율 | 0.96 | 0.97 | 0.96 | 0.97 |

- 전 조향 범위에서 프로파일 평탄 — 설계각 사이 구멍 없음 (`psll_vs_angle.png`)
- 조향 시 주엽 원시 강도 하락(0.68× @ 30°)은 소자폭 1µm의 EF 포락선 물리 —
  간격/위상으로 제거 불가. 배열인자 효율(exEF)은 전 각도 ≥ 0.96으로 주엽 건재.
  더 필요하면 소자폭 축소(w<1µm, fab 단계) 또는 채널 수 증가(N↑ → 바닥 1/N)로만 개선 가능.
- ±30° worst −11.5 dB는 N=32·가시창 확장(|v|≤1.5)·EF 비율(−1.5 dB) 조건의 이론 바닥
  (−12~−13 dB) 근방. 산출물: `final_spacing_multiangle.csv`, `design_report_multiangle.json`,
  `psll_vs_angle.png`, `pattern_steered.png`.

## Lumerical 연동 (v2 확장 경로)

간격은 시뮬레이터-인-더-루프에 넣지 않는다(지오메트리 재메싱 비용 + 상호결합으로 surrogate
gradient 신뢰도 저하). 권장: Stage A 확정 간격으로 .lms 소스/도파로를 1회 재배치(setnamed)한 뒤,
Opt3의 `Lumerical_Func.Phase`를 경로만 바꿔 위상 캘리브레이션에 재사용.

주의: d_min=2µm는 보수적 가정 — 실제 도파로 evanescent 결합 스펙/공정 최소 선폭 확인 후 조정.
