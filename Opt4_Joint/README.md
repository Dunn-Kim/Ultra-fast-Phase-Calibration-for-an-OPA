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

## 최적화 너머 (beyond) — 방법론 한계 실측

`experiment_beyond.py` + 4방향 병렬 탐색으로 대안 방법론을 전수 벤치마크
(공통 하네스 `benchmark_common.py`, 조향 스윕 ±30° worst PSLL):

| 방법 | worst PSLL | 판정 |
|---|---|---|
| 다각도 Adam (기존 챔피언) | −11.54 dB | 기준 |
| bilevel CMA-ES (polish 내장 목적) | −11.62 dB | 구 챔피언 (`final_spacing_beyond.csv`) |
| **손실 정비: β_end 2→6 + 각도 5° 조밀 (`experiment_loss_ablation.py`)** | **−11.75 dB** | **현 챔피언** (`final_spacing_lossfix.csv`, 1° 스윕 검증) |
| v-공간 재정식화 (닫힌형 W(v), 연속각 보증) | −11.36~−11.57 dB | 동급 + @0° −14.2 개선, 7× 고속 |
| 조합 ILS (격자 1-opt) | −11.40 dB | 동급 |
| CMA-ES/좌표하강 (해석위상 목적) | −10.4~−11.4 dB | 순위 역전 — 나쁜 대리변수 실증 |
| 결정론 배열 (golden/chirp/prime) | −5~−9.8 dB | 탈락 (quasi-Bragg 피크) |
| NN 회귀 surrogate (`experiment_dl.py`) | −7.2~−9.3 dB | 탈락 — surrogate 착취 실증 (R²=0.46, 예측 −14.0 vs 실제 −7.2) |
| 신경망 재파라미터화 (deep prior) | −11.03 dB | 탈락 — 동일 목적, NN 파라미터화가 직접 최적화에 열세 |

핵심 결론:
0. **손실 노브 ablation** (`results/loss_ablation.json`): 유효 노브 = soft-max 온도
   β_end 2→6 (+0.20 dB, soft-hard 갭 축소) + 설계각 10°→5° 조밀화 (+0.20 dB).
   무효 노브 = 가드마스크 κ(±0.05), 각도집계 γ(±0.1, 방향 비일관), lr(옵티마이저).
   결합 + 풀예산(24×1500) → **−11.75 dB** (worst, 1° 스윕 검증) — 손실 정비가
   탐색기 교체(CMA +0.08)보다 컸음.
1. **순위 역전 발견**: 해석 조향해 위상만으로 간격을 평가하면 polish 후 순위가 뒤집힘 —
   모든 설계 목적함수는 위상 polish를 내장(bilevel)해야 함.
2. **소프트웨어 포화**: 세 독립 방법군(Adam/CMA/ILS)이 전부 −11.4~−11.66 평탄 군집 수렴 →
   N=32·균일진폭·d_min 2µm·w=1µm 제약의 전역 최적 ≈ −11.6~−12.0 dB. 배치 개선 여지 소진.
3. **다음 지렛대는 하드웨어**: 소자폭 w 1.0→0.4µm 축소가 EF 기울기 페널티(+1.54 dB@30°,
   알고리즘으로 제거 불가) 를 없애 worst −12.3±0.4 dB 예상(무작위 배치 60개 실측 평균 +0.97 dB).
   대가 = 절대 방사효율 하락. 그 외 N 배증(~3 dB), d_min 축소(결합 억제 구조 필요).
4. 기각 확정(실측): 이중피치 인터리브(+0.1 dB 악화), DE(−9.7), SA(기여 0),
   λ-스티어링(이동량 < 빔폭), 진폭 taper 단독(+0.8 dB뿐).

## 채널 수 스케일링 최종 산출물 (조향 ±30° worst-case, d_min 2µm, 5nm 스냅)

| N | worst PSLL | @0° | 개구 | 배치 CSV |
|---|---|---|---|---|
| 32 | **−11.75 dB** | −13.84 | 92 µm | `final_spacing_lossfix.csv` |
| 64 | **−14.76 dB** | −16.64 | 193 µm | `final_spacing_N64.csv` |
| 128 | **−17.41 dB** | −18.39 | 383 µm | `final_spacing_N128.csv` |

## 문헌 검증 (웹 조사 워크플로, 논문 전수 URL 확인)

- 도메인 SOTA = 완전 랜덤 배치 + 전역/구배 최적화 + 다각도 worst-case 목적
  (Komljenovic, Opt. Express 2017 → Wang, Appl. Opt. 2022 → Zang, Photonics 2025).
  본 파이프라인은 이 계보의 상위호환.
- 문헌 앵커 대비: Yu 2024(N=64 랜덤셔플) @0° −13.46 → ±45° **−8.27 붕괴**;
  Hutchison 2016(실칩 N=128, min 5.4µm) ±45° >10 dB; Elsheikh 2024(N=100 GA) −11 dB;
  Qiu LPR 2024(N=120 GA) 12.8–13.5 dB — **본 결과는 전 구간 문헌 상회/상단**.
- 이론 바닥(Lo 1964/Steinberg 확률배열): 평균 1/N + 피크 마진 → N=32 실질 바닥
  −12~−13 dB. −11.75는 바닥 0.3~1 dB 이내.
- 문헌 선정 추가 구현(`experiment_slp.py`): SLP-minimax 볼록 폴리시(You 2017 AWPL
  계보) −11.59, 대량 랜덤시딩 30k(Yu 2024 계보) −11.50 — 챔피언 미돌파,
  포화 재확증. DL 교체 근거는 문헌에도 없음(offline MBO의 objective hacking).

## Gray-box 모델오차 보정 (Kennedy-O'Hagan model discrepancy)

해석 모델은 AF(간섭항)만 정확 — 계통오차는 EF 포락선(obliquity·모드형상·결합)에 몰림.
EF에만 보정변수 추가, AF 해석식 유지 (`experiment_graybox.py`, config `ef_*` 필드):

```
EF_corr(u) = sinc(w·u/λ) · (1−u²)^(p/2) · exp(a1·u²+a2·u⁴+a3·u⁶)
             └ top-hat ┘  └ obliquity 구조항 ┘  └ 잔차 g, g(0)=0 ┘
```

**핵심 규율 (confounding 회피, opus 워크플로가 KOH 2001/Brynjarsdóttir 2014/Plumlee 2017로 검증):**
θ=(p,g)는 관측(단채널 far-field)으로만 회귀, 간격 x는 θ 동결 모델로만 최적화. **분리 필수** —
θ·x 공동최적화는 `inf_θ PSLL=−∞` 퇴화(EF가 broadside 스파이크로 붕괴), `degeneracy_demo()` 실증.

obliquity를 구조항으로 분리한 게 결정적: 자유 다항만으로는 대각도 EF 오차(60° +7.7dB, 80° +17dB)를
꼬리 발산 없이 못 잡음. 구조항 후 잔차 회귀 RMS 0.09 dB, 전 각도 보정오차 ≤0.14 dB.

**모델오차의 실제 영향 — 헤드라인 PSLL은 ~1.6~1.7 dB 낙관 편향** (관측대용 TRUE=Gaussian×√(1−u²) 기준):

| N | 보고값(naive) | 실제(TRUE) | 낙관 편향 |
|---|---|---|---|
| 32 | −11.75 dB | **−10.1 dB** | 1.6 dB |
| 64 | −14.76 dB | **−13.0 dB** | 1.7 dB |
| 128 | −17.41 dB | **−15.7 dB** | 1.7 dB |

- 보정모델 재설계 시 예측-실측 갭 1.6dB → **0.1dB (정직)**. 설계 이득 자체는 +0.2dB로 작음
  (EF는 매끈 → 배치 순위 거의 불변) — 진짜 가치는 **제작 전 신뢰 가능한 절대 PSLL**
- 잔존 리스크: 상호결합은 x-의존이라 단채널 EF로 흡수 못 함. 목표 간격대 2차 소량 여부 확인 필요
- 관측 없으면 자유 다항 금지 → 물리 2변수(w_eff,p) + 로버스트 min-max로 대체
- 기본 config(ef 무보정)는 기존과 항등 (test_opt4 7/7 통과). 산출물: `graybox_experiment.json`

## Lumerical 연동 (v2 확장 경로)

간격은 시뮬레이터-인-더-루프에 넣지 않는다(지오메트리 재메싱 비용 + 상호결합으로 surrogate
gradient 신뢰도 저하). 권장: Stage A 확정 간격으로 .lms 소스/도파로를 1회 재배치(setnamed)한 뒤,
Opt3의 `Lumerical_Func.Phase`를 경로만 바꿔 위상 캘리브레이션에 재사용.

주의: d_min=2µm는 보수적 가정 — 실제 도파로 evanescent 결합 스펙/공정 최소 선폭 확인 후 조정.
