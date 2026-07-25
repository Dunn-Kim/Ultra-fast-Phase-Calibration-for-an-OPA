# Opt5_Surrogate — 수식 전체를 회귀하는 2단계 최적화 (ver_fullRegression)

## 브랜치 정의 (혼동 방지)

- **main** — ① 수식 직접형: canonical 수식(Opt4_Joint)에 Adam 옵티마이저를 직결해
  간격을 최적화한다. 위상은 해석해 φ* = k·xₙ·u₀.
- **ver_fullRegression (현 브랜치)** — ② 모사 경유형: 수식을 근사한 모델을 먼저
  구축·동결하고, 이후에는 최적화 알고리즘(Adam)만이 간격 변수를 조작한다.
  모델 가중치는 최적화 중 불변 — test_opt5 동결 게이트(해시 동일성)로 보증.
  실전형 champion 파이프라인은 ② 탐색 뒤 소량(수식 평가 15%)의 수식 재순위·연마로
  마감하며, 수식 호출이 전혀 없는 순수 ②형은 optimize_spacing.py 가 담당한다.

기존 Opt4(수식을 직접 미분해 간격 최적화)와 목표는 같으나 시작을 달리한다:

1. **회귀 모델 구축** — 수식(canonical far-field, MODE-적합 EF 기본값)을 결과값으로,
   동일 변수 (간격 d, 위상 φ)가 입력되었을 때 오차 없이 수식을 근사하는 회귀 모델을 학습.
2. **모델 자유** — MLP 베이스라인 + NeuralElement(원소별 공유망 합 구조) 병행 비교.
3. **MODE fine-tune** — 낮은 오차의 회귀 모델을 Lumerical MODE 실로그
   (`Opt3_Back_Forward/Resultants`)와 비교 검증하고 미세조정.
4. **간격 최적화** — fine-tune된 모델을 동결하고 간격만 최적화, 성능(PSLL)과
   소요시간을 측정해 직접 최적화(Opt4)·GA와 비교.

## 데이터 계약

| 항목 | 규약 |
|---|---|
| 입력 | 간격 d ∈ [2,5]³¹ µm (정규화 (d−3.5)/1.5), 위상 φ ∈ ℝ³² rad ((cos φ, sin φ) 인코딩) |
| 출력 | 정규화 복소장 E(u) = EF_amp(u)·AF(u)/N 의 (Re, Im), AF = Σₙ exp(j(k·xₙ·u − φₙ)) |
| 그리드 | θ-균등 1801점 (−90°..90°, 0.1°) — MODE 로그와 동일 격자, u = sin θ |
| 강도 | I(u) = Re² + Im² (Opt4 `OPAModel.intensity` 와 항등 — 회귀 테스트로 고정) |
| 오라클 | `Opt4_Joint.model.OPAModel` + `JointConfig` 기본값 (MODE-적합 EF) — 단일 진실원 |
| train | 스트리밍 (배치마다 신선 샘플 — 수식이 무한 오라클이므로 저장 불필요) |
| val | 시드 고정 4096 샘플 팩 (`data/val_pack.npz`) — 검증셋 = 수식 규약 |
| test | MODE varFDTD 로그 100쌍 (d=3 등간격, 위상 가변) — 테스트셋 = MODE 규약 |

## 파일 (챔피언 경로만)

| 파일 | 역할 |
|---|---|
| `opt_core.py` | **공용 코어 정본** — 설계 프로토콜, 배치 필드 조립, soft/hard PSLL, 커플링 페널티, 경사 탐색, 다양성 선택, L-BFGS 연마, 동결 로더 |
| `config.py` / `physics_oracle.py` | 설정 / Opt4 수식 브리지 (물리 상수·EF 단일 진실원) |
| `data_gen.py` / `model.py` / `train.py` | 모사 모델 학습 (스트리밍 샘플러 · MLP·element·siren·ffmlp · 사전학습) |
| **`champion_track1.py`** | **① 순수 최적화 챔피언** — 수식 Adam 30ep → 다양성 top-12 → L-BFGS |
| **`champion_tandem.py`** | **② 모사-최적화 챔피언** — tandem 역설계망 학습 + 사양별 추론·연마 |
| `evaluate_champions.py` | 4축 판정 (PSLL·ISL·HPBW·η·유지 + 시간/세대/평가횟수), `--mc N` 으로 오차 하 성능 병기 |
| `mode_validation.py` | MODE 검증 — `--crosscheck`(기존 로그로 EF 재검증) / `--spec`(랩 실행 명세) / `--check`(결과 대조) |
| `test_opt5.py` | 회귀 게이트 9종 (오라클 항등, 동결 보증, MPS 패리티 등) |
| `backlog/` | 미채택 실험 동결 보존 — `backlog/README.md` 참조 |

산출물 배치:

```
results/              챔피언 최종본 + 4축 판정 + 개선 로그
  training/           모사 학습 로그·벤치 (train_*.jsonl, bakeoff, infer_bench, finetune)
  experiments/        IMP1~6 기록과 비교 기준선 (GA·main Opt4·중간 예산 변형)
checkpoints/          모사 가중치 (git 미추적)
data/                 val 팩 (git 미추적)
```

## 실행

```bash
# 모사 모델 학습 (② 트랙 전제, 1회 33분)
python train.py --arch mlp --steps 20000 --device auto --tag mlp_full

# ① 챔피언: 수식 직접 (5.6s)
python champion_track1.py

# ② 챔피언: tandem (학습 27s + 사양당 0.32s)
python champion_tandem.py --ckpt checkpoints/mlp_full.pt

# 4축 판정
python evaluate_champions.py
```

## 결과 (2026-07-22, ver_fullRegression)

### 1) bake-off — 수식 근사 (val = 수식 홀드아웃 4096, MPS 학습)

| | MLP (20k, 7.46M) | NeuralElement (2k, 0.07M) |
|---|---|---|
| field RMSE | 0.01434 | 5.5e-7 |
| 강도 R² | 0.99695 | 1.00000 |
| PSLL 오차 | 0.216 dB (< 목표 0.3) | 6.5e-6 dB |
| 추론 (designs/s, B=1024) | CPU 71k / **MPS 216k (수식 38×)** | 8.3k / 11.6k |

판정: **주 서러게이트 = MLP** (속도), **검증기 = element** (수식 항등 —
off-manifold 감시·멀티-N 예비). `results/bakeoff_summary.json`.

### 2) MODE fine-tune — replay 변형 (test = MODE 로그, holdout 20프레임)

| | MODE R² pre→post (holdout) | 수식 PSLL 오차 pre→post |
|---|---|---|
| MLP | 0.9943 → **0.999998** | 0.216 → 0.242 dB |
| element | 0.99642 → 0.99642 | 5e-6 → 0.010 dB |

수식이 못 잡던 잔차 0.36%를 MLP가 흡수 (element는 물리 제약상 불가).
단, holdout = 동일 궤적·d=3 상관 프레임 → φ-보간 성능이지 간격 일반화 증거 아님.

### 3) 간격 최적화 벤치 — 동일예산 128k 패턴평가, 심판 = 수식 hard PSLL
worst-angle (±15° 5각 프로토콜, 커플링 페널티 동일). `results/benchmark_ftmlp.json`.

| lane | PSLL(수식) | 시간 | 비고 |
|---|---|---|---|
| surrogate 단독 (MPS 64-multistart) | −10.47 dB | 2.1s | 자기평가 −13.65 → **gap 3.18 dB = off-manifold 악용 실증** |
| **hybrid (탐색→top-8 수식 연마)** | **−12.65 dB** | **3.5s** | 수식 평가 4.3k회(예산 3.4%)만 사용 — **권장 경로** |
| 수식 직접 (CPU f64 배치 Adam) | −13.26 dB | 38.1s | 품질 상한 |
| GA (동예산) | −11.17 dB | 24.4s | 대체 대상 기준선 |

핵심: **hybrid = GA 대비 품질 +1.5 dB·시간 1/7, 수식 직접 대비 0.62 dB 양보에 11× 단축.**
순수 서러게이트 최적화는 모델 오차 봉우리를 파고들어(gap 3.18 dB) 품질 게이트 탈락 —
top-K 수식 연마가 필수 안전장치.

### 4) 개선 실험 IMP1~6 (`results/improvements_log.md` 상세)

판정 회계: **J = hard PSLL + 커플링 페널티** (raw PSLL 단독은 마진을 팔아 점수를 사는
설계를 걸러내지 못함 — IMP5 에서 실증). ±15° 프로토콜 기준:

| 방식 | PSLL | ISL | J | 시간 | 세대 |
|---|---|---|---|---|---|
| GA (대체 대상) | −11.17 | 2.33 | −10.58 | 24.4s | 400 |
| **① 챔피언** (Adam 30ep + L-BFGS + ISL항) | −13.35 | **0.56** | −12.94 | 6.31s | 30 |
| **② 챔피언** (tandem + 단일 연마 + ISL항) | **−13.55** | 0.77 | −12.98 | **0.38s** | **0** |

두 챔피언 모두 GA를 4축 전 항목에서 지배한다. ①은 선행 투자 없이 6.3초,
②는 학습 27.7초를 선지급하고 사양당 0.38초 — 사양 2건 이상이면 ②가 유리하다.
목적함수에는 PSLL(피크)과 ISL(총에너지)이 함께 들어간다 (`--w-isl`).

불채택 (기록 보존): IMP2 앙상블 불일치 페널티(이득 대비 3× 비용), IMP3 SIREN(도메인
오용으로 붕괴)·ffmlp(모사 정확도 −19%가 champion 품질로 전이되지 않음), IMP5 구조적
초기해(로짓 포화로 경계 유착).

### 5) 성능 한계와 검증 상태

**−13.4 dB 는 알고리즘이 아니라 하드웨어의 벽이다.** 위상 자유도를 열어도 +0.02 dB,
진폭까지 열어도 +0.33 dB뿐이고, 설계각 사이 성능 저하도 없다 (0°→±15° 단조).
여러 알고리즘이 같은 값에 수렴한 이유가 이것이며, 알고리즘 추가 투자는 수익이 없다.

**실제 조건에서는 공칭보다 나쁘다.** 위치 σ=50nm·위상 σ=5° 오차 하 p90 은
① 챔피언 −11.18, ② tandem −11.40, GA −9.86 — 격차는 유지되나 절대값은 1.5 dB 손실한다.
로버스트 최적화(`--robust K`)를 넣었으나 이득이 p90 +0.13 dB 에 시간 10배라 기본값은
아니다. tandem 은 명시적 로버스트 없이도 가장 강건하다.

**남은 미검증은 하나** — EF 모델이 d=3µm 등간격 로그로 적합됐으므로 비등간격 배열에서의
정확도는 varFDTD 신규 실행 전까지 미확인이다. 다만 EF 를 크게 흔들어도(보정 제거 포함)
챔피언 > GA 순위는 유지되며, 기존 로그의 소자 수 외삽(N=64/128 홀드아웃)은 R² 0.996 이다.
`mode_validation.py --spec` 이 랩 실행 명세를, `--check` 가 합격 판정을 제공한다.

## 한계 (정직 고지)

MODE 로그는 d=3 등간격뿐 → fine-tune 이 검증하는 것은 위상·EF성 불일치 보정이며,
간격 의존 보정의 일반화는 단일 MODE 신규 실행 전까지 미검증.
벤치 lane 간 장치 상이(서러게이트 MPS f32 / 수식·GA CPU f64)는 현행 생산 경로
그대로의 비교이며, 수식 MPS 이식 시 격차 축소 여지 있음.
