# Ultra-fast Phase Calibration for an Optical Phased Array

광위상배열(OPA)의 초기 위상 오차를 적은 반복으로 보정하는 연구 코드다.
Photonics Conference 2022 포스터([`PC2022_poster_DH_v5_final.pdf`](PC2022_poster_DH_v5_final.pdf))에서
발표한 Adam 기반 위상 캘리브레이션이 출발점이고, 이후 같은 질문을 간격 설계와 서러게이트 모델로
넓힌 후속 연구가 함께 들어 있다.

> Do-Hyung Kim, Dong-Hwan Kim, Ji-Yeong Gwon, Sang-Shin Lee,
> "Ultra-fast Phase Calibration for an Optical Phased Array," Photonics Conference 2022 (poster).
> 광운대학교 전자공학과 Photonics Research Lab · 고려대학교 컴퓨터정보학과.
> DOI [10.13140/RG.2.2.33027.17445](https://www.researchgate.net/publication/366086411_Ultra-fast_Phase_Calibration_for_an_Optical_Phased_Array)

## 포스터가 다룬 문제

**OPA**는 도파로 배열의 채널별 위상을 조절해 빔 방향을 바꾼다. 기계 구동부가 없어 빠르고 작지만,
제작 공정 편차 때문에 채널마다 **초기 위상 오차**가 생긴다. 이 오차는 무작위이고 예측할 수 없어서
빛이 흩어지고 원거리장(far-field)이 왜곡된다. 위상과 원거리장의 관계도 비선형이다. 그래서 실제로는
카메라로 원거리장을 보면서 위상을 반복 갱신하는 **피드백 캘리브레이션**을 쓴다.

기존 방식의 병목은 두 가지다.

- **카메라 프레임 속도** — 프레임 하나를 찍어야 한 번 갱신할 수 있다.
- **반복 횟수** — 언덕 오르기(HC)·SGD·유전 알고리즘(GA) 계열은 규모를 키우기 어렵다. 포스터의 비교표 기준으로
  SPGD는 20000회, DSGD는 7048회, REV는 4096회까지 필요했다.

포스터의 답은 **Adam**이다. 변수(채널 위상)마다 독립적인 적응 학습률과 모멘텀을 주어 국소 극값이
많은 지형에서도 적은 반복으로 수렴한다. 검증은 Lumerical MODE(varFDTD)를 Python API로 자동화한
루프로 했다. Python이 위상을 갱신하면 MODE가 실행되어 원거리장을 돌려준다.

| 설정 | 값 |
|---|---|
| 배열 | 1차원 주기 SiN 도파로, N = 32 / 64 / 128 |
| 파장 λ | 1.55 µm |
| 도파로 길이 · 폭 · 두께 | 20 · 1 · 0.5 µm |
| 채널 간격 d | 3 µm |
| 굴절률 (코어 SiN / 클래드 SiO₂ / 공기) | 1.97 / 1.44 / 1 |
| 초기 위상 오차 | 균등분포, 한 주기 (0, 2π) |
| 손실 | 주엽(가중 1) + 양측 grating lobe(가중 0.5) 강도의 역수 |

**결과:** 채널 수(N=32/64/128)와 관계없이 100회 반복 안에 주엽 강도가 오차 없는 이상 상태의
92% 이상에 도달했다.

## 코드 지도 — 포스터의 어느 부분인가

| 경로 | 포스터 대응 | 내용 |
|---|---|---|
| `Beamforming.m`, `Beamsteering.m`, `Functions/` | Calibration setup (카메라 ↔ OPA 칩 ↔ PC) | 랩 실칩 제어용 MATLAB. 보드와 시리얼로 통신해 DAC 위상·TEC 온도를 설정하고(`Send`, `PhaseSet`, `TempSet`), 화면 캡처로 카메라 영상을 받아(`ScreenShot`, `getSpot`) 비교표의 REV 방식(`REV_method`)을 수행 |
| `ScreenCapture/` | 〃 | MathWorks File Exchange의 화면 캡처 유틸리티 (서드파티, `license.txt`) |
| `Feb24/` | 〃 | REV 실행 산출물 (채널별 이미지, `result.mat`, 64채널 위상 `OPA.phase.mat`) |
| `MODE OPA base files-*.zip` | Simulation environment | MODE 기본 모델(.lms) — 주기 SiN 도파로 배열 |
| `Opt1_Adam_presented/` | Proposed algorithm (발표본) | 포스터에 실은 Adam 위상 최적화 (`Optimizer.py`), MODE 연동 (`Lumerical_Func.py`), Fraunhofer 회절 기반 MATLAB 모델 (`Periodic_OPA_with_MODE.m`) |
| `Opt2_Forward/` | Simulation method | 수식 기울기로 위상을 갱신하고, 매 반복 MODE varFDTD로 원거리장을 평가하는 순방향 루프 |
| `Opt3_Back_Forward/` | Simulation result | 역방향(MODE 초기 패턴을 수식으로 모방해 초기 오차를 추정) → 순방향(보정) 2단계. `Resultants/` 에 N=32/64/128 각 100회 반복의 위상·주엽 강도·원거리장 로그가 있다 |
| `Opt4_Joint/` | 후속 연구 ① | 간격 + 위상 공동 설계, 프레임 5장 캘리브레이션 — [README](Opt4_Joint/README.md) |
| `Opt5_Surrogate/` | 후속 연구 ② | 원거리장 수식 회귀 서러게이트, 초 단위 간격 설계 — [README](Opt5_Surrogate/README.md) |

레포의 이미지와 데이터는 모두 MODE 시뮬레이션이나 pyplot 산출물이며, 실칩 관측 데이터는 들어 있지 않다.

## 포스터에서 후속 연구로 — 같은 질문, 다른 변수

포스터는 "주어진 칩의 위상을 얼마나 빨리 맞출 수 있는가"를 물었다. 후속 코드는 그 질문에서 포스터가
고정해 둔 전제를 하나씩 풀었다.

| 포스터의 전제 | 남은 한계 | 후속 코드의 대응 |
|---|---|---|
| 보정 비용 = **반복 횟수** (Adam 100회) | 반복마다 카메라 프레임 1장이 필요하다 | **Opt4 `calibrate_fast`** — 스팟 하나가 아니라 프레임 전체 패턴에 오차 ε̂을 적합한다. 시뮬레이션 기준 랜덤 프로브 프레임 5장으로 주엽 98% 이상을 복원한다 (`test_opt4` [9], REV는 155장) |
| 변수 = **위상뿐** (간격 3 µm 고정) | d = 3 µm 등간격에서는 ±31° grating lobe가 물리적으로 남고, 위상으로는 없앨 수 없다. 포스터의 손실에도 grating lobe 항이 들어 있다 | **Opt4** — 역할을 나눈다. 위상은 주엽 형성·조향·캘리브레이션을, 간격은 제작 전 1회 비등간격 설계로 사이드로브 억제를 맡는다 |
| 평가 = **반복마다 MODE 1회 실행** | 간격까지 탐색하면 시뮬레이터 루프로는 감당하기 어렵다 | **Opt5** — 원거리장 수식을 신경망으로 회귀해 동결하고, 최적화기는 간격만 조작한다. ±15° 조향 기준 PSLL: GA −11.17 dB / 24.4 s, ① 수식 직접 −13.35 dB / 6.3 s, ② tandem −13.55 dB / 사양당 0.38 s |
| 반복 횟수 = **품질의 대가** (Adam 100회) | 간격 설계도 반복 수천 회에 품질이 시드 운에 좌우된다 | **Opt5 `research/`** — 정확한 수식 위에서 배치 Newton + 판정 목적 직접 연마. 판정 J(PSLL + 커플링) 8시드 평균 −13.552 를 **47~71회차**에 낸다. 다른 방법군은 857~10,317회차에 −13.04~−13.55 ([근거](Opt5_Surrogate/research/README.md)) |
| 물리 모델 = **varFDTD** | 해석식의 소자인자가 실제와 어긋난다 | MODE 로그(`Opt3_Back_Forward/Resultants`)로 소자인자 보정을 적합한다. N=32로 적합하고 N=64/128 홀드아웃에서 R² 0.996이다. 비등간격 배열 검증은 `Opt5_Surrogate/mode_validation.py --spec` 이 랩 실행 명세를 준다 |

## 실행

**Opt4 / Opt5** (CPU로 동작, Opt5 학습은 Apple MPS 선택): Python 3.10, `torch numpy pandas matplotlib pytest`

```bash
cd Opt4_Joint
python -m pytest test_opt4.py          # 회귀 게이트 10종
python main_design_multiangle.py       # 다각도 간격 설계
python main_calibration.py             # 간격 동결 + 위상 캘리브레이션

cd ../Opt5_Surrogate
python -m pytest test_opt5.py          # 회귀 게이트 9종
python champion_track1.py              # ① 수식 직접 간격 설계 (~6 s)
python train.py --arch mlp --device auto --tag mlp_full   # ② 전제: 서러게이트 학습 (~33 min)
python champion_tandem.py --ckpt checkpoints/mlp_full.pt
python evaluate_champions.py           # 4축 판정
```

**Opt1 ~ Opt3** (2022 원본): Windows, Lumerical MODE + `lumapi`, CUDA가 필요하다. API 경로와 장치가
코드에 하드코딩되어 있다(`Lumerical_Func.py`).

**MATLAB** (`Beamforming.m`, `Beamsteering.m`): 랩의 OPA 구동 보드(시리얼)와 카메라 화면이 필요하다.

## 한계

- 소자인자 보정은 d = 3 µm 등간격 MODE 로그로 적합했다. 비등간격 설계에서의 정확도는 varFDTD를 새로 실행하기 전까지 확인되지 않았다.
- 포스터와 후속 연구의 모든 수치는 시뮬레이션 기준이다. 제조·구동 오차(위치 σ 50 nm, 위상 σ 5°)를 넣으면 PSLL이 약 1.5 dB 나빠진다 (`Opt5_Surrogate` README).

## 감사의 글

본 연구는 한국연구재단(NRF) 지원(교육부 2018R1A6A1A03025242, 과학기술정보통신부 2020R1A2C3007007)으로 수행되었다.

라이선스: [MIT](LICENSE)
