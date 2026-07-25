# 실험 백로그 — 챔피언에 채택되지 않은 경로 (동결 보존)

챔피언 경로(`../champion_track1.py`, `../champion_tandem.py`)에 남지 않은 실험들을
**실행 당시 상태 그대로** 보존한다. 부정 결과도 재현 가능해야 다음 판단이 빨라진다.

## 규약

- 공용 로직의 정본은 `../opt_core.py` 다. 여기 파일들이 같은 함수를 자체 정의하고
  있어도 그것은 실험 당시의 사본이며, 새 작업에서 참조하지 말 것.
- 각 파일 상단에 부모 경로 부트스트랩이 삽입되어 있어 `backlog/` 안에서 바로 실행된다.
- 산출물(`results/`, `checkpoints/`)은 상위 디렉터리로 통일되어 저장된다.

## 목록

| 파일 | 실험 | 결과 | 미채택 사유 |
|---|---|---|---|
| `benchmark.py` | 4레인 동일예산 대결 (서러게이트/하이브리드/수식직접/GA) | 순수 ② gap 3.18 dB 실증 | 판정 도구로서의 역할 종료 — `../evaluate_champions.py` 가 대체 |
| `champion_design.py` | IMP1 L-BFGS 연마, IMP2 앙상블 불일치 페널티 | L-BFGS 채택(→opt_core), 앙상블 J −13.47 최고 | 앙상블은 학습 3×·시간 3× 대비 동시간 이득 +0.01 dB |
| `experiment_cmaes_explore.py` | IMP4 아일랜드 CMA-ES 탐색 | J −13.00 @ 6.77s — 단일 사양 품질 최선 | tandem 이 21× 빠르게 동급 달성. **품질 우선 사양에서는 여전히 유효** |
| `experiment_structured_init.py` | IMP5 밀도 테이퍼링·처프 초기해 | raw −13.66 이나 J −9.88 | 로짓 포화로 경계 유착 — 마진을 팔아 산 점수 |
| `experiment_phase_via_surrogate.py` | 등간격 위상 최적화 (요구 매트릭스 4번째 칸) | 주엽 전력 99.3~99.8% 회복 | 증빙 완료 — 챔피언 경로와 무관 |
| `finetune_mode.py` | MODE 로그 fine-tune (pure/replay) | MODE holdout R² 0.9943→0.999998 | 현 프레임(판정=수식)에서는 선택 단계. MODE 검증 재개 시 복귀 |
| `optimize_spacing.py` | 순수 ② (최적화 루프에 수식 호출 0회) | gap 3.18 dB 로 품질 게이트 탈락 | 연마 없는 순수 모사 최적화는 악용에 취약 |

## 되살리는 법

`experiment_cmaes_explore.py` 가 가장 복귀 가능성이 높다 (품질 우선 사양):

```bash
python backlog/experiment_cmaes_explore.py --ckpt checkpoints/mlp_full.pt --gens 250
```

SIREN(`arch=siren`)은 모델 정의만 `../model.py` 에 남아 있고 별도 실험 파일은 없다 —
w0=30 에서 발산, w0=5 에서도 R² 0.215 로 실패했다. 좌표망을 607차원 조건부 회귀에
적용한 도메인 오용이며, 재도전한다면 FiLM 조건화가 전제다.
