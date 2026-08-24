# GateMol-BBB 최적화 전략 — 학습 단계 calibration (Path 1, 평가 잣대 불변)

**작성일**: 2026-05-18
**전제 문서**:
- [`comparison_pre_optimization_analysis.md`](comparison_pre_optimization_analysis.md)
- [`calibration_analysis_combo3.json`](calibration_analysis_combo3.json)

---

## 1. 원칙

본 전략은 **GateMol을 학습 단계에서 더 잘 calibrate되도록 만드는 변경**에만 한정한다. Post-hoc threshold tuning, temperature/Platt/isotonic, cross-combo ensemble 등 **평가 잣대를 바꾸는 후처리는 제외**한다.

이유:
- Baseline(MLP2/3/4·XGBoost·LightGBM)은 frozen reference. 후처리를 GateMol에만 적용하면 비교가 무효해지고, baseline에도 동일 적용하면 잣대 자체가 바뀜.
- **MCC@0.5라는 잣대를 그대로 두고도** GateMol이 우위를 가져갈 수 있는지를 검증하는 것이 본 전략의 목적.

확정:
- 평가 잣대 — holdout 5 subset(`internal`/`external`/`nn03`/`nn05`/`total`) × {ROC-AUC, MCC, F1, ACC} @threshold=0.5, soft voting ensemble. **변경 없음**.
- Baseline 결과 — `results/comparison_pre_optimization.csv` 그대로 frozen reference.
- GateMol 학습 protocol — 본 문서 §3의 변경만 허용.

---

## 2. 현황 요약

| 지표 | Validation (per-seed mean) | Holdout `total` (ensemble) |
|---|---|---|
| ROC-AUC | GateMol 0.846 > LightGBM 0.816 | GateMol 0.886 > LightGBM 0.871 |
| MCC | GateMol 0.515 > LightGBM 0.481 | **GateMol 0.599 < LightGBM 0.629** |

진단 (combo3 holdout `total`, n=888):
- **Overconfidence**: GateMol 예측의 65.1%가 prob>0.9 또는 prob<0.1 (LightGBM 52.7%). 특히 prob<0.1이 LightGBM의 2배 (12.5% vs 6.3%) → "confident wrong" 예측이 영구 손실.
- **Ensemble lift 부족**: val→holdout MCC 이득이 GateMol +0.084 vs LightGBM +0.149 (≈56%).

→ 두 원인 모두 학습 단계 변경으로 직접 공략 가능.

---

## 3. 학습-단계 변경 카탈로그

`train.py` editable scope 내에서 적용. 추론 후처리·calibration set fitting 없음 → 평가 잣대 불변.

| ID | 변경 | 위치 | overconfidence | diversity | val_auc 영향 | 우선순위 |
|---|---|---|---|---|---|---|
| A1 | **Label smoothing** (target 0/1 → 0.05/0.95) | `train_model` BCE target | ↓↓ | – | 미세 ↓ | ★ |
| A4 | **SWA** (마지막 25% epoch 가중치 평균) | `train_model` 후반부 | ↓ (flat minima) | – | flat or ↑ | ★ |
| A2 | **MC dropout @ inference** (N=20 forward 평균) | `eval_model` + holdout path | ↓ | ↑ | 무관 | ★ |
| A6 | **Learnable logit temperature** (학습되는 scalar T) | model class 출력부 | ↓ | – | 작음 | ◇ |
| A3 | **Stronger regularization** (dropout 0.2→0.3, stochastic depth) | model classes | ↓ | ↑ | 살짝 ↓ 위험 | ◇ |
| A5 | **Multi-head** (1 backbone, K heads 평균) | `MultiModalGMLPFromFlat` | – | ↑↑ | 잘 짜면 ↑ | △ |

**Path 1 적합성 확인**:
- A1/A3/A4/A5 — 학습 신호 또는 구조 변경. 추론 후처리 없음. ✓
- A2 — 추론 시 dropout 활성화 후 같은 모델의 다중 forward 평균. **calibration set·threshold tuning 미사용**. ✓
- A6 — temperature가 **학습 중 backprop으로 학습되는 파라미터**. Post-hoc로 val에 fitting하는 게 아님. ✓

---

## 4. 2-Phase 운영 구조 (결정 사항)

본 전략은 **두 단계로 분리 운영**한다. Phase 1 도중에 criterion을 바꾸지 않고, iter100까지 현 규칙으로 완료한 뒤에야 Phase 2로 전환한다.

| Phase | 기간 | Keep criterion | 허용 변경 | 목적 |
|---|---|---|---|---|
| **Phase 1** | iter1 → iter100 (combo별 독립) | `mean_val_auc` (현행 그대로) | `train.py` 내 model classes / `train_model` / `eval_model` (현 `program.md` scope) | Ranking-only optimization의 한계점 확인 |
| **Phase 2** | iter101+ | `mean_val_auc + mean_val_mcc` (composite, α=1) | Phase 1과 동일 + 본 문서 §3의 A1/A4/A2 누적 적용 | Calibration-aware 변경 도입, holdout MCC 격차 메우기 |

이유:
- Phase 1 도중 criterion을 바꾸면 search log의 비교 일관성 깨짐.
- Dry-run으로 확인됨: composite criterion 적용해도 현 best(combo1 iter17 / combo2 iter17 / combo3 iter8)는 유지 → Phase 1 완료 후 전환해도 출발점 손상 없음.
- Phase 2는 Phase 1 종료점 위에 누적되므로, Phase 1의 ranking-only ceiling이 baseline 역할.

### 4.1 현재 combo별 진행 상태 (snapshot @ 2026-05-18)

| combo | 디렉토리 | 마지막 iter | 최근 KEEP | Phase 1 잔여 | Phase 2 시작 iter |
|---|---|---|---|---|---|
| combo1 (`maccs+avalon+scage2+mole`) | `/home/minji/combos-v2-combo1` | iter19 | iter17 (EMA decay=0.999) | **81 iter** | iter101 |
| combo2 (`maccs+scage1+mole`) | `/home/minji/combos-v2-combo2` | iter17 | iter17 (EMA decay=0.999) | **83 iter** | iter101 |
| combo3 (`maccs+scage1+scage2+mole`) | `/home/minji/combos-v2-combo3` | iter8 | iter8 (cross-modal FiLM) | **92 iter** | iter101 |

3 combo는 **독립 search loop**. GPU 여유 시 병렬 실행 가능. combo별 도달 시점이 다르므로 Phase 2 전환 시점도 combo별로 다를 수 있음(전 combo가 iter100에 도달한 뒤 동시 전환 권장).

관찰: combo1·combo2의 마지막 KEEP은 둘 다 "EMA weights" (= 가중치 평균화). 이는 Path 1의 A4 SWA와 같은 계열의 calibration-친화 변경이라, search가 `val_auc only` 기준 하에서도 부분적으로 calibration 방향을 찾아냈음을 시사. Phase 1 후반에 더 많은 calibration-친화 변경이 자연 누적될 가능성 있음.

### 4.2 Phase 2 criterion 패치 명세

`evaluate_combo.py`의 keep 기준을 composite으로 변경. 실제 위치는 약 6줄 (line 77~84의 `current_best_val_auc()` 함수와 line 180~181의 호출부).

```python
# 이전 (Phase 1, line 77-84 + 180-181)
def current_best_val_auc(combo: str) -> float:
    ...
    keepers = df[df["keep"] == True]
    if keepers.empty:
        return ...
    return float(keepers["mean_val_auc"].max())

best_so_far = current_best_val_auc(args.combo)
keep = bool(mean_auc > best_so_far)

# Phase 2 (composite)
def current_best_metric(combo: str) -> float:
    ...
    keepers = df[df["keep"] == True]
    if keepers.empty:
        return ...
    return float((keepers["mean_val_auc"] + keepers["mean_val_mcc"]).max())

best_so_far = current_best_metric(args.combo)
keep = bool((mean_auc + mean_mcc) > best_so_far)
```

근거:
- 두 지표 모두 **validation 단계** 산출 → holdout leakage 없음.
- Validation에서 GateMol은 두 지표가 동일 방향 → 새 criterion은 기존 행동과 baseline에선 일치하고, calibration 변경에만 차이를 만듦.
- α=1: AUC scale (~0.85)과 MCC scale (~0.5)이 유사해 한쪽이 sum을 압도하지 않음.

`evaluate_combo.py`는 `program.md`의 "Do NOT touch" 목록에 있으므로 이 패치는 **autoresearch in-loop 변경이 아니라 protocol amendment** — 사용자(=운영자) 권한으로 Phase 1 종료 시점에 한 번만 적용.

### 4.3 Dry-run 결과 (Phase 1 results 기준, 2026-05-18 시점)

기존 results.tsv에 composite metric을 사후 적용했을 때의 라벨 변동:

| combo | total iter | old KEEP | new KEEP | 플립 iter |
|---|---|---|---|---|
| combo1 | 19 | 4 | 2 | iter6 (modality dropout), iter12 (multi-head SGU n=2) — 둘 다 KEEP→DISCARD |
| combo2 | 17 | 4 | 4 | iter4 (label smoothing 0.1), iter6 (modality dropout) DISCARD→**KEEP** / iter7, iter10 KEEP→DISCARD |
| combo3 | 8 | 3 | 2 | iter7 (skip-gate) KEEP→DISCARD |

**핵심 발견**: combo2 iter4(label smoothing)는 old criterion으로 reject됐지만 composite으로는 KEEP. 즉 strategy doc이 가설한 "calibration 친화 변경이 ranking-only criterion 때문에 reject"되는 패턴이 search log에 실재함을 입증. Phase 2의 A1 priority가 정당화됨.

**Caveat**: 이 dry-run은 "같은 iter sequence를 새 criterion으로 라벨 재매기기"일 뿐 진정한 counterfactual은 아님. 만약 combo2 iter4가 실제 KEEP됐다면 iter5+가 다른 architecture 위에 쌓였을 것.

---

## 5. 실행 절차

### 5.1 Phase 1 — `val_auc only` 기준으로 iter100까지

```
[Pre-Phase 1] 변경 사항 없음. 현재 program.md / evaluate_combo.py 그대로.

[Phase 1 진행]
  3 워크트리 독립 실행. 각자 program.md의 iteration loop를 따른다:
    1. results.tsv 마지막 KEEP 확인
    2. architecture_ideas.md 참고하여 변경 선정
    3. train.py partial Edit
    4. Haiku 4.5 sanity check
    5. commit "iter<N>: <short>"
    6. evaluate_combo.py 실행 (val_auc only criterion)
    7. KEEP → final_holdout_eval.py 실행 / DISCARD → git revert

[Phase 1 종료 조건]
  각 combo의 iter 카운트가 100에 도달.

[Phase 1 운영 노트]
  - 진행 중 evaluate_combo.py / program.md 절대 불변.
  - composite metric은 results.tsv에 mean_val_mcc 컬럼으로 이미 기록됨 → 진행 중
    아무 시점에 dry-run 분석으로 "Phase 2에서 KEEP될 것 같은 iter" 정찰 가능.
  - architecture_ideas.md 고갈 위험 (특히 combo1·combo2는 iter80 무렵): 필요 시
    카탈로그 보강은 program.md scope 내라 허용.
  - 3 combo의 도달 시점이 다를 수 있음. 전 combo가 100 도달까지 기다린 뒤 Phase 2
    동시 전환.

[Phase 1 종료 시 안전장치]
  각 워크트리에서:
    git tag phase1-final
  Phase 2 실패 시 즉시 복원 가능.

[Phase 1 종료 산출물]
  - results/comparison_phase1.csv (현 comparison_pre_optimization.csv 갱신판)
  - results/optimization_strategy.md 에 "Phase 1 종료 보고" 섹션 추가
  - 각 combo의 Phase 1 final best vs baseline 비교 (MCC 격차 잔존 여부)
```

### 5.2 Phase 2 — composite criterion 전환 + calibration-aware 변경

```
[Step 0 — Protocol amendment]
  3 워크트리에 동시에 single commit으로 다음을 적용:
    · evaluate_combo.py: §4.2 패치
    · program.md line 21-22: "mean_val_auc" → "mean_val_auc + mean_val_mcc"
    · 본 문서에 "Phase 2 시작" 시점 명시
  commit msg: "protocol: switch keep criterion to composite (start Phase 2)"
  ※ 학습 코드 변경 0건. 새 iter는 아직 없음.

[Step 1 — A1: label smoothing] (iter101)
  train.py: BCEWithLogitsLoss → target을 0.05/0.95로 smooth.
  3 워크트리 독립 실행. composite val metric으로 KEEP 판정.
  KEEP 시 holdout 5 subset 평가 (잣대 MCC@0.5 그대로).

[Step 2 — A4: SWA 누적] (iter102)
  Step 1 결과를 보고 진행. combo별로 다를 수 있음:
   - Step 1이 KEEP된 combo: KEEP 상태 위에 SWA 추가
   - DISCARD된 combo: 직전 KEEP iter 위에 SWA 단독 시도
  ※ combo1·combo2는 Phase 1에서 EMA(=SWA 친척)를 이미 보유하므로 추가 효과 작을
    가능성. combo3는 EMA가 Phase 1에 들어와 있을지 100 iter 도달 시점에 결정됨.

[Step 3 — A2: MC dropout 누적] (iter103)
  Inference path만 수정 (eval_model + holdout). N=20.

[Step 4 — A6/A3/A5: 잔여 격차 대응]
  Step 3까지 진행해도 holdout total MCC가 LightGBM 0.629 미달인 combo만
  A6 → A3 → A5 순으로 시도.

[검증]
  Baseline 결과(comparison_pre_optimization.csv)는 frozen reference.
  각 Step 종료 후 comparison_post_optimization.csv 갱신.
  동일 protocol, 동일 잣대(MCC@0.5).
  표에 각 combo의 "마지막 적용 Step"과 "Phase 2 누적 iter 수"를 함께 기록.
```

### 5.3 병렬 실행 가이드

GPU 메모리·feature_cache IO 여유 시 3 워크트리 병렬 실행 가능:

```bash
( cd /home/minji/combos-v2-combo1 && [autoresearch loop for next iter] ) &
( cd /home/minji/combos-v2-combo2 && [autoresearch loop for next iter] ) &
( cd /home/minji/combos-v2-combo3 && [autoresearch loop for next iter] ) &
wait
```

각 워크트리는 독립 git 브랜치 + 독립 results/ 디렉토리이므로 충돌 없음. 공유 자원은 `pool.npz` / `feature_cache_holdouts/` (read-only).

---

## 6. 성공 기준

### 6.1 Phase 1 종료 시점의 진단 지표 (정량 요건 아님)

Phase 1은 "ranking-only ceiling 측정" 단계. 다음을 기록·보고:
- 3 combo 각각의 Phase 1 best (iter ≤100 중 최고 mean_val_auc 모델)의 holdout total ROC-AUC / MCC
- LightGBM과의 MCC 격차 (3 combo 평균)
- Phase 1 후반(iter80~100) keep rate

Phase 2 진입 여부는 진단 결과와 무관하게 진행. Phase 1은 "비교 baseline 확보"가 목적.

### 6.2 Phase 2 최종 성공 기준 (3 combo 동시 만족)

1. **Holdout `total` MCC**: GateMol ≥ 0.629 (현재 LightGBM)
2. **Holdout `total` ROC-AUC**: GateMol ≥ Phase 1 best ROC-AUC (떨어지지 않음)
3. **Holdout `external`/`nn05` MCC**: GateMol이 모든 baseline 이상
4. **Validation AUC/MCC**: Phase 1 best 대비 ±0.005 이내 유지

(1)+(2)를 한 combo라도 미달하면 §3의 후순위 항목(A6/A3/A5) 추가 또는 전략 재검토.

---

## 7. 본 전략에서 제외된 항목 (Path 2, 참고)

다음은 평가 잣대를 바꾸므로 **본 전략에서 제외**. 적용한다면 baseline 5 모델 전부에 동일 절차로 적용한 별도 부록 표(`comparison_posthoc_calibrated.csv`)로 분리 보고해야 함.

- Val 기반 threshold tuning (per-model best_t)
- Temperature scaling (post-training fit on val)
- Platt / isotonic regression
- Cross-combo ensemble (3 combo 모델 확률 평균)

---

## 8. 사용자 결정 사항

- **결정 8.1** — 2-phase 운영 구조 동의 여부. 권장: 적용 (방법론 일관성 + 출발점 안정).
- **결정 8.2** — Phase 1 종료 기준 iter100 동의 여부. 대안: "20 consecutive DISCARD" adaptive stop.
- **결정 8.3** — Phase 2 우선순위 (A1 → A4 → A2 → A6/A3/A5) 동의 여부.
- **결정 8.4** — Phase 2 성공 기준 §6.2 동의 여부.

---

## 9. 한 문단 요약

> **GateMol-BBB의 holdout MCC 약점은 sigmoid 출력의 overconfidence와 10-seed ensemble의 diversity 부족에서 온다. 평가 잣대(MCC@0.5)와 baseline 결과를 frozen으로 두고 GateMol 학습 단계만 수정하는 Path 1 전략은 두 단계로 운영된다. Phase 1은 현재 `val_auc only` criterion을 그대로 두고 3 워크트리(`combos-v2-combo{1,2,3}`)에서 iter100까지 ranking-only optimization을 완주하여 "Phase 1 best"를 baseline으로 확정한다. Phase 2는 `evaluate_combo.py`와 `program.md`에 single commit으로 protocol amendment를 적용해 keep 기준을 `mean_val_auc + mean_val_mcc` composite으로 전환하고, iter101부터 label smoothing → SWA → MC dropout을 우선순위대로 누적 적용하여 calibration-aware 변경이 search에 진입할 수 있게 한다. Dry-run으로 confirm된 바, 새 criterion 적용은 현재 best(iter17/iter17/iter8)를 무효화하지 않으면서 combo2 iter4(label smoothing) 같은 calibration 친화 변경을 admits함. Post-hoc threshold·temperature·cross-combo ensemble은 잣대를 바꾸므로 본 전략에서 제외하며, 필요시 baseline에도 동등 적용한 별도 부록으로 분리한다.**

---

## 10. 산출물 위치

- 본 문서: `results/optimization_strategy.md`
- Frozen reference: `results/comparison_pre_optimization.csv`
- 비교 분석: `results/comparison_pre_optimization_analysis.md`
- Calibration 진단: `results/calibration_analysis_combo3.{json,npz,_hist.png,_threshold.png}`
- Phase 1 종료 시 추가될 산출물: `results/comparison_phase1.csv` + 본 문서 "Phase 1 종료 보고" 섹션
- Phase 2 종료 시 추가될 산출물: `results/comparison_post_optimization.csv`
- Baseline 실행: `baselines/run_baselines.py`
- Calibration 진단 스크립트: `baselines/calibration_analysis.py`
- 비교 표 생성: `baselines/build_comparison_table.py`
