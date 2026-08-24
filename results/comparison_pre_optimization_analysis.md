# Pre-optimization 비교: GateMol-BBB iter1 vs Baselines

**작성일**: 2026-05-17
**대상 combo**: `maccs+avalon+scage2+mole` / `maccs+scage1+mole` / `maccs+scage1+scage2+mole`
**비교 모델**: GateMol-BBB (autoresearch iter1 = baseline architecture reproduction) vs MLP2 / MLP3 / MLP4 / XGBoost / LightGBM
**Raw data**: [`comparison_pre_optimization.csv`](comparison_pre_optimization.csv)

---

## 1. Protocol

| 항목 | 설정 |
|---|---|
| Training pool | `/home/minji/feature_cache_merged/pool.npz` (internal_remaining ∪ external_remaining, deduped, **n=9786**) |
| Split | 8:2 train/val, scaffold-grouped, 10 seeds (42, 100, 200, …, 900) |
| Holdout subsets | 5종 (`internal`, `external`, `nn03`, `nn05`, `total`), simfilter≥0.9 |
| Validation 지표 | per-seed val_roc_auc / val_mcc → **10-seed mean ± std** |
| Holdout 지표 | 10 seed 모델의 sigmoid 확률 평균(**soft voting**) → ensemble ROC-AUC/MCC/F1/ACC |

GateMol-BBB는 `train.py` baseline architecture(iter1, "baseline reproduction"). Baseline 모델 코드와 하이퍼파라미터는 4월 비교 실험과 동일 (`baselines/models.py`).

### 1.1 데이터 분포 (positive 비율)

| 데이터 | n | pos_rate |
|---|---|---|
| Training pool | 9,786 | 69.4% |
| Holdout internal | 179 | **76.0%** |
| Holdout external | 709 | 71.8% |
| Holdout nn03 | 39 | 71.8% |
| Holdout nn05 | 329 | **78.7%** |
| Holdout total | 888 | 72.6% |

→ Holdout subset은 모두 학습 pool보다 positive 비율이 +3~+10 pp 더 높음 (분포 shift).

---

## 2. Validation 결과 (10-seed mean ± std)

### 2.1 Validation ROC-AUC

| combo | GateMol iter1 | mlp2 | mlp3 | mlp4 | xgboost | lightgbm |
|---|---|---|---|---|---|---|
| combo1 (maccs+avalon+scage2+mole) | **0.8443 ± 0.0042** | 0.8021 ± 0.0056 | 0.8043 ± 0.0085 | 0.8025 ± 0.0071 | 0.8026 ± 0.0029 | 0.8121 ± 0.0049 |
| combo2 (maccs+scage1+mole) | **0.8454 ± 0.0029** | 0.8088 ± 0.0060 | 0.8129 ± 0.0071 | 0.8122 ± 0.0078 | 0.8100 ± 0.0060 | 0.8183 ± 0.0050 |
| combo3 (maccs+scage1+scage2+mole) | **0.8464 ± 0.0047** | 0.8132 ± 0.0078 | 0.8129 ± 0.0077 | 0.8149 ± 0.0090 | 0.8124 ± 0.0069 | 0.8164 ± 0.0040 |

→ 3 combo 모두 GateMol이 가장 강한 baseline(lightgbm) 대비 **+2.7 ~ +3.2 %p** AUC 우위.

### 2.2 Validation MCC

| combo | GateMol iter1 | mlp2 | mlp3 | mlp4 | xgboost | lightgbm |
|---|---|---|---|---|---|---|
| combo1 | **0.5330 ± 0.0136** | 0.4397 | 0.4458 | 0.4326 | 0.4596 | 0.4685 |
| combo2 | **0.5181 ± 0.0157** | 0.4519 | 0.4479 | 0.4469 | 0.4595 | 0.4700 |
| combo3 | **0.5152 ± 0.0236** | 0.4525 | 0.4466 | 0.4477 | 0.4674 | 0.4805 |

→ 3 combo 모두 GateMol이 best baseline 대비 **+0.035 ~ +0.065** MCC 우위.
→ Validation 단계에선 ROC-AUC와 MCC가 같은 방향으로 일관되게 GateMol을 가리킴.

---

## 3. Holdout 결과 (soft voting ensemble)

### 3.1 Holdout ROC-AUC

#### combo1: `maccs+avalon+scage2+mole`

| model | internal | external | nn03 | nn05 | total |
|---|---|---|---|---|---|
| **GateMol-BBB iter1** | 0.8858 | **0.8804** | 0.8539 | **0.8595** | **0.8832** |
| mlp2 | 0.9012 | 0.8342 | 0.8214 | 0.8173 | 0.8511 |
| mlp3 | 0.9017 | 0.8385 | 0.8344 | 0.8197 | 0.8545 |
| mlp4 | **0.9032** | 0.8367 | 0.8247 | 0.8179 | 0.8535 |
| xgboost | 0.8806 | 0.8646 | 0.8734 | 0.8503 | 0.8695 |
| lightgbm | 0.8870 | 0.8611 | **0.8896** | 0.8484 | 0.8684 |

#### combo2: `maccs+scage1+mole`

| model | internal | external | nn03 | nn05 | total |
|---|---|---|---|---|---|
| **GateMol-BBB iter1** | 0.9015 | **0.8777** | **0.8799** | **0.8723** | **0.8839** |
| mlp2 | 0.9010 | 0.8461 | 0.8442 | 0.8309 | 0.8601 |
| mlp3 | **0.9034** | 0.8498 | 0.8214 | 0.8305 | 0.8631 |
| mlp4 | 0.8998 | 0.8473 | 0.8442 | 0.8324 | 0.8610 |
| xgboost | 0.8955 | 0.8690 | 0.8636 | 0.8648 | 0.8764 |
| lightgbm | 0.9001 | 0.8622 | 0.8766 | 0.8602 | 0.8723 |

#### combo3: `maccs+scage1+scage2+mole`

| model | internal | external | nn03 | nn05 | total |
|---|---|---|---|---|---|
| **GateMol-BBB iter1** | 0.8965 | **0.8818** | **0.8766** | **0.8628** | **0.8857** |
| mlp2 | 0.8995 | 0.8483 | 0.8377 | 0.8294 | 0.8614 |
| mlp3 | 0.9027 | 0.8481 | 0.8344 | 0.8285 | 0.8622 |
| mlp4 | **0.9030** | 0.8506 | 0.8344 | 0.8270 | 0.8639 |
| xgboost | 0.8875 | 0.8678 | 0.8604 | 0.8636 | 0.8738 |
| lightgbm | 0.8945 | 0.8616 | 0.8604 | 0.8499 | 0.8706 |

### 3.2 Holdout MCC

#### combo1

| model | internal | external | nn03 | nn05 | total |
|---|---|---|---|---|---|
| GateMol-BBB iter1 | 0.7140 | 0.5879 | 0.6759 | 0.5666 | 0.6118 |
| mlp2 | 0.7274 | 0.5491 | 0.6803 | 0.5185 | 0.5830 |
| mlp3 | 0.7119 | 0.5634 | **0.7462** | 0.5366 | 0.5916 |
| mlp4 | **0.7448** | 0.5741 | 0.6803 | 0.5425 | 0.6063 |
| xgboost | 0.6757 | **0.6182** | 0.5224 | **0.5928** | 0.6292 |
| lightgbm | 0.7103 | 0.6140 | 0.6803 | 0.5696 | **0.6322** |

#### combo2

| model | internal | external | nn03 | nn05 | total |
|---|---|---|---|---|---|
| GateMol-BBB iter1 | 0.7140 | 0.5779 | **0.6933** | 0.5741 | 0.6036 |
| mlp2 | **0.7443** | 0.5515 | 0.5283 | 0.5234 | 0.5875 |
| mlp3 | 0.7119 | 0.5256 | 0.5283 | 0.5115 | 0.5610 |
| mlp4 | 0.7106 | 0.5136 | 0.5283 | 0.4836 | 0.5508 |
| xgboost | 0.6932 | **0.6107** | 0.6118 | 0.5789 | **0.6260** |
| lightgbm | 0.7274 | 0.5939 | 0.6803 | **0.5909** | 0.6192 |

#### combo3

| model | internal | external | nn03 | nn05 | total |
|---|---|---|---|---|---|
| GateMol-BBB iter1 | 0.6761 | 0.5811 | 0.6201 | 0.5254 | 0.5991 |
| mlp2 | 0.6945 | 0.5306 | 0.5283 | 0.4836 | 0.5615 |
| mlp3 | 0.7106 | 0.5405 | 0.5977 | 0.5399 | 0.5726 |
| mlp4 | **0.7278** | 0.5158 | 0.5283 | 0.4995 | 0.5561 |
| xgboost | 0.7103 | 0.6022 | **0.7462** | **0.5812** | 0.6226 |
| lightgbm | 0.7103 | **0.6100** | 0.5977 | 0.5801 | **0.6290** |

### 3.3 Column-wise best (best model in each holdout subset)

| subset | combo1 | combo2 | combo3 |
|---|---|---|---|
| **ROC-AUC** | | | |
| internal | mlp4 (0.9032) | mlp3 (0.9034) | mlp4 (0.9030) |
| external | **GateMol (0.8804)** | **GateMol (0.8777)** | **GateMol (0.8818)** |
| nn03 | lightgbm (0.8896) | **GateMol (0.8799)** | **GateMol (0.8766)** |
| nn05 | **GateMol (0.8595)** | **GateMol (0.8723)** | **GateMol (0.8628)** |
| total | **GateMol (0.8832)** | **GateMol (0.8839)** | **GateMol (0.8857)** |
| **MCC** | | | |
| internal | mlp4 (0.7448) | mlp2 (0.7443) | mlp4 (0.7278) |
| external | xgboost (0.6182) | xgboost (0.6107) | lightgbm (0.6100) |
| nn03 | mlp3 (0.7462) | **GateMol (0.6933)** | xgboost (0.7462) |
| nn05 | xgboost (0.5928) | lightgbm (0.5909) | xgboost (0.5812) |
| total | lightgbm (0.6322) | xgboost (0.6260) | lightgbm (0.6290) |

---

## 4. 분석

### 4.1 핵심 관찰

| 지표 | Validation | Holdout |
|---|---|---|
| ROC-AUC | GateMol 3 combo 모두 우세 (+2.7~3.2 pp) | GateMol 3 combo 모두 `total`/`external`/`nn05` 우세, `internal`에선 MLP에 약간 밀림 |
| MCC | GateMol 3 combo 모두 우세 (+0.035~0.065) | **혼재** — `internal`은 MLP, `total`/`external`/`nn05`는 tree 모델이 우세 |

→ Validation에선 두 지표가 같은 방향이지만, **holdout에선 ROC-AUC와 MCC가 갈리는 현상**이 발생.

### 4.2 원인: 네 가지 요인의 결합

#### 요인 1 — Validation MCC는 ensemble이 아닌 per-seed 평균

- Validation MCC = `mean([per-seed MCC])` (단일 모델 능력의 평균)
- Holdout MCC = soft-voted 확률에 threshold 0.5 적용 (10-seed ensemble 보정 효과 포함)

→ Validation은 ensemble 보정 없음, holdout은 보정 있음. 두 단계의 비교 기준이 다름.

#### 요인 2 — Ensemble diversity 차이

`val → holdout total` MCC 변화량 (모델별, combo3 기준):

| model | val MCC | holdout total MCC | Δ |
|---|---|---|---|
| GateMol iter1 | 0.5152 | 0.5991 | +0.084 |
| mlp4 | 0.4477 | 0.5561 | +0.108 |
| xgboost | 0.4674 | 0.6226 | **+0.155** |
| lightgbm | 0.4805 | 0.6290 | **+0.149** |

→ Tree 모델은 ensemble으로 MCC가 GateMol 대비 2배 가까이 더 점프.
→ GateMol(d_model=512, depth=4 gMLP)은 10 seed가 비슷한 해로 수렴 → **model diversity 낮음** → ensemble 보정 효과 약함.
→ 반대로 tree 모델은 random subsample(0.8)/colsample(0.8) + bootstrap 효과로 10 seed 사이 변동이 더 큼 → soft voting이 더 효과적.

#### 요인 3 — Calibration sharpness

- **GateMol-BBB** (multi-modal gMLP, 표현력 큼): 확률을 0/1로 극단화하는 경향 (overconfident).
- **Tree 모델**: leaf의 empirical frequency 기반 → 확률이 base rate 근처에 머무름 (smooth).
- **ROC-AUC**: ranking만 보므로 sharpness 무관 → GateMol 우위.
- **MCC@0.5**: 절대 threshold 의존 → 확률 분포가 살짝 shift하면 sharp distribution이 더 큰 페널티를 받음.

#### 요인 4 — Train/Holdout 분포 shift

- Pool/val: 69.4% pos
- Holdout subset: 72~79% pos (특히 nn05 78.7%, internal 76.0%)

→ Threshold 0.5는 holdout 분포에선 약간 suboptimal (분포가 positive 쪽으로 치우쳐서 최적 threshold가 더 낮음).
→ Sharp model(GateMol)이 wrong인 경우 confidence가 0.99처럼 극단적이라 threshold 변경으로도 살릴 수 없음 → MCC 손실.
→ Smooth model(tree)은 wrong인 경우라도 확률이 ~0.5 근처라 손실이 작음.

#### 요인 5 (보정) — pos_weight는 본 비교의 차이 요인이 **아님**

`baselines/models.py:80`의 baseline MLP:
```python
pw = torch.tensor([max(n_neg / n_pos, 1.0)], ...)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
```

학습 pool은 n_neg/n_pos = 2995/6791 = **0.44 < 1.0** → `max(0.44, 1.0) = 1.0` → effective `pos_weight=1.0` = pos_weight 없음과 수학적으로 동일.
GateMol-iter1도 `train.py:328`에서 `nn.BCEWithLogitsLoss()` (pos_weight 없음). 즉 **MLP baseline과 GateMol은 같은 effective loss**.
→ 본 비교에서 calibration 차이는 pos_weight 때문이 아니라 **architecture sharpness + ensemble diversity**에서 발생.

### 4.3 종합 해석

> **GateMol-BBB는 ranking(ROC-AUC)에선 일관되게 우수하지만, threshold@0.5 기반 절대 분류(MCC)에선 분포 shift와 ensemble diversity 부족 때문에 baseline tree 모델에 밀리는 경향**이 holdout에서 나타남.

이는 architecture를 깎는 autoresearch만으로는 자동 해결되지 않는, **calibration / ensemble strategy** 차원의 문제. ROC-AUC와 MCC가 같은 방향으로 움직이게 하려면 architectural search 외 다른 손잡이가 필요함.

---

## 5. 후속 분석 / 잠재적 개선

| 옵션 | 효과 예상 | 비고 |
|---|---|---|
| **Threshold tuning on val** | MCC 직접 개선 (분포 shift 대응) | 학습 안 건드림, post-hoc만. `total` subset에서 +0.02~0.04 추가 기대. |
| **Temperature / Platt scaling** | confidence 부드럽게 → MCC 안정 | 별도 calibration set 필요. |
| **`BASE_CONFIG`에 pos_weight=auto 추가** | pool 이 70% pos인데 굳이 1.0 유지 → 음성 쪽 weight 키워 calibration 보정 | program.md상 BASE_CONFIG 변경은 frozen 영역. 새 protocol 결정사항. |
| **다양한 architecture로 ensemble** | seed diversity 부족 보완 | autoresearch 산출물 여러 개를 같이 voting |
| **Mixup / label smoothing** | overconfidence 억제 | architectural search 결과에 plug-in 가능 |

### 5.1 정량 검증 가능한 후속 작업

1. **확률 히스토그램** — GateMol vs lightgbm의 holdout `total` ensemble probability 분포 비교 → sharpness 차이 시각화
2. **Threshold sweep** — GateMol holdout MCC를 threshold 0.3~0.7로 sweep → 최적 threshold가 0.5보다 낮음을 정량 확인 → 분포 shift 가설 검증
3. **Per-seed MCC vs ensemble MCC** — 단일 seed MCC 평균과 ensemble MCC 차이 정량 비교 → ensemble lift 정확히 분리

---

## 6. 산출물 위치

- `results/comparison_pre_optimization.csv` — 18 rows × 28 cols, ROC-AUC/MCC/F1/Accuracy 풀 데이터
- `results/<combo>/baselines/summary.tsv` — combo별 baseline 5 모델 raw 결과
- `results/<combo>/baselines/<model>.json` — 모델별 per-seed 디테일 (10 seed × 5 subset 메트릭)
- `/home/minji/combos-v2-combo<N>/results/<combo>/holdout_eval/iter0001_*.json` — GateMol iter1 holdout 원본
- `baselines/run_baselines.py` — baseline 실행 스크립트
- `baselines/build_comparison_table.py` — 통합 표 생성 스크립트

## 7. 재현 방법

```bash
# 1) Baseline 5 모델 학습 + 5 holdout subset 평가 (combo당 ~5분)
cd /home/minji/autoresearch_combos_v2
conda run -n rapids-25.02 python baselines/run_baselines.py --combo maccs+avalon+scage2+mole
conda run -n rapids-25.02 python baselines/run_baselines.py --combo maccs+scage1+mole
conda run -n rapids-25.02 python baselines/run_baselines.py --combo maccs+scage1+scage2+mole

# 2) GateMol iter1 + baselines 통합 표 생성
conda run -n rapids-25.02 python baselines/build_comparison_table.py
# → results/comparison_pre_optimization.csv
```
