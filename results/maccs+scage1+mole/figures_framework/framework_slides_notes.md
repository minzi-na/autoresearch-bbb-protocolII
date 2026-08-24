# BBB AutoResearch Framework — combo2 (`maccs+scage1+mole`)

PPT 슬라이드 2장(`bbb_autoresearch_phase1_combo2.png`,
`bbb_autoresearch_phase2_combo2.png`) 의 본문 내용 + 검증 결과 + 원본
autoresearch(bbb-combo1) 와의 비교 정리. 슬라이드는
`make_framework_slides.py` 로 재생성 가능.

---

## 1. 슬라이드 1 — Phase 1 : 모델 아키텍쳐 최적화 (Structure search)

**부제**: `iter1 → iter200, best=iter198 (mean_val_AUC 0.8527 ± 0.0036)`

### 1.1 AutoResearch 외부 루프

코드 자체를 실험 단위로 바꾸고 결과가 좋으면 keep, 아니면 discard.

| Step | 내용 |
|---|---|
| Step 1. Baseline 고정 | d_model, d_ffn, depth, dropout, lr, batch_size, 10 seeds 고정 |
| Step 2. Agent가 아키텍쳐 변경 제안 | projection norm, SGU 구조, residual scaling, pooling, EMA 등 |
| Step 3. `train.py` 수정 후 `evaluate_combo.py` 실행 | 10-seed scaffold 학습 → keep 이면 `final_holdout_eval.py` (mandatory) |
| Step 4. 결과 파싱 | 주 판단지표 = `mean_val_auc` (scaffold) / 보조 reporting = 5-subset holdout |

**Keep**
- `git commit` 유지 (`results.tsv` keep=True)
- `final_holdout_eval.py` 실행 → 5-subset holdout 기록
- best architecture 갱신

**Discard**
- `git revert <commit>` (soft only)
- `git reset --hard` 금지 — discard 도 history 보존
- 다음 architectural change 로 이동

### 1.2 Step 3 안에서 실행되는 내부 학습 루프

| 단계 | 내용 |
|---|---|
| Input | feature = `maccs (167) + scage1 (2048) + mole (768)` → 일괄 z-score 후 concat |
| Forward / Backward | 예측값 → BCE loss → gradient → AdamW step (architecture 고정, weight 만 학습) |
| Validation / Early stopping | scaffold val loss/AUC patience 추적 → best state 선택, 10-seed 평균으로 architecture 평가 |

### 1.3 파일 표

| 파일 | 역할 |
|---|---|
| `prepare.py` | 데이터 로딩 / 피처 생성 / split / 평가 함수 / 상수 정의 (고정) |
| `train.py` | **수정 대상** — 모델 구조 + train_model/eval_model 변경 |
| `evaluate_combo.py` | 10-seed scaffold 학습 + iter 행 TSV append, keep 판정 |
| `final_holdout_eval.py` | keep 직후 mandatory, 5-subset holdout 평가 + JSON 저장 |
| `program.md` | Autoresearch agent 작동 문서 (loop rules) |
| `architecture_ideas.md` | 구조 변경 카탈로그 + priority 표 |
| `results.tsv` | iter 결과 누적 로그 (커밋과 1:1 매핑) |
| `results/<combo>/weights/iter<N>/` | keep 시 자동 저장 — best state per seed (.pt) |
| `results/<combo>/holdout_eval/iter<N>_<hash>.json` | 5-subset holdout 지표 JSON |

---

## 2. 슬라이드 2 — Phase 2 : Optuna 탐색 설계 최적화 (hyperparameter search)

**부제**: `Phase-1 best architecture frozen · 현재 iter1 완료
(d_model/d_ffn/depth 탐색 제외), keep 0개`

### 2.1 AutoResearch 외부 루프

| Step | 내용 |
|---|---|
| Step 1. Best architecture 고정 | Phase-1 best (iter198) 를 `best_train.py` 에 freeze, `build_and_train` 공개 |
| Step 2. Agent가 Optuna 탐색 설계 변경 제안 | search space, sampler (TPE/CMA-ES), pruner, objective 집계 방식 수정 |
| Step 3. `optuna_combo.py` 수정 후 `evaluate_hpo.py` 실행 | BUDGET 고정: `n_trials=50, top_k=3, search 3-seed × 30 epochs → confirm 10-seed` |
| Step 4. 결과 파싱 | 1차: 3-seed search mean / 최종: `best_confirm_mean_val_auc` (10-seed) |

**Keep**
- `best_confirm_mean_val_auc > threshold_mean + threshold_std`
- `results/<combo>/hpo/results.tsv` keep=True 행 추가
- best HPO config 갱신

**Discard**
- `git revert <commit>` (soft only)
- `git reset --hard` 금지 — 시도 history 그대로 보존
- 다음 탐색 설계 변경으로 이동

### 2.2 Step 3 안에서 실행되는 내부 학습 루프

| 단계 | 내용 |
|---|---|
| Input | 고정 architecture + trial-sampled hyperparameters (lr, wd, dropout, drop_path, mod_drop, head_dropout, batch, grad_clip) |
| Trial training | sampled hp 로 모델 훈련 (search: 3-seed × 30 epochs, pruner 현재 미설정 — 추후 추가 가능) |
| Trial objective | 3-seed scaffold validation ROC-AUC mean → TPESampler 가 다음 trial 제안 (study 50 trials) |
| Final reevaluation | top-3 trial → 10-seed scaffold confirm (`best_confirm_mean_val_auc`) + 5-subset holdout |

### 2.3 파일 표

| 파일 | 역할 |
|---|---|
| `prepare.py` | 고정 (Phase 1과 동일 데이터/스플릿) |
| `best_train.py` | 고정 backend — Phase-1 best (iter198) architecture frozen |
| `optuna_combo.py` | **수정 대상** — `suggest_config` / sampler / pruner / objective 정의 |
| `evaluate_hpo.py` | frozen runner — study 실행, top-3 confirm, BUDGET 고정 (`n_trials=50`) |
| `program_phase2.md` | Phase 2 agent 작동 문서 (Optuna loop rules) |
| `hpo/results.tsv` | iter 결과 누적 로그 (keep 판정 + threshold 추적, **현재 0 keeps**) |
| `hpo/study_*.json` | Optuna study 직렬화 + `top_k_confirm` rows |
| `hpo/trials_*.tsv` | trial 단위 raw 기록 (suggested hp + search `mean_val_auc`) |

---

## 3. 슬라이드와 실제 코드 정합성 검증 결과

| 항목 | 슬라이드 표기 | 실제 코드/git 상태 | 일치 |
|---|---|---|---|
| Phase 1 best iter / val_AUC | iter198 / 0.8527 ± 0.0036 | `results.tsv` iter198 keep=True 행 mean_val_auc=0.852661, std=0.003551 | OK |
| Phase 1 mandatory final_holdout_eval | "keep 시 mandatory" | `program.md` step 8 mandatory 명시 | OK |
| Phase 1 파일 set | `train.py / evaluate_combo.py / final_holdout_eval.py / ...` | 실제 존재 | OK |
| Phase 2 best_train.py frozen | "Phase-1 best (iter198) freeze" | `best_train.py` 존재, `optuna_combo.py` 가 `import best_train as bt` | OK |
| Phase 2 inner loop seeds | "search 3-seed × 30 epochs" | `SEARCH_SEEDS_DEFAULT=[42,100,200]`, `search_num_epochs=30` | OK |
| Phase 2 confirm | "top-3 → 10-seed scaffold confirm" | `top_k=3`, `CONFIRM_SEEDS_DEFAULT=10 seeds` | OK |
| Phase 2 `n_trials` | 50 (Step 3 본문) | `BUDGET["n_trials"]=50` | OK |
| Phase 2 진행 상태 | iter1 완료, keep 0개 | `git log` iter1 commit 1개 (`765e978`), `hpo/results.tsv` 부재 | OK |
| Phase 2 pruner | "현재 미설정 — 추후 추가 가능" | `optuna_combo.py`에 pruner 미설정, `program_phase2.md`에 "can add"로만 언급 | OK |

---

## 4. 원본 autoresearch (bbb-combo1) 와의 비교

### 4.1 구조적으로 **동일한** 부분 (autoresearch 본체 그대로)

| 요소 | bbb-combo1 (원본) | combos-v2-combo2 (현재) |
|---|---|---|
| 2-step 분할 | Phase 1 architecture-only → Phase 2 HPO-only | 동일 |
| Phase 1 agent 행위 | "train 파일 편집 → 평가 → keep/discard" | 동일 |
| Phase 2 agent 행위 | "optuna 파일 편집 → study → keep/discard" | 동일 |
| Phase 1 결정 기준 | 10-seed scaffold ROC-AUC mean | 동일 |
| Phase 2 search objective | 3-seed scaffold val ROC-AUC mean | 동일 |
| Phase 2 최종 검증 | 10-seed full reevaluation | 동일 |
| Phase 2 architecture frozen | `bbb_train.py` freeze 상태 | `best_train.py` 로 분리 freeze |
| Holdout 평가 위치 | reporting-only, keep 결정 비반영 | 동일 (subset 5종으로 세분화) |
| Agent 문서 양식 | `bbb_program.md` / `bbb_optuna_program.md` | `program.md` / `program_phase2.md` 동일 양식 |

### 4.2 달라진 부분 (전부 운영/인프라 측면)

| 측면 | bbb-combo1 | combos-v2 |
|---|---|---|
| Repo 격리 | 단일 repo 안에 combo별 파일들 | **worktree per combo + git branch 분리** |
| 파일 네이밍 | `bbb_*` 접두사, combo suffix in TSV | `bbb_` 제거, worktree 안에서는 단순 이름 |
| Phase 2 architecture 고정 방식 | `bbb_train.py` 그대로 freeze | `train.py` 와 별도로 `best_train.py` 신설 |
| Phase 2 keep threshold | strict `> best_mean` | `> best_mean + best_std` (std buffer 추가) |
| Holdout 평가 깊이 | 단일 holdout | **5 subset 분해** (internal/external/nn03/nn05/total) |
| Phase 1 holdout 실행 | 비-mandatory | **keep 마다 mandatory** (`final_holdout_eval.py`) |
| Iter 산출물 보존 | tsv + log 중심 | `weights/iter<N>/`, `holdout_eval/iter<N>_<hash>.json` 영구 보존 |
| Optuna runner 분리 | `bbb_optuna.py` 안에 study 실행 포함 | `evaluate_hpo.py` 로 BUDGET runner 분리 (`optuna_combo.py` 는 search 설계만) |

### 4.3 한 줄 요약

> 프레임워크 본체 (Phase 1 architecture-only → Phase 2 HPO-only, 단계마다 단일 파일을 agent 가 편집하는 autoresearch loop) 는 **bbb-combo1 과 사실상 동일**. 달라진 것은 ① worktree per combo 로 격리, ② holdout 평가의 세분화 + 의무화, ③ Phase 2 threshold 의 std buffer, ④ best_train.py 분리, ⑤ Optuna runner 와 search design 파일 분리 — 모두 "같은 방법론을 더 깔끔하게 운영" 수준.

---

## 5. Phase 2 keep threshold 의 std buffer 상세

### 5.1 규칙

`program_phase2.md` 명시:

> **keep** iff `best_confirm_mean_val_auc > threshold_mean + threshold_std`
> where the threshold is the max over:
>   - iter-200 phase-1 baseline (`results/<combo>/results.tsv` best keep row)
>   - all prior phase-2 keep rows in `results/<combo>/hpo/results.tsv`
> with `threshold_std` taken from the same row.

비교:

| Phase | Keep 규칙 |
|---|---|
| Phase 1 | `mean_val_auc > best_mean` (strict 부등호 하나) |
| Phase 2 | `best_confirm_mean_val_auc > best_mean + best_std` |

### 5.2 왜 buffer 가 필요한가

- confirm 은 **10 seed 로 고정**. 무한 sample 이 아니므로 같은 model 설정이라도 seed 운에 따라 `mean_val_auc` 가 자연적으로 ±std 정도 흔들림.
- Phase 1 처럼 `> best_mean` 만 쓰면, 의미 없는 노이즈도 50% 확률로 통과해버림 (false-keep).
- `+ best_std` 만큼의 buffer 를 더 두면 "**최소 한 seed-variance 만큼은 진짜로 이겨야** keep" 이라는 안전 마진이 생김.
- 1σ buffer → 정규성 가정 하 false-keep rate 약 16% (Phase 1 의 ~50% 보다 훨씬 보수적).

### 5.3 combo2 현재 상태에서의 실제 통과선

| 항목 | 값 |
|---|---|
| Phase 1 best iter | iter198 |
| threshold_mean (= iter198 `mean_val_auc`) | 0.852661 |
| threshold_std (= iter198 `std_val_auc`)   | 0.003551 |
| **Phase 2 keep 통과선** | **0.856212** |

즉 Phase 2 HPO 가 어떤 config 로 10-seed confirm 을 돌렸을 때
`best_confirm_mean_val_auc > 0.856212` 를 만족해야만 keep. 이는 Phase 1
best (0.8527) 보다 약 **+0.0036 AUC 더 좋아야** 한다는 뜻이며, 이후
Phase 2 keep 이 누적되면 threshold 도 같이 갱신되어 (가장 최근 keep row
의 mean+std 로) 점점 더 빡빡해짐.

### 5.4 부가 효과

- discard 도 `git revert` 만 하고 history 는 보존되므로, "노이즈로 보이지만 의미 있을 수도 있던 trial" 의 commit hash 가 TSV 에 남아 사후 재현 가능.
- buffer = "한 seed-variance" 정도이므로 너무 엄격하지도, 너무 느슨하지도 않은 trade-off.

---

## 6. 슬라이드 재생성

```bash
conda run -n scage_new python \
  /home/minji/combos-v2-combo2/results/maccs+scage1+mole/make_framework_slides.py
```

출력:
- `figures_framework/bbb_autoresearch_phase1_combo2.png`
- `figures_framework/bbb_autoresearch_phase2_combo2.png`
