# Phase-2 Overview (combos-v2-combo1)

Phase-2 autoresearch HPO 관련 파일·문서·결과·정책을 한 곳에 모은 요약본.
세부 운영 룰은 `program_phase2.md`가 source-of-truth.

---

## 1. Phase-1 vs Phase-2

| 단계 | 목표 | 주 lever 파일 | 산출물 |
|---|---|---|---|
| **Phase-1** (iter1~200) | Model architecture sweep — multi-head SGU, R-Drop, EMA decay, AdamW split, pool skip-gate 등 | `train.py` (단일 lever) | `results/<combo>/results.tsv` 200 row, iter197 best (`mean_val_auc=0.854947±0.002638`) |
| **Phase-2** (iter1~18, ceiling 18) | Phase-1 frozen architecture에 대한 HP 최적화. Architecture는 절대 안 건드림. | `optuna_combo.py` + `best_train.py`의 BASE_CONFIG/training-procedure HP | `results/<combo>/hpo/*` + `architecture_log.md` 의 phase-2 row |

Phase-1은 **"어떤 architecture가 좋은가"**, Phase-2는 **"그 architecture에 어떤 HP가 가장 잘 맞는가"**. Phase-2가 Phase-1의 결과(=iter197 BASE_CONFIG)를 능가하면 KEEP, 아니면 DISCARD + `git revert`.

---

## 2. 파일 카탈로그

### 2.1 운영 문서

| 파일 | 역할 |
|---|---|
| `program_phase2.md` | Phase-2 agent operating spec. 모든 룰 (files, iteration loop, keep/discard, direction-switch, ceiling, final reporting) source-of-truth |
| `PHASE2_OVERVIEW.md` | 이 문서 — phase-2 관련 자료 한 페이지 요약 |
| `results/<combo>/hpo/conclusion.md` | Phase-2 종료 시 작성하는 최종 보고서 (아직 미작성; ceiling 18 or 5-consecutive-discard 시 작성) |

### 2.2 Frozen (절대 수정 안 하는 코드)

| 파일 | 이유 |
|---|---|
| `train.py` | Phase-1 active training file. iter197 시점에서 freeze. `evaluate_combo.py`/`final_holdout_eval.py`가 import해서 phase-1 결과 재현 |
| `prepare.py` | 데이터 로딩, scaffold split, feature 빌딩 |
| `evaluate_combo.py` | Phase-1 iter runner |
| `final_holdout_eval.py` | Phase-1 5-subset (internal/external/nn03/nn05/total) holdout breakdown runner |
| `data/*` | Raw / split 데이터 |

### 2.3 Phase-2 Lever (편집 가능)

| 파일 | 편집 범위 | Frozen 부분 |
|---|---|---|
| `optuna_combo.py` | **전면 자유 수정** — `suggest_config()` search space, sampler (TPE/CmaEs/Random), pruner (Median/Nop), `objective()` 멀티시드, `run_seeds()` 평가 protocol, `SEARCH_SEEDS_DEFAULT` 등 | `HOLDOUT_SUBSETS=["nn05","total"]` 만 frozen (architecture_log column 일관성 유지) |
| `best_train.py` | **제한 편집** — `BASE_CONFIG`에 새 HP 추가, `train_model`/`build_and_train`에 새 optional 인자·hook 추가 (LR scheduler, label smoothing, optimizer family, R-Drop alpha-max expose 등) | Model class들 (`SpatialGatingUnit` multi-head, `gMLPBlock`, `gMLP`, `MultiModalGMLPFromFlat`)은 frozen. 새 HP의 default 값은 **phase-1 byte-identical** 보장 필수 |
| `evaluate_hpo.py` | 제한적 — 사용자 결정에 따라 `BUDGET` 수정 가능 (combo2 패턴 적용: n_trials=30, mean-only keep) | 그 외 protocol 구조 변경 금지 |

---

## 3. Phase-2 Iteration Loop (8 step)

각 iter `N`마다:

1. `results/<combo>/hpo/results.tsv` 마지막 keep=True 행(없으면 phase-1 baseline)에서 threshold 읽기 + 이전 note들 스캔
2. **단일 lever 선택** — search space 조정, sampler/pruner 교체, objective 멀티시드, BASE_CONFIG 새 HP 도입 등
3. `optuna_combo.py` (필요시 `best_train.py`) partial Edit
4. (옵션) Haiku 4.5 sanity check
5. `git commit -m "iter<N>: <short>"`
6. Background 실행:
   ```
   conda run -n rapids-25.02 python evaluate_hpo.py \
     --combo maccs+avalon+scage2+mole --iter-id <N> --commit <sha10> \
     --note "<short>; top-region <one-liner>; next: <hint>"
   ```
   (search 30 trial + confirm top-3 × 10 seed + inline holdout nn05/total)
7. `results.tsv` 새 row의 `keep` 컬럼 확인
   - `True` → commit 유지
   - `False` → `git revert <commit>`
   - **Holdout (nn05/total) per-seed AUC/MCC/Acc** 는 study JSON `top_k_confirm[*]["holdout"]` + `architecture_log.md`에 자동 기록
8. Optuna SQLite (`combo1_hpo.db`)는 gitignored, 모든 iter study 보존

### Termination rules
- **5 consecutive discards** → 조기 종료 + conclusion.md
- **3 consecutive same-family discards** → direction switch (다른 lever 종류로)
- **Ceiling 18 iter** → 도달 시 종료
- **Autonomy (default)**: agent가 매 iter design 자율 선택. 사용자는 milestone만 알림 (KEEP, 조기 종료 trigger, direction switch, hard failure)

### Decision rule
- **Keep** iff `best_confirm_mean_val_auc > threshold_mean` (mean only, std buffer 없음)
- **Threshold** = max(phase-1 baseline mean, prior phase-2 keep means)
- **Holdout**은 트래킹만, keep/final selection에 직접 영향 없음 (bbb-combo1 / combo2 패턴)
- **Final selection** = `keep=True` 중 `val_mean` 최대 row. Holdout은 generalization 보고용.

---

## 4. 결과 Artifacts (results/<combo>/)

### Phase-1
- `results.tsv` — 200 row architecture sweep history
- `iter<NNNN>_per_seed.json` — iter별 10-seed val metrics 상세
- `holdout_eval/iter<NNNN>_<commit>.json` — phase-1 best 후보의 5-subset holdout 상세

### Phase-2
- `hpo/results.tsv` — phase-2 iter별 결과 (iter, commit, search/confirm val_auc, threshold, keep, note)
- `hpo/study_<stamp>.json` — Optuna study summary (best params, top_k_confirm + inline holdout block per trial)
- `hpo/trials_<stamp>.tsv` — 모든 trial의 params + state + val_auc
- `hpo/conclusion.md` — 종료 시 작성하는 최종 보고서
- `combo1_hpo.db` — Optuna SQLite store (모든 study trace, gitignored)

### 공통
- `architecture_log.md` — phase-1 (iter별 row) + phase-2 (study별 row) 통합 표. val_auc / nn05·total 의 AUC·MCC·Acc 직접 비교

---

## 5. Phase-1 Reference (combo1: maccs+avalon+scage2+mole)

iter197 best (`results.tsv`의 best keep row, commit `9bc22da482`, 10 scaffold seed):

| 지표 | mean ± std (10 seed) |
|---|---|
| val ROC-AUC | **0.854947 ± 0.002638** |
| nn05 ROC-AUC (per-seed) | 0.856316 ± 0.007349 |
| nn05 MCC (per-seed) | 0.557481 ± 0.011201 |
| nn05 Accuracy (per-seed) | 0.857751 ± 0.005049 |
| total ROC-AUC (per-seed) | 0.876385 ± 0.003684 |
| total MCC (per-seed) | 0.609062 ± 0.010027 |
| total Accuracy (per-seed) | 0.849887 ± 0.004363 |

Phase-2 keep 기준 = val mean > **0.854947**.

---

## 6. Phase-2 진행 상태

iter1 시작 전. `hpo/results.tsv`는 비어있음 (첫 `evaluate_hpo.py` 실행 시 헤더와 함께 생성).

| iter | lever (단일 변경) | family | confirm val | keep | 비고 |
|---|---|---|---|---|---|
| (아직 없음) | | | | | |

### iter1 starting state (best_train.py / optuna_combo.py 포팅 직후)

- **best_train.py**: combo1 train.py (commit `9bc22da482`) 베이스 + phase-2 lever 확장. 5개 새 HP 키 추가, default는 모두 iter197 byte-identical:
  - `lr_schedule="constant"`, `lr_warmup_epochs=0`, `lr_min_ratio=0.0`
  - `label_smoothing=0.0`
  - `ema_decay=0.9993`, `ema_warmup_epochs=1`
- **optuna_combo.py**: iter1 starting `suggest_config()`는 6-D narrow pilot — structural pinned (d_model/d_ffn/depth/use_gated_pool/batch_size at iter197 values), search lr (5e-5~2e-4) + 5 phase-2 keys. iter1 agent가 첫 design choice로 이걸 자체적으로 재설계할 것.
- **evaluate_hpo.py**: BUDGET = `{n_trials:30, top_k:3, search_num_epochs:30, search_seeds:[42,100,200], confirm_seeds:[42,100,...,900]}`. combo2 iter5-onward와 동일.

### Combo1 phase-2 특이사항 (combo2와 다른 점)

1. **`BASE_CONFIG.weight_decay` 무효화** — `train_model` 안의 AdamW override (`wd=0.003` hardcoded, iter148 frozen) 때문에 BASE_CONFIG의 weight_decay는 모델에 영향 없음. Optuna가 sweep해도 효과 없음 → iter1+ 에서 expose 가능 (best_train.py에 `adamw_wd` 새 키 추가).
2. **AdamW lr multiplier 1.25 hardcoded** — `lr = optimizer.param_groups[0]["lr"] * 1.25` (iter38 frozen). Optuna가 BASE_CONFIG.lr=2e-4 setting하면 실제 AdamW lr=2.5e-4. 추후 untangle 가능.
3. **R-Drop hardcoded** — 2-pass loss + 5-epoch alpha warmup + alpha_max=1.0 (iter80/106 frozen). expose 가능: `rdrop_alpha_max`, `rdrop_warmup_epochs`.
4. **Model class에 hardcoded HP** — `Dropout(0.10)`, `mod_drop_p=0.10`, `SGU_N_HEADS=2`, `DROP_V_PATH=0.08`. 기본적으로 frozen이지만 부분적으로 phase-2 lever로 expose 가능 (constructor kwarg화).

### bbb-combo1 phase-2 / combo2 phase-2 lessons

- HP만으로 phase-1 holdout 능가 어렵다는 게 두 case 모두에서 관측됨.
- val mean 향상 (+0.0001 ~ +0.0002)은 paired t-test로 noise 수준인 경우가 많음.
- Holdout AUC는 phase-1보다 미세하게 낮거나 비슷, MCC/Accuracy는 미세하게 높음 (모두 std band 안).
- combo1 phase-2도 같은 패턴 예상되지만, lever surface가 combo2보다 넓음 (hardcoded HP가 많아 expose 여지 큼) → 더 의미있는 향상 가능성 존재.

---

## 7. Protocol 적용 (combo2 결정 그대로 채택)

combo2 phase-2에서 사용자가 결정한 protocol을 combo1에도 동일 적용:

| 항목 | combo2에서 결정 | combo1 적용 |
|---|---|---|
| Keep 룰 | mean only (std buffer 없음) | 동일 |
| `evaluate_hpo.py` BUDGET `n_trials` | 30 (iter5-onward) | iter1부터 30 |
| Inline holdout in confirm phase | iter3~ 채택 | iter1부터 적용 (별도 `phase2_holdout_eval.py` 없음) |
| Ceiling | 18 (iter11+ 확장) | iter1부터 18 |
| Autonomy mode | iter3 명문화 | 시작부터 적용 |
| `*_with_pruning` helper 허용 | iter2부터 | best_train.py에 처음부터 포함 |
| best_train.py BASE_CONFIG 확장 lever | iter7 공식 허용 | 시작부터 5개 키 (lr_schedule + lr_warmup + lr_min_ratio + label_smoothing + ema_decay + ema_warmup_epochs) 포함 |

iter 진행 중 사용자 결정으로 추가 protocol 변경 발생 시 이 표에 row 추가.

---

## 8. 빠른 명령 참조

```bash
# Phase-2 iter 실행 (n_trials=30, top-3 confirm, inline holdout)
nohup conda run -n rapids-25.02 python evaluate_hpo.py \
  --combo maccs+avalon+scage2+mole --iter-id <N> --commit <sha10> \
  --note "<short>; top-region; next:" \
  > results/maccs+avalon+scage2+mole/hpo/iter<N>_run.log 2>&1 &

# 진행 상황 확인
ps -fp $(pgrep -f optuna_combo)
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Study DB inspect
conda run -n rapids-25.02 python -c "
import optuna
s = optuna.load_study(study_name='auto_iter00N_<sha>', storage='sqlite:///combo1_hpo.db')
print(f'trials: {len(s.trials)}, best: #{s.best_trial.number} val={s.best_value:.6f}')
"
```
