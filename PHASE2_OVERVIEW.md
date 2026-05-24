# Phase-2 Overview (combos-v2-combo2)

Phase-2 autoresearch HPO 관련 파일·문서·결과·정책을 한 곳에 모은 요약본.
세부 운영 룰은 `program_phase2.md`가 source-of-truth.

---

## 1. Phase-1 vs Phase-2

| 단계 | 목표 | 주 lever 파일 | 산출물 |
|---|---|---|---|
| **Phase-1** (iter1~200) | Model architecture sweep — gMLP layers, attention pool, head 구조 등 | `train.py` (단일 lever) | `results/<combo>/results.tsv` 200 row, iter198 best (`mean_val_auc=0.852661±0.003551`) |
| **Phase-2** (iter1~12, ceiling 12) | Phase-1 frozen architecture에 대한 HP 최적화. Architecture는 절대 안 건드림. | `optuna_combo.py` + `best_train.py`의 BASE_CONFIG/training-procedure HP | `results/<combo>/hpo/*` + `architecture_log.md` 의 phase-2 row |

Phase-1은 **"어떤 architecture가 좋은가"**, Phase-2는 **"그 architecture에 어떤 HP가 가장 잘 맞는가"**. Phase-2가 Phase-1의 결과(=iter198 BASE_CONFIG)를 능가하면 KEEP, 아니면 DISCARD + `git revert`.

---

## 2. 파일 카탈로그

### 2.1 운영 문서

| 파일 | 역할 |
|---|---|
| `program_phase2.md` | Phase-2 agent operating spec. 모든 룰 (files, iteration loop, keep/discard, direction-switch, ceiling, final reporting) source-of-truth |
| `PHASE2_OVERVIEW.md` | 이 문서 — phase-2 관련 자료 한 페이지 요약 |
| `results/<combo>/hpo/conclusion.md` | Phase-2 종료 시 작성하는 최종 보고서 (현재 iter6까지 작성됨; iter7+로 invalidate된 상태) |

### 2.2 Frozen (절대 수정 안 하는 코드)

| 파일 | 이유 |
|---|---|
| `train.py` | Phase-1 active training file. iter198 시점에서 freeze. `evaluate_combo.py`/`final_holdout_eval.py`가 import해서 phase-1 결과 재현 |
| `prepare.py` | 데이터 로딩, scaffold split, feature 빌딩 |
| `evaluate_combo.py` | Phase-1 iter runner |
| `final_holdout_eval.py` | Phase-1 5-subset (internal/external/nn03/nn05/total) holdout breakdown runner |
| `data/*` | Raw / split 데이터 |

### 2.3 Phase-2 Lever (편집 가능)

| 파일 | 편집 범위 | Frozen 부분 |
|---|---|---|
| `optuna_combo.py` | **전면 자유 수정** — `suggest_config()` search space, sampler (TPE/CmaEs/Random), pruner (Median/Nop), `objective()` 멀티시드, `run_seeds()` 평가 protocol, `SEARCH_SEEDS_DEFAULT` 등 | `HOLDOUT_SUBSETS=["nn05","total"]` 만 frozen (architecture_log column 일관성 유지) |
| `best_train.py` | **제한 편집** — `BASE_CONFIG`에 새 HP 추가, `train_model`/`build_and_train`에 새 optional 인자·hook 추가 (LR scheduler, label smoothing, optimizer family 등) | Model class들 (`SpatialGatingUnit`, `gMLPBlock`, `gMLP`, `MultiModalGMLPFromFlat`)은 frozen. 새 HP의 default 값은 **phase-1 byte-identical** 보장 필수 |
| `evaluate_hpo.py` | 제한적 — 사용자 결정에 따라 `BUDGET` 수정 가능 (이미 50→30), keep 룰 수정 가능 (이미 mean+std → mean only) | 그 외 protocol 구조 변경 금지 |

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
     --combo <combo> --iter-id <N> --commit <sha10> \
     --note "<short>; top-region <one-liner>; next: <hint>"
   ```
   (search 30 trial + confirm top-3 × 10 seed + inline holdout nn05/total)
7. `results.tsv` 새 row의 `keep` 컬럼 확인
   - `True` → commit 유지
   - `False` → `git revert <commit>`
   - **Holdout (nn05/total) per-seed AUC/MCC/Acc** 는 study JSON `top_k_confirm[*]["holdout"]` + `architecture_log.md`에 자동 기록
8. Optuna SQLite (`combo2_hpo.db`)는 gitignored, 모든 iter study 보존

### Termination rules
- **5 consecutive discards** → 조기 종료 + conclusion.md
- **3 consecutive same-family discards** → direction switch (다른 lever 종류로)
- **Ceiling 12 iter** → 도달 시 종료
- **Autonomy (default)**: agent가 매 iter design 자율 선택. 사용자는 milestone만 알림 (KEEP, 조기 종료 trigger, direction switch, hard failure)

### Decision rule
- **Keep** iff `best_confirm_mean_val_auc > threshold_mean` (mean only, std buffer 없음)
- **Threshold** = max(phase-1 baseline mean, prior phase-2 keep means)
- **Holdout**은 트래킹만, keep/final selection에 직접 영향 없음 (bbb-combo1 패턴)
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
- `combo2_hpo.db` — Optuna SQLite store (모든 study trace, gitignored)

### 공통
- `architecture_log.md` — phase-1 (iter별 row) + phase-2 (study별 row) 통합 표. val_auc / nn05·total 의 AUC·MCC·Acc 직접 비교

---

## 5. Phase-1 Reference (combo2: maccs+scage1+mole)

iter198 best (`results.tsv`의 best keep row, 10 scaffold seed):

| 지표 | mean ± std (10 seed) |
|---|---|
| val ROC-AUC | **0.852661 ± 0.003551** |
| nn05 ROC-AUC (per-seed) | 0.870011 ± 0.008095 |
| total ROC-AUC (per-seed) | 0.879823 ± 0.004205 |

Phase-2 keep 기준 = val mean > **0.852661**.

---

## 6. Phase-2 진행 상태 (iter1~6 + iter7 진행 중)

| iter | lever (단일 변경) | family | confirm val | keep | 비고 |
|---|---|---|---|---|---|
| 1 | structural narrow (drop d_model/d_ffn/depth from search) | space | 0.852668 | False* | TPE wide가 trial#7로 수렴 |
| 2 | + MedianPruner + TPE n_startup=15 multivariate (best_train.py에 `*_with_pruning` helper 추가) | pruner+sampler | 0.852850 | False* | trial#7 동일, pruner는 wall-time 도구 |
| 3 | + 5-seed mean objective | objective | **0.852850** | **True** | val +0.000189 (paired p=0.881 noise) |
| 4 | search space narrow around trial#7 | space | 0.852740 | False | search-to-confirm overfit |
| 5 | sampler TPE → CmaEsSampler | sampler | 0.852850 (=#7 HP) | False | CmaEs도 trial#7 정확 발견 |
| 6 | pruner MedianPruner → NopPruner | pruner | 0.852850 (=#7 HP) | False | full-epoch eval도 trial#7 |
| (7) | search seeds confirm-disjoint pool [1000-5000] | objective | — | (aborted) | design flaw (confirm 기준 변화 X) |
| **7** | **BASE_CONFIG 확장: LR schedule + label_smoothing + ema_decay** | best_train.py kwargs | 진행 예정 | — | iter7 새 design (commit pending) |

\* iter1/2는 옛 룰 (mean+1·std) 기준. 새 룰 (mean only)로는 marginal keep이지만 retroactive rewrite 없음.

### iter1~6 종합 결론
- **trial#7 (lr=1.5e-4, wd=1.5e-4, bs=128, d_model=512, d_ffn=1536, depth=5, dropout=0.243, drop_path=0.002, mod_drop_p=0.090, head_dropout=0.198)** 가 wide HP search space의 robust peak.
- TPE / CmaEs / NopPruner 어떤 변형이든 동일 HP 발견.
- val mean 향상 (+0.000189)은 paired t-test로 noise 수준.
- Holdout AUC는 phase-1보다 미세하게 낮음, MCC/Accuracy는 미세하게 높음 (모두 std band 안).
- **bbb-combo1 phase-2와 동일 패턴**: HP만으로는 phase-1 holdout 능가 어려움.

### iter7 design (현재 commit pending)
- **lever**: `best_train.py` BASE_CONFIG에 LR schedule + label smoothing + EMA decay 도입 + `optuna_combo.py`에서 search.
- 새 HP 5개 추가, 모두 default phase-1 byte-identical:
  - `lr_schedule="constant"` / `lr_warmup_epochs=0` / `lr_min_ratio=0.0`
  - `label_smoothing=0.0`
  - `ema_decay=0.999` (기존 hardcoded 값 expose)
- Search 범위: lr_schedule ∈ {constant, cosine, warmup_cosine}, warmup 0~10 epoch, min_ratio 0~0.3, label_smoothing 0~0.1, ema_decay 0.99~0.9999 (log).

---

## 7. Protocol 진화 History

phase-2 진행하면서 사용자 결정으로 protocol을 몇 번 갱신:

| 변경 | iter | 이유 |
|---|---|---|
| `*_with_pruning` helper를 best_train.py에 추가 허용 | iter2 | pruning 도입 위해 train_model에 hook 필요 |
| Keep 룰 mean+1·std → **mean only** | iter3 직후 (사용자 결정) | std buffer가 3.16·SE에 해당해 너무 엄격, noise 미만 향상도 KEEP로 잡아 search direction 유지 |
| Ceiling 12 → 40 → 12 | iter3 직후 → iter6 직후 (사용자 결정) | 처음 40으로 확장했다가 wall-time 부담으로 12로 복귀 |
| Autonomy mode 명문화 | iter3 직후 | autoresearch 본질 = agent가 매 iter design 자율 선택 |
| `evaluate_hpo.py` BUDGET `n_trials` 50 → **30** | iter5부터 적용 (사용자 결정) | iter1~4가 모두 trial#7로 deterministic 수렴 → trial 수 줄여도 fairness loss 미미 |
| Inline holdout in confirm phase (별도 `phase2_holdout_eval.py` 제거) | iter3~ | bbb-combo1 패턴 정합 — confirm 한 step에서 val+holdout 동시 측정 |
| best_train.py BASE_CONFIG 확장 lever **공식 허용** | iter7 (사용자 결정) | optuna_combo.py 안의 lever로는 trial#7 능가 못함이 iter5/6에서 확인 — bbb-combo1처럼 training-procedure HP 추가 |

---

## 8. 빠른 명령 참조

```bash
# Phase-2 iter 실행 (n_trials=30, top-3 confirm, inline holdout)
nohup conda run -n rapids-25.02 python evaluate_hpo.py \
  --combo maccs+scage1+mole --iter-id <N> --commit <sha10> \
  --note "<short>; top-region; next:" \
  > results/maccs+scage1+mole/hpo/iter<N>_run.log 2>&1 &

# 진행 상황 확인
ps -fp $(pgrep -f optuna_combo)
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Study DB inspect
conda run -n rapids-25.02 python -c "
import optuna
s = optuna.load_study(study_name='auto_iter00N_<sha>', storage='sqlite:///combo2_hpo.db')
print(f'trials: {len(s.trials)}, best: #{s.best_trial.number} val={s.best_value:.6f}')
"
```
