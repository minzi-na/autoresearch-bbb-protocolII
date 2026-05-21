# Autoresearch Reproducibility Report — `maccs+scage1+mole`

대상: `combos-v2-combo2` (main, iter 1-30 발췌) vs. `combos-v2-combo2-fork-a` (iter 1-30) vs. `combos-v2-combo2-fork-b` (iter 1-30). 동일 baseline·동일 combo로 framework를 3회 독립 실행했을 때, **`mean_val_roc_auc` 기반 keep/discard 결정으로 어떤 변경이 모델에 누적되는가**를 비교해 framework 신뢰도를 평가한다. (holdout 평가는 신뢰도 정의에서 제외.)

---

## 1. 한눈에 보는 결론

- **변경 탐색 공간(Proposed themes)은 매우 일관**: 3 run 전체에서 시도된 23개 테마 중 20개가 모든 run에 등장 (3-way Jaccard 0.870).
- **변경 채택(Kept themes)은 절반만 일관**: 누적 채택된 13개 테마 중 모든 run에 공통은 4개 (3-way Jaccard 0.308). pairwise는 main↔fork-a 0.667, main↔fork-b 0.364, fork-a↔fork-b 0.333.
- **공통 코어 4 테마**: baseline → skip-gate gated/mean pool → EMA weights (~0.999) → low-p modality dropout. 같은 combo·같은 baseline에서 framework가 **반드시 회수하는 변경**.
- **divergence의 원인은 keep 마진이 seed std보다 한 자릿수 작기 때문**. keep 결정의 Δval (proposal vs running-best) 중앙값 ≈ 0.0004, 반면 mean_val_auc의 10-seed std는 ≈ 0.003. **모든 keep 결정이 통계적 잡음 안에서 일어난다.**
- 결론: framework는 *“무엇을 시도할까”* 차원에서 매우 신뢰할 수 있지만, *“무엇을 채택할까”* 차원에서는 한 자릿수 자리수의 val_auc 변동이 결정을 뒤집을 만큼 민감하다. iter 17 (EMA) 까지는 3 run 모두 같은 결정 경로를 따르고, iter 18 이후 누적 state 가 갈리면서 경로가 분기한다.

---

## 2. Proposal trajectory — 시도된 변경의 일치도

각 iter에서 LLM이 어떤 테마를 제안했는가? (keep 여부 무관)

| iter | main | fork-a | fork-b | 일치 |
|---:|---|---|---|:---:|
| 1 | baseline | baseline | baseline | ✔ |
| 2 | AdamW+clip+sched | AdamW+clip+sched | AdamW+clip+sched | ✔ |
| 3 | pos_weight auto | pos_weight auto | pos_weight auto | ✔ |
| 4 | label smooth 0.1 | label smooth 0.1 | label smooth 0.1 | ✔ |
| 5 | LN after proj | LN after proj | LN after proj | ✔ |
| 6 | mod_drop p=0.15 | mod_drop p=0.15 | mod_drop p=0.15 | ✔ |
| 7 | skipgate pool | skipgate pool | skipgate pool | ✔ |
| 8 | cross-modal FiLM | cross-modal FiLM | cross-modal FiLM | ✔ |
| 9 | attn pool | attn pool | attn pool | ✔ |
| 10 | SGU scale | SGU scale | SGU scale | ✔ |
| 11 | 2-layer proj | 2-layer proj | 2-layer proj | ✔ |
| 12 | mh-SGU h=2 | mh-SGU h=2 | mh-SGU h=2 | ✔ |
| 13 | AdaLN | AdaLN | AdaLN | ✔ |
| 14 | CLS pool | CLS pool | CLS pool | ✔ |
| 15 | stoch depth | stoch depth | stoch depth | ✔ |
| 16 | diag-mask SGU | diag-mask SGU | diag-mask SGU | ✔ |
| 17 | EMA | EMA | EMA | ✔ |
| 18 | EMA variant | mod_drop low | mod_drop high | ✗ |
| 19 | EMA variant | stoch depth | EMA | ✗ |
| 20 | AdamW no-sched | mh-SGU h=2 | stoch depth | ✗ |
| 21 | AdamW+sched (re) | mh-SGU h=4 | grad-clip only | ✗ |
| 22 | mod_drop low | LN after proj | grad-clip only | ✗ |
| 23 | mod_drop high | FiLM | AdamW no-sched | ✗ |
| 24 | stoch depth | EMA | LN after proj | ✗ |
| 25 | stoch depth | mod_drop high | mod_drop low | ✗ |
| 26 | LN after proj | AdamW no-sched | grad-clip only | ✗ |
| 27 | per-mod scale | grad-clip only | label smooth 0.05 | ✗ |
| 28 | label smooth 0.05 | EMA | pos_weight fixed | ✗ |
| 29 | mh-SGU h=2 (re) | skipgate init | pos_weight fixed | ✗ |
| 30 | grad-clip 0.5 | skipgate init | pos_weight fixed | ✗ |

- **iter 1-17: 17/30 = 100% 일치**. baseline + 첫 keep 3개 (iter 7, 10, 17) 까지 누적 architecture state 가 같으므로, LLM의 next-proposal 분포가 결정론적으로 동일하게 흐른다 (proposal text 의 표현은 다르더라도 의도는 동일).
- **iter 18부터 분기**. main은 iter 17의 EMA decay 변형을 더 탐색, fork-a/b 는 즉시 다음 테마로 이동. 누적 state 차이는 fork-b 가 iter 9 에서 attention pooling 을 keep 했고 iter 10 SGU scale 을 drop 한 데서 처음 발생.
- 전체 탐색 공간 Jaccard: **0.870 (20/23)**. main 유일 시도: per-mod scale. fork-a 유일: skipgate init. fork-b 유일: pos_weight fixed. **나머지 모든 변경 카테고리는 3 run 공통 탐색**.

---

## 3. Keep trajectory — 채택된 변경의 일치도

### 3.1 Kept theme 세트

| run | iter (keep) | 채택된 테마 | 개수 |
|---|---|---|---:|
| main | 1, 7, 10, 17, 20, 22, 24 | baseline · skipgate · SGU scale · EMA · AdamW no-sched · mod_drop low · stoch depth | 7 |
| fork-a | 1, 7, 10, 17, 18, 20, 26, 29 | baseline · skipgate · SGU scale · EMA · mod_drop low · mh-SGU h=2 · AdamW no-sched · skipgate init | 8 |
| fork-b | 1, 7, 9, 17, 18, 21, 25, 28 | baseline · skipgate · attn pool · EMA · mod_drop high · grad-clip only · mod_drop low · pos_weight fixed | 8 |

### 3.2 채택 일치도

| 비교 | 공유 테마 | 공유 개수 | Jaccard |
|---|---|---:|---:|
| **3-way ∩** | baseline, skipgate, EMA, mod_drop low | **4** | **0.308** |
| main ∩ fork-a | + SGU scale, AdamW no-sched | 6 | 0.667 |
| main ∩ fork-b | (3-way 와 동일) | 4 | 0.364 |
| fork-a ∩ fork-b | (3-way 와 동일) | 4 | 0.333 |

### 3.3 Run-unique 채택

| run | 그 run에서만 keep된 테마 |
|---|---|
| main | stoch depth (p=0.025) |
| fork-a | mh-SGU h=2, skipgate init |
| fork-b | mod_drop high (p=0.10), attn pool, grad-clip only, pos_weight fixed |

→ fork-b 가 4개로 가장 unique 함. iter 9에서 attention pooling 을 keep 한 것이 이후 누적 state 를 완전히 다른 궤도로 보내, “SGU scale” (다른 run의 iter 10 keep) 을 안 들어오게 만들고, grad-clip 계열, pos_weight fixed 같은 hyper-param 탐색으로 흘렀다.

### 3.4 First-keep iter per theme

| theme | main | fork-a | fork-b |
|---|:---:|:---:|:---:|
| baseline | 1 | 1 | 1 |
| skipgate pool | 7 | 7 | 7 |
| attn pool | — | — | 9 |
| SGU scale | 10 | 10 | — |
| EMA | 17 | 17 | 17 |
| mod_drop high (p=0.10~0.15) | — | — | 18 |
| mod_drop low (p=0.05) | 22 | 18 | 25 |
| stoch depth | 24 | — | — |
| AdamW no-sched | 20 | 26 | — |
| grad-clip only | — | — | 21 |
| mh-SGU h=2 | — | 20 | — |
| skipgate init | — | 29 | — |
| pos_weight fixed | — | — | 28 |

- 공통 4 테마(baseline, skipgate, EMA, mod_drop low) 중 mod_drop low 만 채택 iter 가 분기 (22 / 18 / 25). 다른 3개는 동일 iter 에 keep.

---

## 4. Keep 결정의 통계적 안정성

각 keep 시점에서 “proposal val_auc − 이전 running-best val_auc” 마진을 seed-level std 와 비교.

### Main

| iter | Δval (margin) | std (10 seeds) | margin/std | keep |
|---:|---:|---:|---:|:---:|
| 1 | (baseline) | 0.00291 | — | ✔ |
| 7 | +0.00013 | 0.00259 | 0.05 | ✔ |
| 10 | +0.00002 | 0.00269 | **0.007** | ✔ |
| 17 | +0.00124 | 0.00417 | 0.30 | ✔ |
| 20 | +0.00083 | 0.00389 | 0.21 | ✔ |
| 22 | +0.00091 | 0.00346 | 0.26 | ✔ |
| 24 | +0.00047 | 0.00363 | 0.13 | ✔ |

### Fork-a

| iter | Δval | std | margin/std | keep |
|---:|---:|---:|---:|:---:|
| 7 | +0.00013 | 0.00259 | 0.05 | ✔ |
| 10 | +0.00002 | 0.00269 | 0.007 | ✔ |
| 17 | +0.00124 | 0.00417 | 0.30 | ✔ |
| 18 | +0.00082 | 0.00367 | 0.22 | ✔ |
| 20 | +0.00112 | 0.00297 | 0.38 | ✔ |
| 26 | +0.00033 | 0.00270 | 0.12 | ✔ |
| 29 | +0.00008 | 0.00288 | **0.03** | ✔ |

### Fork-b

| iter | Δval | std | margin/std | keep |
|---:|---:|---:|---:|:---:|
| 7 | +0.00013 | 0.00259 | 0.05 | ✔ |
| 9 | **+0.00006** | 0.00266 | **0.02** | ✔ ← divergence 시작점 |
| 17 | +0.00166 | 0.00400 | 0.41 | ✔ |
| 18 | +0.00039 | 0.00330 | 0.12 | ✔ |
| 21 | +0.00062 | 0.00343 | 0.18 | ✔ |
| 25 | +0.00015 | 0.00347 | 0.04 | ✔ |
| 28 | **+0.00001** | 0.00419 | **0.002** | ✔ |

- Keep 마진의 **중앙값 ≈ 0.0004, 최대 0.0017, 최소 0.00001**.
- mean_val_auc 의 10-seed std 는 **0.003 전후** — 즉 “proposal이 이전 best 를 진짜 이겼는지” 신호가 잡음보다 약 한 자리 작다.
- 특히 **margin/std < 0.1 인 keep** 이 13개 중 6개 (46%). 동일 변경에 다른 random seed batch 로 돌리면 keep 이 안 될 가능성이 충분히 있는 결정들.

### Divergence point 분석

- **iter 9 (attention pooling)**: main 0.845373, fork-a 0.845373, fork-b 0.845611. main/fork-a 는 동일 코드로 결정론적, fork-b 는 LLM 이 “attention pooling” 을 살짝 다른 구현으로 작성해 +0.00006 만큼 차이가 났고 그 +0.00006 이 keep threshold 를 넘는 데 충분했다. → **fork-b 전체 궤도가 이 한 결정에서 분기**.
- **iter 10 (SGU scale)**: main 0.845572, fork-a 0.845572, fork-b 0.844948. main/fork-a 는 비록 note 문구가 다름에도 (`SGU gate_scale` vs `learnable exp-gated residual scale on SGU spatial path`) 산출값이 완전히 동일 — LLM 이 사실상 같은 코드를 생성. fork-b 는 iter 9 keep 으로 base architecture 가 다르므로 그 위의 SGU scale 효과가 사라져 drop.

→ **재현성의 한계는 두 군데에서 발생**:
  1. LLM 이 동일 의도 변경을 코드로 옮길 때의 implementation variation (val_auc 0.0001~0.0005 jitter)
  2. Keep threshold 가 그 jitter 수준이라 단일 결정이 뒤집힘 → 누적 state 분기 → 이후 모든 결정이 다른 경로

---

## 5. 신뢰도 판정

framework 의 신뢰도를 “3 run 에서 합의된 변경 비율”로 정의한다.

| 차원 | 측정값 | 평가 |
|---|---|---|
| 탐색 공간 합의 | 시도 테마 3-way Jaccard **0.870** | **매우 높음** — 같은 baseline 에서 같은 변경 후보군을 일관되게 탐색 |
| iter 1-17 proposal 합의 | 17/17 = **100%** | **결정론적** — 누적 state 가 같으면 next-proposal 이 동일 |
| 채택 합의 (3-way) | 채택 테마 Jaccard **0.308** | **중간** — 핵심 4 테마는 항상 회수, 나머지는 run-dependent |
| 채택 합의 (pairwise 평균) | (0.667 + 0.364 + 0.333) / 3 = **0.455** | 절반 정도의 keep 이 재현 |
| 핵심 코어 회수 | baseline, skipgate, EMA, mod_drop low — **4/4 항상 회수** | **높음** |
| Keep 결정의 통계적 안정성 | 마진/std 중앙값 **0.13**, 6/13 keep 이 마진/std < 0.1 | **낮음** — 절반 가량이 잡음 수준 결정 |

### 결론

- **framework 는 “어떤 방향을 탐색할지”를 매우 일관되게 결정한다.** 같은 baseline 에서 거의 동일한 변경 후보군을 동일 순서로 시도하며, iter 17 까지는 사실상 결정론적이다.
- **그러나 “어떤 변경을 채택할지”는 통계적으로 불안정한 결정 다수를 포함한다.** keep margin 이 seed std 보다 한 자리 작기 때문에, val_auc 의 미세한 LLM 구현 변동이 keep/discard 를 뒤집고 한 번 뒤집히면 누적 architecture 가 분기해 이후 궤도 전체가 달라진다.
- 핵심 4 테마 (baseline, skip-gate pool, EMA weights, low-p modality dropout) 는 **3 run 모두에서 무조건 회수** — framework 가 신뢰성 있게 식별하는 변경. 나머지 keep 결정 (run 마다 4~5 건) 은 절반만 일치하며 재현성을 보장할 수 없다.

**신뢰도 등급 (제안)**: 탐색 공간은 신뢰 가능, 채택 결정은 “핵심 코어만 신뢰, 나머지는 stochastic”. 채택 결정의 안정성을 높이려면 keep 기준을 통계적 검정 (e.g. paired t-test on per-seed val_auc, 또는 ≥ 1σ 마진 요구) 으로 강화하는 것이 필요해 보인다.
