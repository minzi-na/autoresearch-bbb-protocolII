#!/usr/bin/env python3
"""Build pre/post-autoresearch vs baseline comparison report at 100-iter snapshot.

Generates:
  - pre_post_baseline_comparison_100.md
  - figures/grid_ensemble_100.png
  - figures/grid_perseed_100.png
  - figures/delta_post_pre_100.png
  - figures/total_compare_100.png

Post checkpoints are now read from the per-combo `weights/iter<N>/` directories
(autoresearch ran out to 100 iters per combo). For each combo we use the best
keep iter that has a saved `final_holdout_eval` JSON.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 110,
})

ROOT = Path("/home/minji/autoresearch_combos_v2")
OUT_DIR = ROOT / "results" / "comparison"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# (name, combo_str, it_pre, it_post, worktree_root)
# it_post is the iter saved as the current "best" under results/<combo>/weights/iter<N>/
COMBOS = [
    ("combo1", "maccs+avalon+scage2+mole",  1, 93,  "/home/minji/combos-v2-combo1"),
    ("combo2", "maccs+scage1+mole",          1, 100, "/home/minji/combos-v2-combo2"),
    ("combo3", "maccs+scage1+scage2+mole",   1, 90,  "/home/minji/combos-v2-combo3"),
]
# Previous-report post iters, kept for the 100-vs-50 progression comparison
OLD_POST_ITER = {"combo1": 38, "combo2": 60, "combo3": 54}
BASELINES = ["mlp2", "mlp3", "mlp4", "xgboost", "lightgbm"]
MODEL_ORDER = ["GateMol pre", "GateMol post"] + BASELINES
SUBSETS = ["internal", "external", "nn03", "nn05", "total"]
SUBSET_N = {"internal": 179, "external": 709, "nn03": 39, "nn05": 329, "total": 888}

MODEL_COLORS = {
    "GateMol pre":  "#1f77b4",
    "GateMol post": "#0b3d91",
    "mlp2":         "#ffbb78",
    "mlp3":         "#ff7f0e",
    "mlp4":         "#d62728",
    "xgboost":      "#2ca02c",
    "lightgbm":     "#8c564b",
}


def find_holdout(holdout_dir, it):
    for fn in os.listdir(holdout_dir):
        m = re.match(r"iter(\d+)_", fn)
        if m and int(m.group(1)) == it:
            return holdout_dir / fn
    return None


def collect_keep_iters(combo_dir, last_iter):
    """Return list of iters in 1..last_iter where summary['keep'] is True."""
    keeps = []
    for it in range(1, last_iter + 1):
        fp = combo_dir / f"iter{it:04d}_per_seed.json"
        if not fp.exists():
            continue
        s = json.load(open(fp))["summary"]
        if s.get("keep") is True:
            keeps.append(it)
    return keeps


# Load all data
DATA = {}
PRE_POST_META = {}
KEEP_ITERS = {}
for name, combo_str, it_pre, it_post, wt in COMBOS:
    bdir = ROOT / "results" / combo_str / "baselines"
    cdir = Path(wt) / "results" / combo_str
    hd = cdir / "holdout_eval"
    pre_h  = json.load(open(find_holdout(hd, it_pre)))
    post_h = json.load(open(find_holdout(hd, it_post)))
    bjsons = {m: json.load(open(bdir / f"{m}.json")) for m in BASELINES}
    old_it = OLD_POST_ITER[name]
    old_post_h = json.load(open(find_holdout(hd, old_it)))
    DATA[name] = {
        "combo_str": combo_str,
        "pre":  pre_h["summary"]["subsets"],
        "post": post_h["summary"]["subsets"],
        "old_post":      old_post_h["summary"]["subsets"],
        "old_post_iter": old_it,
        "pre_val":  pre_h["summary"]["val"],
        "post_val": post_h["summary"]["val"],
        "baselines": {m: bjsons[m]["summary"] for m in BASELINES},
    }
    pre_meta  = json.load(open(cdir / f"iter{it_pre:04d}_per_seed.json"))["summary"]
    post_meta = json.load(open(cdir / f"iter{it_post:04d}_per_seed.json"))["summary"]
    old_meta  = json.load(open(cdir / f"iter{old_it:04d}_per_seed.json"))["summary"]
    PRE_POST_META[name] = {
        "combo_str": combo_str,
        "pre":  {"iter": it_pre,  "commit": pre_meta["commit"],  "note": pre_meta.get("note", ""),
                  "val_auc": pre_meta["mean_val_auc"],  "val_mcc": pre_meta["mean_val_mcc"],
                  "val_auc_std": pre_meta["std_val_auc"], "val_mcc_std": pre_meta["std_val_mcc"]},
        "old":  {"iter": old_it,  "commit": old_meta["commit"],  "note": old_meta.get("note", ""),
                  "val_auc": old_meta["mean_val_auc"],  "val_mcc": old_meta["mean_val_mcc"],
                  "val_auc_std": old_meta["std_val_auc"], "val_mcc_std": old_meta["std_val_mcc"]},
        "post": {"iter": it_post, "commit": post_meta["commit"], "note": post_meta.get("note", ""),
                  "val_auc": post_meta["mean_val_auc"], "val_mcc": post_meta["mean_val_mcc"],
                  "val_auc_std": post_meta["std_val_auc"], "val_mcc_std": post_meta["std_val_mcc"]},
    }
    KEEP_ITERS[name] = collect_keep_iters(cdir, it_post)


def get_value(combo_name, model_label, subset, metric_key, agg):
    """agg in {'ensemble', 'per_seed_mean'}; metric_key in {'roc_auc','mcc'}."""
    if model_label == "GateMol pre":
        block = DATA[combo_name]["pre"][subset]
    elif model_label == "GateMol post":
        block = DATA[combo_name]["post"][subset]
    elif model_label == "GateMol post (old)":
        block = DATA[combo_name]["old_post"][subset]
    else:
        block = DATA[combo_name]["baselines"][model_label]["subsets"][subset]
    if agg == "ensemble":
        return block["ensemble"][metric_key], None
    else:
        return block[f"per_seed_mean_{metric_key}"], block[f"per_seed_std_{metric_key}"]


# Compute deltas to drive section 3 wording programmatically
def summarize_deltas():
    """Return per-subset stats: how many of (combo × metric) improved for each agg."""
    out = {}
    for sub in SUBSETS:
        cell = {"ensemble": {"up": 0, "down": 0, "sum": 0.0, "n": 0},
                "per_seed_mean": {"up": 0, "down": 0, "sum": 0.0, "n": 0}}
        for agg in ("ensemble", "per_seed_mean"):
            for cname, *_ in COMBOS:
                for key in ("roc_auc", "mcc"):
                    pre_v, _ = get_value(cname, "GateMol pre",  sub, key, agg)
                    post_v, _ = get_value(cname, "GateMol post", sub, key, agg)
                    d = post_v - pre_v
                    cell[agg]["sum"] += d
                    cell[agg]["n"] += 1
                    if d > 0:
                        cell[agg]["up"] += 1
                    elif d < 0:
                        cell[agg]["down"] += 1
        out[sub] = cell
    return out


DELTA_STATS = summarize_deltas()


# Markdown content
md_lines = []
md_lines.append("# GateMol-BBB pre vs post (autoresearch, 100-iter) vs baselines — holdout subset 비교")
md_lines.append("")
md_lines.append("**작성일**: 2026-05-19  ")
md_lines.append("**대상**: `autoresearch_combos_v2` 의 3개 combo (combo1/2/3) × 5개 holdout subset × 2개 집계 방식  ")
md_lines.append("**비교군**: GateMol pre (iter1, autoresearch 전) / GateMol post (현재 best, weights/iter<N> 저장본) / baseline 5종 (mlp2/3/4, xgboost, lightgbm)  ")
md_lines.append("**학습 데이터·split**: 모두 `autoresearch_combos_v2` 동일 셋업 (merged pool, scaffold 80/20, 10 seed)  ")
md_lines.append("**autoresearch 진행도**: 3 combo 모두 100 iter까지 진행")
md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## 1. GateMol pre / post 정의")
md_lines.append("")
md_lines.append("두 컬럼 모두 **GateMol-BBB 동일 코드베이스**의 서로 다른 시점입니다. "
                "`post` 는 각 worktree (`combos-v2-combo{1,2,3}`) 의 `results/<combo>/weights/iter<N>/` "
                "에 보존된 현재 best 체크포인트의 iter 입니다.")
md_lines.append("")
md_lines.append("### 1.1 GateMol pre = autoresearch 적용 **전** 시작점 (iter1 `baseline reproduction`)")
md_lines.append("")
md_lines.append("| Combo | iter | commit | val_AUC | val_MCC | note |")
md_lines.append("|---|---|---|---|---|---|")
for name in ("combo1", "combo2", "combo3"):
    m = PRE_POST_META[name]["pre"]
    md_lines.append(f"| {name} | iter{m['iter']} | {m['commit']} | {m['val_auc']:.4f} | {m['val_mcc']:.4f} | {m['note']} |")
md_lines.append("")
md_lines.append("### 1.2 GateMol post = autoresearch 적용 **후** 현재 best (weights/iter<N>/ 저장본)")
md_lines.append("")
md_lines.append("| Combo | iter | commit | val_AUC | val_MCC | note |")
md_lines.append("|---|---|---|---|---|---|")
for name in ("combo1", "combo2", "combo3"):
    m = PRE_POST_META[name]["post"]
    md_lines.append(f"| {name} | iter{m['iter']} | {m['commit']} | {m['val_auc']:.4f} | {m['val_mcc']:.4f} | {m['note']} |")
md_lines.append("")
md_lines.append("이전 보고서 (2026-05-18) 와 달리 3 combo 모두 best 체크포인트의 `final_holdout_eval` 파일이 "
                "이미 저장되어 있어 (`final_holdout_eval.py` 의무화 정책 commit fbbcc41 이후), 별도 대체 iter 없이 "
                "best iter 의 holdout 결과를 그대로 사용합니다.")
md_lines.append("")
md_lines.append("### 1.3 pre → post 사이 누적 적용된 keep iter")
md_lines.append("")
md_lines.append("```")
for name in ("combo1", "combo2", "combo3"):
    iters = KEEP_ITERS[name]
    # show as "1 -> rest"
    rest = ", ".join(str(i) for i in iters[1:])
    md_lines.append(f"{name}: iter{iters[0]} -> {rest}")
md_lines.append("```")
md_lines.append("")
md_lines.append("주요 변경 (combo별 후반부 추가 keep):")
md_lines.append("- combo1: R-Drop consistency reg, mod_drop / head dropout / AdamW wd 조정, DROP_V_PATH fine sweep")
md_lines.append("- combo2: quad-pool 위에 final LayerNorm, per-pool functional/learnable LN, multi-head attention pool (n_heads=2)")
md_lines.append("- combo3: SGU learnable gate scale, CLS token pooling, EMA decay fine sweep (0.88 → 0.83)")
md_lines.append("")
md_lines.append("### 1.4 val 지표 변화 (pre / post-old / post-new, 10-seed mean ± std)")
md_lines.append("")
md_lines.append("이전 보고서의 `post-old` (iter38/60/54) 와 본 보고서의 `post-new` (iter93/100/90) 사이 차이를 같이 보여주기 위해 3-state 표로 정리합니다.")
md_lines.append("")
md_lines.append("| Combo | metric | pre | post-old | post-new | Δ(new−old) |")
md_lines.append("|---|---|---|---|---|---|")
for name in ("combo1", "combo2", "combo3"):
    pre = PRE_POST_META[name]["pre"]
    old = PRE_POST_META[name]["old"]
    post = PRE_POST_META[name]["post"]
    for met, label in [("val_auc", "AUC"), ("val_mcc", "MCC")]:
        std = met + "_std"
        d = post[met] - old[met]
        md_lines.append(
            f"| {name} | {label} | {pre[met]:.4f}±{pre[std]:.4f} | "
            f"{old[met]:.4f}±{old[std]:.4f} | {post[met]:.4f}±{post[std]:.4f} | {d:+.4f} |"
        )
md_lines.append("")
md_lines.append("→ val_AUC 는 3 combo 모두 미세 상승 (Δ < 1σ). val_MCC 는 combo1/3 가 사실상 flat~소폭 후퇴 (Δ ≈ −0.0006, −0.0005). autoresearch 의 keep gate 가 val_AUC 만 보기 때문에 val_MCC 는 따라오지 않음.")
md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## 2. 5개 holdout subset 비교표 (ensemble · per-seed mean)")
md_lines.append("")
md_lines.append("- ★ = 해당 (combo × metric) 1위")
md_lines.append("- Δ = GateMol-BBB post − pre")
md_lines.append("- per-seed mean 셀은 `mean ± std` (std는 10 seed 표본분산, ddof=0)")
md_lines.append("")


def render_subset_section(subset):
    n = SUBSET_N[subset]
    md = []
    md.append(f"### 2.{SUBSETS.index(subset)+1} `{subset}` subset (n={n})")
    md.append("")

    for agg_name, agg_label in [("ensemble", "ensemble"), ("per_seed_mean", "per-seed mean")]:
        md.append(f"#### {subset} — {agg_label} 기준")
        md.append("")
        md.append("| Combo | metric | GateMol pre | GateMol post | mlp2 | mlp3 | mlp4 | xgb | lgbm | Δ |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for name in ("combo1", "combo2", "combo3"):
            for metric_label, key in [("AUC", "roc_auc"), ("MCC", "mcc")]:
                cells_v = []
                for lbl in MODEL_ORDER:
                    v, std = get_value(name, lbl, subset, key, agg_name)
                    cells_v.append((lbl, v, std))
                max_v = max(v for _, v, _ in cells_v)

                def fmt(lbl, v, std):
                    if std is not None:
                        s = f"{v:.4f}±{std:.4f}"
                    else:
                        s = f"{v:.4f}"
                    if v == max_v:
                        s = f"**★{s}**"
                    return s
                pre_v = cells_v[0][1]
                post_v = cells_v[1][1]
                d = post_v - pre_v
                row = (
                    f"| {name} | {metric_label} | "
                    + " | ".join(fmt(lbl, v, std) for lbl, v, std in cells_v)
                    + f" | {d:+.4f} |"
                )
                md.append(row)
        md.append("")
    return md


for s in SUBSETS:
    md_lines.extend(render_subset_section(s))

md_lines.append("---")
md_lines.append("")
md_lines.append("## 3. 종합 패턴 요약")
md_lines.append("")
md_lines.append("| Subset | ensemble: Δ post-pre (up / down out of 6) | per-seed: Δ post-pre (up / down out of 6) |")
md_lines.append("|---|---|---|")
for sub in SUBSETS:
    e = DELTA_STATS[sub]["ensemble"]
    p = DELTA_STATS[sub]["per_seed_mean"]
    md_lines.append(
        f"| {sub} | up={e['up']} down={e['down']}, mean Δ={e['sum']/e['n']:+.4f} "
        f"| up={p['up']} down={p['down']}, mean Δ={p['sum']/p['n']:+.4f} |"
    )
md_lines.append("")
md_lines.append("### 결정적 패턴 (이전 보고서 패턴과의 비교)")
md_lines.append("")
md_lines.append("- **per-seed 향상은 더 견고해짐**: 100-iter 시점에서 per-seed mean Δ 는 작은 표본인 `nn03` (n=39) 를 제외한 "
                "모든 subset 에서 양수. external/total 은 6/6, internal/nn05 는 5/6 항목이 향상. "
                "특히 total / external 의 per-seed AUC 와 MCC 는 GateMol post 가 ★1위.")
md_lines.append("- **ensemble bonus 의 dilution 은 오히려 더 심화**: 이전 보고서에서 관찰된 \"ensemble 미세 후퇴\" 패턴이 "
                "100-iter 시점에도 유지되며, 일부 subset (`nn03`, `total`) 은 6/6 항목 모두 후퇴. "
                "후반 keep (R-Drop / per-pool LN / EMA decay sweep / multi-head attn pool) 이 개별 모델 품질을 더 끌어올렸지만 "
                "동시에 seed 간 예측 다양성을 더 줄여 ensemble 시너지가 약해진 결과로 보임.")
md_lines.append("- **운영 metric 선택의 중요성 (재확인)**: 단일 모델 운영이면 autoresearch 는 더 큰 폭으로 성공, "
                "10-seed soft-vote 운영이면 효과가 더 희석됨. 이전 보고서 결론이 후반 50 iter 누적 후에도 동일하게 유지됨.")
md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## 4. 시각화")
md_lines.append("")
md_lines.append("![Ensemble basis — 5 subsets × 2 metrics](figures/grid_ensemble_100.png)")
md_lines.append("")
md_lines.append("*Figure 1: Ensemble (soft-voting) 기준 — 5 subsets × 2 metrics. 3 combo 묶음, 7 모델 비교.*")
md_lines.append("")
md_lines.append("![Per-seed mean basis — 5 subsets × 2 metrics](figures/grid_perseed_100.png)")
md_lines.append("")
md_lines.append("*Figure 2: Per-seed mean (±std error bar) 기준 — 동일 레이아웃.*")
md_lines.append("")
md_lines.append("![autoresearch Δ post − pre by subset](figures/delta_post_pre_100.png)")
md_lines.append("")
md_lines.append("*Figure 3: autoresearch 가 GateMol-BBB 의 subset 별 metric 에 미친 변화량 heatmap. 빨강 = 후퇴, 파랑 = 향상.*")
md_lines.append("")
md_lines.append("![total subset — ensemble vs per-seed comparison](figures/total_compare_100.png)")
md_lines.append("")
md_lines.append("*Figure 4: total subset 집중 비교 — 같은 데이터를 두 집계 방식으로 그리면 결론이 달라지는 지점이 보임.*")
md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## 5. 데이터 출처")
md_lines.append("")
md_lines.append("- GateMol pre/post: `combos-v2-combo{1,2,3}/results/<combo_str>/holdout_eval/iter<N>_*.json`")
md_lines.append("- post 체크포인트: `combos-v2-combo{1,2,3}/results/<combo_str>/weights/iter<N>/seed*.pt` "
                "(combo1=iter93, combo2=iter100, combo3=iter90)")
md_lines.append("- baselines: `autoresearch_combos_v2/results/<combo_str>/baselines/{mlp2,mlp3,mlp4,xgboost,lightgbm}.json` "
                "(2026-05-17 실행분, 재사용)")
md_lines.append("- 학습 데이터: `/home/minji/feature_cache_merged/pool.npz` (internal_remaining + external_remaining, 9786개, scaffold 80/20)")
md_lines.append("- holdout subset: `/home/minji/holdout_subset/merged_holdout_10pct_seed42_simfilter09_{internal,external,nn03,nn05,total}/`")
md_lines.append("")

# Section 6: per-seed single-model deployment deep-dive
md_lines.append("---")
md_lines.append("")
md_lines.append("## 6. per-seed (single-model 배포 관점) 심층 분석")
md_lines.append("")
md_lines.append(
    "이전 보고서 (`pre_post_baseline_comparison.md`, 2026-05-18, post=iter38/60/54) 와 본 보고서 "
    "(post=iter93/100/90) 의 차이는 사실상 \"마지막 ~40 keep iter 가 per-seed 에 얼마를 더 보탰는가\" 와 동치입니다. "
    "10-seed soft-vote ensemble 이 아니라 단일 모델 (= 학습된 1개 seed 의 모델) 을 배포하는 시나리오에서는 "
    "per-seed mean ± std 가 배포 시 평균 성능 ± 변동의 직접 추정치이므로 이 섹션은 그 관점에서 결론을 정리합니다."
)
md_lines.append("")
md_lines.append(
    "**선결론 — 모든 combo 에서 post-new 가 post-old 보다 좋아진 것은 아닙니다.**"
)
md_lines.append("")
md_lines.append(
    "- internal 은 in-domain 이라 배포 의사결정 가중치가 낮고, nn03 은 n=39 로 표본이 너무 작아 통계적 의미가 약합니다. "
    "따라서 \"좋아졌다\" 의 판단 기준은 **external (n=709) / nn05 (n=329) / total (n=888)** 의 per-seed AUC 와 MCC 입니다. "
    "이 6 개 셀에서 Δ(new − old) 의 부호와 |Δ| vs std 비율로 판정합니다."
)
md_lines.append("")
md_lines.append("### 6.1 Δ(post-new − post-old) — 마지막 ~40 iter 의 한계 효용 (per-seed mean)")
md_lines.append("")
md_lines.append("| Combo | metric | internal | external | nn03 | **nn05** | **total** |")
md_lines.append("|---|---|---|---|---|---|---|")
for name in ("combo1", "combo2", "combo3"):
    for met_label, key in [("AUC", "roc_auc"), ("MCC", "mcc")]:
        cells = []
        for sub in SUBSETS:
            old_v, _ = get_value(name, "GateMol post (old)", sub, key, "per_seed_mean")
            new_v, _ = get_value(name, "GateMol post",       sub, key, "per_seed_mean")
            d = new_v - old_v
            cells.append(f"{d:+.4f}")
        md_lines.append(f"| {name} (iter{OLD_POST_ITER[name]}→{dict([(n, i) for n,_,_,i,_ in COMBOS])[name]}) | {met_label} | "
                        + " | ".join(cells) + " |")
md_lines.append("")
md_lines.append("**external / nn05 / total 만 모아 본 6 개 핵심 셀:**")
md_lines.append("")
md_lines.append("| Combo | AUC external | AUC nn05 | AUC total | MCC external | MCC nn05 | MCC total |")
md_lines.append("|---|---|---|---|---|---|---|")
for name in ("combo1", "combo2", "combo3"):
    cells = []
    for met_label, key in [("AUC", "roc_auc"), ("MCC", "mcc")]:
        for sub in ("external", "nn05", "total"):
            old_v, _ = get_value(name, "GateMol post (old)", sub, key, "per_seed_mean")
            new_v, _ = get_value(name, "GateMol post",       sub, key, "per_seed_mean")
            d = new_v - old_v
            cells.append(f"{d:+.4f}")
    md_lines.append(f"| {name} | " + " | ".join(cells) + " |")
md_lines.append("")
md_lines.append("**combo 별 판정 (external/nn05/total 6 셀 기준)**")
md_lines.append("")
md_lines.append("- **combo1 (iter38 → iter93, +55 iter): 좋아짐 ✓** — 6 셀 모두 양수. 특히 `nn05` MCC +0.0181 은 post-new std (0.0237) 대비 약 0.77σ 폭으로, 단순 noise 로 보기 어려움. `nn05` AUC +0.0058, `total` AUC +0.0025, `total` MCC +0.0014 까지 holdout 의 핵심 분포에서 단조 향상. R-Drop / mod_drop / DROP_V_PATH fine sweep 이 실제로 옮겨감.")
md_lines.append("- **combo2 (iter60 → iter100, +40 iter): plateau ≈** — 6 셀 |Δ| ≤ 0.004, 모두 1σ 미만. external MCC +0.0006 외에는 0 또는 음수 (AUC 3개 셀 모두 −0.0003 ~ −0.0005). 후반 keep (per-pool LN / multi-head attn pool) 은 val_AUC 0.8503 을 유지만 했고 holdout 에는 이동 없음.")
md_lines.append("- **combo3 (iter54 → iter90, +36 iter): MCC 후퇴 ✗** — AUC 는 external/total 미세 양수, nn05 −0.0039. MCC 는 3 셀 모두 음수 (external −0.0039, **nn05 −0.0220 ≈ 1σ**, total −0.0029). val_AUC 만 +0.0011 올랐을 뿐 holdout MCC 분포는 분명히 더 나빠짐. EMA decay 0.88→0.83 fine sweep 이 val 에 과적합한 정황.")
md_lines.append("")
md_lines.append("즉 후반 ~40 iter 가 holdout 으로 transfer 된 것은 **combo1 하나** 입니다. "
                "combo2 는 변화 없음, combo3 는 MCC 가 후퇴. autoresearch keep gate 가 val_AUC tie 도 keep 하는 정책이 "
                "후반에 val 과적합으로 흘러간 모습으로, 후속 run 에서는 holdout per-seed 변화를 보조 평가 metric 으로 두는 것을 권장합니다.")
md_lines.append("")
md_lines.append("### 6.2 post-new vs 가장 강한 baseline (per-seed gap)")
md_lines.append("")
md_lines.append("각 (combo × subset × metric) 셀에서 5개 baseline 중 per-seed mean 1위 와 GateMol post (new) 의 차이를 보고합니다. 양수면 GateMol post 가 우위.")
md_lines.append("")
md_lines.append("| Combo | metric | internal | external | nn03 | nn05 | total |")
md_lines.append("|---|---|---|---|---|---|---|")
for name in ("combo1", "combo2", "combo3"):
    for met_label, key in [("AUC", "roc_auc"), ("MCC", "mcc")]:
        cells = []
        for sub in SUBSETS:
            post_v, _ = get_value(name, "GateMol post", sub, key, "per_seed_mean")
            best_b_v = -1.0
            best_b_name = ""
            for b in BASELINES:
                v, _ = get_value(name, b, sub, key, "per_seed_mean")
                if v > best_b_v:
                    best_b_v = v
                    best_b_name = b
            gap = post_v - best_b_v
            cells.append(f"{gap:+.4f} ({best_b_name})")
        md_lines.append(f"| {name} | {met_label} | " + " | ".join(cells) + " |")
md_lines.append("")
md_lines.append("- **external (n=709) + nn05 (n=329) + total (n=888)** — 가장 큰 분포 3개에서 GateMol post 가 AUC 기준 모든 combo 에서 +0.015~+0.023 우위. MCC 기준은 combo1/2 가 모두 ★우위, combo3 만 nn05/total 에서 lgbm 에 소폭 (-0.04~-0.01) 뒤짐.")
md_lines.append("- **internal (n=179)** — 모든 combo 에서 mlp 계열에 -0.01~-0.03 패. internal 은 학습 분포와 가장 가까운 \"쉬운\" subset 이라 MLP 계열 깊이가 잘 먹는 분포.")
md_lines.append("- **nn03 (n=39)** — 표본이 작아 std 가 0.02~0.10 수준으로 큼. 어느 모델이 1위인지에 통계적 의미가 약함. 배포 결정의 가중치를 크게 두기 어렵다고 봄.")
md_lines.append("")
md_lines.append("### 6.3 single-model 배포 후보 — `total` (n=888) per-seed 기준")
md_lines.append("")
md_lines.append(
    "각 combo 의 post-new 를 baseline 5종 과 한 풀에 놓고 ranking (6 모델 중 몇 위) 을 매기고, "
    "후반 ~40 iter 가 total 에서 얼마를 더 가져왔는지 (Δ(new−old)) 를 함께 표시합니다."
)
md_lines.append("")
md_lines.append(
    "| Combo (post iter) | total AUC (per-seed) | total MCC (per-seed) | total AUC rank | total MCC rank | Δ total AUC (new−old) | Δ total MCC (new−old) |"
)
md_lines.append("|---|---|---|---|---|---|---|")
for name in ("combo1", "combo2", "combo3"):
    it_post = dict([(n, i) for n,_,_,i,_ in COMBOS])[name]
    a, asd = get_value(name, "GateMol post", "total", "roc_auc", "per_seed_mean")
    m, msd = get_value(name, "GateMol post", "total", "mcc",     "per_seed_mean")
    a_old, _ = get_value(name, "GateMol post (old)", "total", "roc_auc", "per_seed_mean")
    m_old, _ = get_value(name, "GateMol post (old)", "total", "mcc",     "per_seed_mean")
    da = a - a_old
    dm = m - m_old
    a_pool = [(lbl, get_value(name, lbl, "total", "roc_auc", "per_seed_mean")[0])
              for lbl in ["GateMol post"] + BASELINES]
    m_pool = [(lbl, get_value(name, lbl, "total", "mcc", "per_seed_mean")[0])
              for lbl in ["GateMol post"] + BASELINES]
    a_pool.sort(key=lambda x: -x[1])
    m_pool.sort(key=lambda x: -x[1])
    a_rank = 1 + [lbl for lbl, _ in a_pool].index("GateMol post")
    m_rank = 1 + [lbl for lbl, _ in m_pool].index("GateMol post")
    md_lines.append(
        f"| {name} (iter{it_post}) | {a:.4f} ± {asd:.4f} | {m:.4f} ± {msd:.4f} | "
        f"{a_rank}/6 | {m_rank}/6 | {da:+.4f} | {dm:+.4f} |"
    )
md_lines.append("")
md_lines.append(
    "- **combo1 (iter93)** — total AUC 1/6, total MCC 1/6. "
    "Δ(new−old) 는 AUC +0.0025, **MCC +0.0014 — 후반 keep 이 holdout 에 실제로 옮겨간 유일한 combo**. "
    "MCC 가 매우 중요한 응용 (BBB 분류 cutoff) 이면 명확한 1순위."
)
md_lines.append(
    "- **combo2 (iter100)** — total AUC 1/6 (0.8775, std 0.0038 으로 가장 안정적), total MCC 1/6 (0.6011). "
    "단, Δ(new−old) AUC −0.0005, MCC −0.0002 로 iter60 시점과 통계적 동일. "
    "iter100 의 multi-head attn pool 자체가 의미 있는 progress 는 아니므로, **combo2 만 쓸 거면 iter60 도 동급**."
)
md_lines.append(
    "- **combo3 (iter90)** — total AUC 1/6 (0.8762) 이지만 total MCC 2/6 (lgbm 에 −0.0092 뒤짐). "
    "Δ(new−old) AUC +0.0010, MCC −0.0029 로 MCC 측 후퇴. single-model 운영 우선순위 하위."
)
md_lines.append("")
md_lines.append("### 6.4 결론 — single-model 배포로 갈 때")
md_lines.append("")
md_lines.append(
    "1. **post-new 가 post-old 보다 holdout 에서 진짜로 좋아진 것은 combo1 하나**: external/nn05/total 6 셀 모두 양수. "
    "combo2 는 plateau, combo3 는 MCC 후퇴. val_AUC 만으로 후반 keep 을 평가했기 때문에 combo2/3 는 val 에 과적합한 정황. "
    "후속 autoresearch run 에서는 holdout per-seed 변화를 보조 평가 metric 으로 도입하는 것을 권장."
)
md_lines.append(
    "2. **GateMol post(new) 의 per-seed 우위는 baseline 대비로는 여전히 견고**: "
    "external/nn05/total 에서 강한 baseline (lgbm) 대비 AUC +0.015~+0.023 우위, "
    "MCC 도 combo1/2 우위 (combo3 만 nn05/total 에서 lgbm 에 소폭 패). "
    "ensemble bonus 가 만들던 \"baseline 이 더 좋아 보이는\" 환상은 사라짐."
)
md_lines.append(
    "3. **배포 1순위 추천 — combo1 iter93** (DROP_V_PATH 0.10→0.08). "
    "total per-seed AUC 0.8756 (rank 1/6) + MCC 0.6088 (rank 1/6), 그리고 후반 ~55 iter 가 holdout 으로 transfer 된 유일한 combo. "
    "robustness 우선이면 total AUC 가 0.0019 더 높고 std 가 약간 작은 combo2 iter100 도 후보지만, "
    "iter60 대비 iter100 의 progress 는 noise 수준이라 굳이 iter100 을 고집할 이유는 없습니다."
)
md_lines.append("")
md_lines.append("![per-seed single-model deep-dive](figures/perseed_singlemodel_100.png)")
md_lines.append("")
md_lines.append("*Figure 5: (a) per-seed mean Δ(post-new − post-old) — 마지막 ~40 keep iter 의 추가 효용 heatmap. "
                "(b) per-seed mean (post-new − best baseline) — GateMol post 가 deployment race 에서 어느 subset 을 이기는지. "
                "(c) `total` (n=888) per-seed mean ± std 막대그래프 — 7개 모델 × 3 combo.*")
md_lines.append("")

out_md = OUT_DIR / "pre_post_baseline_comparison_100.md"
out_md.write_text("\n".join(md_lines), encoding="utf-8")
print(f"[md ] wrote {out_md}")


# Figures
def grid_figure(agg, fname, with_errorbars):
    fig, axes = plt.subplots(5, 2, figsize=(15, 18), sharex=False)

    for r, subset in enumerate(SUBSETS):
        for c, (metric_label, key) in enumerate([("AUC", "roc_auc"), ("MCC", "mcc")]):
            ax = axes[r, c]
            x_pos = np.arange(len(COMBOS))
            bar_w = 0.11
            for mi, lbl in enumerate(MODEL_ORDER):
                vals, errs = [], []
                for ci, (cname, *_rest) in enumerate(COMBOS):
                    v, std = get_value(cname, lbl, subset, key, agg)
                    vals.append(v)
                    errs.append(std if std is not None else 0)
                offset = (mi - (len(MODEL_ORDER)-1)/2) * bar_w
                kw = dict(width=bar_w, color=MODEL_COLORS[lbl],
                          edgecolor="black", linewidth=0.4, label=lbl)
                if with_errorbars:
                    ax.bar(x_pos + offset, vals, yerr=errs, capsize=2,
                           error_kw=dict(elinewidth=0.6), **kw)
                else:
                    ax.bar(x_pos + offset, vals, **kw)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([c[0] for c in COMBOS])
            ax.set_title(f"{subset} (n={SUBSET_N[subset]}) — {metric_label}", fontsize=10)
            if metric_label == "AUC":
                ax.set_ylim(0.78, 0.93)
            else:
                ax.set_ylim(0.35, 0.80)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.set_ylabel(metric_label)
            if r == 0 and c == 1:
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.95)

    fig.suptitle(f"Holdout subset metrics — {agg.replace('_', ' ')} basis (100-iter snapshot)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.92, 0.985])
    fig.savefig(FIG_DIR / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {FIG_DIR / fname}")


grid_figure("ensemble",      "grid_ensemble_100.png", with_errorbars=False)
grid_figure("per_seed_mean", "grid_perseed_100.png",  with_errorbars=True)


def delta_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    row_labels = []
    for cname, _cs, *_rest in COMBOS:
        for met in ("AUC", "MCC"):
            row_labels.append(f"{cname} {met}")

    for ax, (agg, title) in zip(axes, [("ensemble", "Ensemble"),
                                        ("per_seed_mean", "Per-seed mean")]):
        mat = np.zeros((6, len(SUBSETS)))
        for ri, (cname, _cs, *_rest) in enumerate(COMBOS):
            for mi, (met_label, key) in enumerate([("AUC", "roc_auc"), ("MCC", "mcc")]):
                for ci, sub in enumerate(SUBSETS):
                    pre_v, _ = get_value(cname, "GateMol pre",  sub, key, agg)
                    post_v, _ = get_value(cname, "GateMol post", sub, key, agg)
                    mat[ri * 2 + mi, ci] = post_v - pre_v
        vmax = max(0.05, np.abs(mat).max())
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(SUBSETS)))
        ax.set_xticklabels(SUBSETS, rotation=0)
        ax.set_yticks(range(6))
        ax.set_yticklabels(row_labels)
        ax.set_title(f"{title} — Δ(post − pre)")
        for i in range(6):
            for j in range(len(SUBSETS)):
                ax.text(j, i, f"{mat[i,j]:+.3f}", ha="center", va="center",
                        fontsize=8,
                        color="black" if abs(mat[i,j]) < vmax*0.65 else "white")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    fig.suptitle("autoresearch effect (100-iter): GateMol-BBB Δ(post − pre) by subset × metric × combo",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "delta_post_pre_100.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {FIG_DIR / 'delta_post_pre_100.png'}")


delta_heatmap()


def total_compare():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    bar_w = 0.11
    x_pos = np.arange(len(COMBOS))
    for ci, (metric_label, key) in enumerate([("AUC", "roc_auc"), ("MCC", "mcc")]):
        for ri, (agg, title) in enumerate([("ensemble", "Ensemble"),
                                            ("per_seed_mean", "Per-seed mean")]):
            ax = axes[ri, ci]
            for mi, lbl in enumerate(MODEL_ORDER):
                vals, errs = [], []
                for cname, *_rest in COMBOS:
                    v, std = get_value(cname, lbl, "total", key, agg)
                    vals.append(v)
                    errs.append(std if std is not None else 0)
                offset = (mi - (len(MODEL_ORDER)-1)/2) * bar_w
                kw = dict(width=bar_w, color=MODEL_COLORS[lbl],
                          edgecolor="black", linewidth=0.4, label=lbl)
                if agg == "per_seed_mean":
                    ax.bar(x_pos + offset, vals, yerr=errs, capsize=2,
                           error_kw=dict(elinewidth=0.6), **kw)
                else:
                    ax.bar(x_pos + offset, vals, **kw)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([c[0] for c in COMBOS])
            ax.set_title(f"{title} — total {metric_label}")
            ax.set_ylabel(metric_label)
            if metric_label == "AUC":
                ax.set_ylim(0.82, 0.91)
            else:
                ax.set_ylim(0.45, 0.66)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.95)
    fig.suptitle("Total subset (100-iter) — ensemble vs per-seed mean (3 combos × 7 models)",
                 fontsize=13, y=1.00)
    fig.tight_layout(rect=[0, 0, 0.92, 0.985])
    fig.savefig(FIG_DIR / "total_compare_100.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {FIG_DIR / 'total_compare_100.png'}")


total_compare()


def perseed_singlemodel_figure():
    """3-panel figure for section 6:
       (top-left)  Δ(post-new − post-old) per-seed heatmap (6×5)
       (top-right) post-new − best baseline per-seed heatmap (6×5)
       (bottom)    total subset per-seed bars: 3 combos × 7 models with errorbars.
    """
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.05], hspace=0.45, wspace=0.30)

    row_labels = []
    for cname, *_ in COMBOS:
        for met in ("AUC", "MCC"):
            row_labels.append(f"{cname} {met}")

    # (a) Δ(post-new − post-old)
    ax = fig.add_subplot(gs[0, 0])
    mat_a = np.zeros((6, len(SUBSETS)))
    for ri, (cname, *_) in enumerate(COMBOS):
        for mi, (met_label, key) in enumerate([("AUC", "roc_auc"), ("MCC", "mcc")]):
            for ci, sub in enumerate(SUBSETS):
                old_v, _ = get_value(cname, "GateMol post (old)", sub, key, "per_seed_mean")
                new_v, _ = get_value(cname, "GateMol post",       sub, key, "per_seed_mean")
                mat_a[ri * 2 + mi, ci] = new_v - old_v
    vmax_a = max(0.02, np.abs(mat_a).max())
    im = ax.imshow(mat_a, cmap="RdBu_r", vmin=-vmax_a, vmax=vmax_a, aspect="auto")
    ax.set_xticks(range(len(SUBSETS)))
    ax.set_xticklabels(SUBSETS)
    ax.set_yticks(range(6))
    ax.set_yticklabels(row_labels)
    ax.set_title("(a) per-seed mean Δ(post-new − post-old)\nmarginal gain of last ~40 keep iter")
    for i in range(6):
        for j in range(len(SUBSETS)):
            ax.text(j, i, f"{mat_a[i,j]:+.3f}", ha="center", va="center",
                    fontsize=8,
                    color="black" if abs(mat_a[i,j]) < vmax_a*0.65 else "white")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    # (b) post-new − best baseline (per-seed)
    ax = fig.add_subplot(gs[0, 1])
    mat_b = np.zeros((6, len(SUBSETS)))
    for ri, (cname, *_) in enumerate(COMBOS):
        for mi, (met_label, key) in enumerate([("AUC", "roc_auc"), ("MCC", "mcc")]):
            for ci, sub in enumerate(SUBSETS):
                post_v, _ = get_value(cname, "GateMol post", sub, key, "per_seed_mean")
                best_bv = max(get_value(cname, b, sub, key, "per_seed_mean")[0] for b in BASELINES)
                mat_b[ri * 2 + mi, ci] = post_v - best_bv
    vmax_b = max(0.03, np.abs(mat_b).max())
    im = ax.imshow(mat_b, cmap="RdBu_r", vmin=-vmax_b, vmax=vmax_b, aspect="auto")
    ax.set_xticks(range(len(SUBSETS)))
    ax.set_xticklabels(SUBSETS)
    ax.set_yticks(range(6))
    ax.set_yticklabels(row_labels)
    ax.set_title("(b) per-seed mean (GateMol post − best baseline)\ndeployment-race gap, positive ⇒ GateMol wins")
    for i in range(6):
        for j in range(len(SUBSETS)):
            ax.text(j, i, f"{mat_b[i,j]:+.3f}", ha="center", va="center",
                    fontsize=8,
                    color="black" if abs(mat_b[i,j]) < vmax_b*0.65 else "white")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    # (c) total subset per-seed bars: pre / post-old / post-new + 5 baselines
    ax = fig.add_subplot(gs[1, :])
    models = ["GateMol pre", "GateMol post (old)", "GateMol post"] + BASELINES
    display_label = {
        "GateMol pre":         "GateMol pre",
        "GateMol post (old)":  "GateMol post-old",
        "GateMol post":        "GateMol post-new",
    }
    model_colors = {
        "GateMol pre":         "#9ecae1",
        "GateMol post (old)":  "#3182bd",
        "GateMol post":        "#08306b",
        "mlp2":                "#ffbb78",
        "mlp3":                "#ff7f0e",
        "mlp4":                "#d62728",
        "xgboost":             "#2ca02c",
        "lightgbm":            "#8c564b",
    }
    n_models = len(models)
    bar_w = 0.10
    x_pos = np.arange(len(COMBOS))
    # AUC and MCC overlaid? No — split horizontally as two subplots inside gs[1,:]
    # Easier: 2 columns inside the bottom row
    ax.set_visible(False)
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], wspace=0.30)

    last_sub_ax = None
    for ci, (metric_label, key) in enumerate([("AUC", "roc_auc"), ("MCC", "mcc")]):
        sub_ax = fig.add_subplot(inner[0, ci])
        for mi, lbl in enumerate(models):
            vals, errs = [], []
            for cname, *_ in COMBOS:
                v, std = get_value(cname, lbl, "total", key, "per_seed_mean")
                vals.append(v)
                errs.append(std if std is not None else 0)
            offset = (mi - (n_models-1)/2) * bar_w
            sub_ax.bar(x_pos + offset, vals, yerr=errs, capsize=2,
                       width=bar_w, color=model_colors[lbl],
                       edgecolor="black", linewidth=0.4,
                       label=display_label.get(lbl, lbl),
                       error_kw=dict(elinewidth=0.6))
        sub_ax.set_xticks(x_pos)
        sub_ax.set_xticklabels([c[0] for c in COMBOS])
        sub_ax.set_title(f"(c) total (n={SUBSET_N['total']}) per-seed {metric_label}")
        sub_ax.set_ylabel(metric_label)
        if metric_label == "AUC":
            sub_ax.set_ylim(0.82, 0.90)
        else:
            sub_ax.set_ylim(0.50, 0.65)
        sub_ax.grid(axis="y", linestyle=":", alpha=0.5)
        last_sub_ax = sub_ax

    handles, labels = last_sub_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models),
               bbox_to_anchor=(0.5, -0.02), framealpha=0.95, fontsize=9,
               handlelength=1.5, columnspacing=1.2)

    fig.suptitle("Per-seed (single-model deployment) deep-dive — 100-iter snapshot",
                 fontsize=13, y=0.995)
    fig.savefig(FIG_DIR / "perseed_singlemodel_100.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {FIG_DIR / 'perseed_singlemodel_100.png'}")


perseed_singlemodel_figure()

print("[done]")
