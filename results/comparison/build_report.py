#!/usr/bin/env python3
"""Build pre/post-autoresearch vs baseline comparison report (md + figures).

Generates:
  - pre_post_baseline_comparison.md
  - figures/ensemble_<subset>.png  (5 subsets)
  - figures/perseed_<subset>.png   (5 subsets)
  - figures/delta_post_pre.png     (autoresearch effect heatmap)
  - figures/total_ensemble_vs_perseed.png
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

COMBOS = [
    ("combo1", "maccs+avalon+scage2+mole",     1, 38, 53, "/home/minji/combos-v2-combo1"),
    ("combo2", "maccs+scage1+mole",             1, 60, 60, "/home/minji/combos-v2-combo2"),
    ("combo3", "maccs+scage1+scage2+mole",      1, 54, 54, "/home/minji/combos-v2-combo3"),
]
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


# ── Load all data ───────────────────────────────────────────────────────────
DATA = {}
PRE_POST_META = {}
for name, combo_str, it_pre, it_post_h, it_post_last, wt in COMBOS:
    bdir = ROOT / "results" / combo_str / "baselines"
    hd = Path(wt) / "results" / combo_str / "holdout_eval"
    pre_h  = json.load(open(find_holdout(hd, it_pre)))
    post_h = json.load(open(find_holdout(hd, it_post_h)))
    bjsons = {m: json.load(open(bdir / f"{m}.json")) for m in BASELINES}
    DATA[name] = {
        "combo_str": combo_str,
        "pre":  pre_h["summary"]["subsets"],
        "post": post_h["summary"]["subsets"],
        "pre_val":  pre_h["summary"]["val"],
        "post_val": post_h["summary"]["val"],
        "baselines": {m: bjsons[m]["summary"] for m in BASELINES},
    }
    # also pull notes from the iter_per_seed.json for clarity
    per_seed_fp_pre  = Path(wt) / "results" / combo_str / f"iter{it_pre:04d}_per_seed.json"
    per_seed_fp_post_h = Path(wt) / "results" / combo_str / f"iter{it_post_h:04d}_per_seed.json"
    per_seed_fp_post_last = Path(wt) / "results" / combo_str / f"iter{it_post_last:04d}_per_seed.json"
    pre_meta  = json.load(open(per_seed_fp_pre))["summary"]
    post_meta = json.load(open(per_seed_fp_post_h))["summary"]
    last_meta = json.load(open(per_seed_fp_post_last))["summary"]
    PRE_POST_META[name] = {
        "combo_str": combo_str,
        "pre":  {"iter": it_pre,  "commit": pre_meta["commit"],  "note": pre_meta.get("note", ""),
                  "val_auc": pre_meta["mean_val_auc"],  "val_mcc": pre_meta["mean_val_mcc"]},
        "post_for_holdout": {"iter": it_post_h, "commit": post_meta["commit"], "note": post_meta.get("note",""),
                              "val_auc": post_meta["mean_val_auc"], "val_mcc": post_meta["mean_val_mcc"]},
        "post_last_keep":   {"iter": it_post_last, "commit": last_meta["commit"], "note": last_meta.get("note",""),
                              "val_auc": last_meta["mean_val_auc"], "val_mcc": last_meta["mean_val_mcc"]},
    }


# ── Helpers ─────────────────────────────────────────────────────────────────
def get_value(combo_name, model_label, subset, metric_key, agg):
    """agg in {'ensemble', 'per_seed_mean'}; metric_key in {'roc_auc','mcc'}."""
    if model_label == "GateMol pre":
        block = DATA[combo_name]["pre"][subset]
    elif model_label == "GateMol post":
        block = DATA[combo_name]["post"][subset]
    else:
        block = DATA[combo_name]["baselines"][model_label]["subsets"][subset]
    if agg == "ensemble":
        return block["ensemble"][metric_key], None
    else:
        return block[f"per_seed_mean_{metric_key}"], block[f"per_seed_std_{metric_key}"]


# ── Markdown content ────────────────────────────────────────────────────────
md_lines = []
md_lines.append("# GateMol-BBB pre vs post (autoresearch) vs baselines — holdout subset 비교")
md_lines.append("")
md_lines.append("**작성일**: 2026-05-18  ")
md_lines.append("**대상**: `autoresearch_combos_v2` 의 3개 combo (combo1/2/3) × 5개 holdout subset × 2개 집계 방식  ")
md_lines.append("**비교군**: GateMol pre (iter1, autoresearch 전) / GateMol post (마지막 keep iter, autoresearch 후) / baseline 5종 (mlp2/3/4, xgboost, lightgbm)  ")
md_lines.append("**학습 데이터·split**: 모두 `autoresearch_combos_v2` 동일 셋업 (merged pool, scaffold 80/20, 10 seed)")
md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## 1. GateMol pre / post 정의")
md_lines.append("")
md_lines.append("두 컬럼 모두 **GateMol-BBB 동일 코드베이스**의 서로 다른 시점입니다.")
md_lines.append("")
md_lines.append("### 1.1 GateMol pre = autoresearch 적용 **전** 시작점 (iter1 `baseline reproduction`)")
md_lines.append("")
md_lines.append("| Combo | iter | commit | val_AUC | val_MCC | note |")
md_lines.append("|---|---|---|---|---|---|")
for name in ("combo1", "combo2", "combo3"):
    m = PRE_POST_META[name]["pre"]
    md_lines.append(f"| {name} | iter{m['iter']} | {m['commit']} | {m['val_auc']:.4f} | {m['val_mcc']:.4f} | {m['note']} |")
md_lines.append("")
md_lines.append("### 1.2 GateMol post = autoresearch 적용 **후** 최종 상태 (마지막 keep=True iter)")
md_lines.append("")
md_lines.append("| Combo | iter (holdout 평가용) | commit | val_AUC | val_MCC | note |")
md_lines.append("|---|---|---|---|---|---|")
for name in ("combo1", "combo2", "combo3"):
    m = PRE_POST_META[name]["post_for_holdout"]
    md_lines.append(f"| {name} | iter{m['iter']} | {m['commit']} | {m['val_auc']:.4f} | {m['val_mcc']:.4f} | {m['note']} |")
md_lines.append("")
md_lines.append("**combo1 주의**: 실제 마지막 keep은 **iter53** (`shadow patience 10→15`, commit "
                f"{PRE_POST_META['combo1']['post_last_keep']['commit']}) 이지만 `final_holdout_eval.py` 의무화 정책 (commit fbbcc41) 이전에 keep된 iter라 holdout_eval 파일이 없음. "
                "iter53의 val 지표(val_AUC=0.8520, val_MCC=0.5399)가 iter38과 **완전히 동일**하므로 iter38의 holdout 결과로 iter53 상태를 대표하게 함.")
md_lines.append("")
md_lines.append("### 1.3 pre → post 사이 누적 적용된 keep iter")
md_lines.append("")
md_lines.append("```")
md_lines.append("combo1: iter1 → 6, 12, 17, 20, 22, 27, 29, 30, 32, 38, (53)")
md_lines.append("combo2: iter1 → 7, 10, 17, 20, 22, 24, 33, 50, 58, 60")
md_lines.append("combo3: iter1 → 7, 8, 9, 12, 13, 20, 29, 30, 34, 43, 54")
md_lines.append("```")
md_lines.append("")
md_lines.append("주요 변경: modality dropout, multi-head SGU, EMA weights/warmup, AdamW 전환, "
                "cross-modal FiLM/attention pool, grad-clip, quad-pool, patience 조정 등.")
md_lines.append("")
md_lines.append("### 1.4 val 지표 변화 (참고)")
md_lines.append("")
md_lines.append("| Combo | val_AUC pre → post | val_MCC pre → post |")
md_lines.append("|---|---|---|")
for name in ("combo1", "combo2", "combo3"):
    pre = PRE_POST_META[name]["pre"]
    post = PRE_POST_META[name]["post_for_holdout"]
    da = post["val_auc"] - pre["val_auc"]
    dm = post["val_mcc"] - pre["val_mcc"]
    md_lines.append(f"| {name} | {pre['val_auc']:.4f} → {post['val_auc']:.4f} ({da:+.4f}) | "
                    f"{pre['val_mcc']:.4f} → {post['val_mcc']:.4f} ({dm:+.4f}) |")
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
                # rendering
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
md_lines.append("| Subset | ensemble: 누가 강한가 | per-seed mean: 누가 강한가 | autoresearch 효과 (Δ post-pre) |")
md_lines.append("|---|---|---|---|")
md_lines.append("| internal | mlp 계열 AUC/MCC 1위 (모든 combo) | mlp 계열 AUC/MCC 1위 | ensemble 약간 후퇴 / per-seed 약간 향상 |")
md_lines.append("| **external** | GateMol AUC, xgb/lgbm MCC | **GateMol AUC/MCC 거의 1위** | **per-seed 모든 항목 명확 향상** |")
md_lines.append("| nn03 (n=39) | lgbm AUC, 변동 매우 큼 | GateMol AUC (combo2/3), lgbm (combo1) | 작은 표본 → ensemble 임계값 민감, 큰 변동 |")
md_lines.append("| nn05 | GateMol AUC, xgb/lgbm MCC | GateMol AUC, lgbm/GateMol MCC | 일관된 향상 |")
md_lines.append("| **total** | pre가 ensemble AUC 최고, lgbm/xgb MCC | **GateMol post AUC/MCC 거의 1위** | **per-seed 명확 향상 / ensemble 미세 후퇴** |")
md_lines.append("")
md_lines.append("### 결정적 패턴")
md_lines.append("")
md_lines.append("- **ensemble vs per-seed 의 결론 정반대**: GateMol-BBB의 seed 다양성이 작아 ensemble bonus가 거의 0 — 반면 tree 계열은 ensemble bonus가 큼 (xgb 평균 +0.05 on total MCC).")
md_lines.append("- **autoresearch의 성격**: 개별 모델 품질(per-seed mean)은 모든 subset/지표에서 향상, 그러나 ensemble bonus를 줄이는 부작용 → ensemble holdout 지표는 미세 후퇴.")
md_lines.append("- **운영 metric 선택의 중요성**: 단일 모델 운영이면 autoresearch가 성공, 10-seed soft-vote 운영이면 효과가 희석됨.")
md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## 4. 시각화")
md_lines.append("")
md_lines.append("![Ensemble basis — 5 subsets × 2 metrics](figures/grid_ensemble.png)")
md_lines.append("")
md_lines.append("*Figure 1: Ensemble (soft-voting) 기준 — 5 subsets × 2 metrics. 3 combo 묶음, 7 모델 비교.*")
md_lines.append("")
md_lines.append("![Per-seed mean basis — 5 subsets × 2 metrics](figures/grid_perseed.png)")
md_lines.append("")
md_lines.append("*Figure 2: Per-seed mean (±std error bar) 기준 — 동일 레이아웃.*")
md_lines.append("")
md_lines.append("![autoresearch Δ post − pre by subset](figures/delta_post_pre.png)")
md_lines.append("")
md_lines.append("*Figure 3: autoresearch가 GateMol-BBB의 subset별 metric에 미친 변화량 heatmap. 빨강 = 후퇴, 파랑 = 향상.*")
md_lines.append("")
md_lines.append("![total subset — ensemble vs per-seed comparison](figures/total_compare.png)")
md_lines.append("")
md_lines.append("*Figure 4: total subset 집중 비교 — 같은 데이터를 두 집계 방식으로 그리면 결론이 달라지는 지점이 보임.*")
md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## 5. 데이터 출처")
md_lines.append("")
md_lines.append("- GateMol pre/post: `combos-v2-combo{1,2,3}/results/<combo_str>/holdout_eval/iter<N>_*.json`")
md_lines.append("- baselines: `autoresearch_combos_v2/results/<combo_str>/baselines/{mlp2,mlp3,mlp4,xgboost,lightgbm}.json`")
md_lines.append("- 베이스라인 학습/평가 스크립트: `autoresearch_combos_v2/baselines/run_baselines.py` (이미 2026-05-17 실행)")
md_lines.append("- 학습 데이터: `/home/minji/feature_cache_merged/pool.npz` (internal_remaining + external_remaining, 9786개, scaffold 80/20)")
md_lines.append("- holdout subset: `/home/minji/holdout_subset/merged_holdout_10pct_seed42_simfilter09_{internal,external,nn03,nn05,total}/`")
md_lines.append("")

(OUT_DIR / "pre_post_baseline_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")
print(f"[md ] wrote {OUT_DIR / 'pre_post_baseline_comparison.md'}")


# ── Figures ─────────────────────────────────────────────────────────────────
def grid_figure(agg, fname, with_errorbars):
    """5 rows (subsets) × 2 cols (AUC, MCC). Each cell: 3 combo groups of 7 model bars."""
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

    fig.suptitle(f"Holdout subset metrics — {agg.replace('_', ' ')} basis",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.92, 0.985])
    fig.savefig(FIG_DIR / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {FIG_DIR / fname}")


grid_figure("ensemble",      "grid_ensemble.png", with_errorbars=False)
grid_figure("per_seed_mean", "grid_perseed.png",  with_errorbars=True)


# Δ post-pre heatmap
def delta_heatmap():
    """Heatmap: rows = combo×metric (6 rows), cols = subset (5).
    Two side-by-side panels: ensemble vs per-seed mean."""
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

    fig.suptitle("autoresearch effect: GateMol-BBB Δ(post − pre) by subset × metric × combo",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "delta_post_pre.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {FIG_DIR / 'delta_post_pre.png'}")


delta_heatmap()


# Total subset focus
def total_compare():
    """Side-by-side: total AUC and total MCC, two panels (ensemble vs per-seed mean)."""
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
    fig.suptitle("Total subset — ensemble vs per-seed mean (3 combos × 7 models)",
                 fontsize=13, y=1.00)
    fig.tight_layout(rect=[0, 0, 0.92, 0.985])
    fig.savefig(FIG_DIR / "total_compare.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {FIG_DIR / 'total_compare.png'}")


total_compare()

print("[done]")
