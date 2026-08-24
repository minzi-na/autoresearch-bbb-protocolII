"""
iter200 summary table — GateMol post (last-kept iter ≤ 200) vs TabPFN-2.5 baseline.

Layout:
  rows  = combo1·GateMol post / combo1·TabPFN / combo2·… / combo3·…  (6 rows)
  cols  = Combo | Features | Model | Val(AUC,MCC,ACC) | Holdout·nn05(...) | Holdout·total(...)

GateMol post = each combo's last-kept holdout_eval JSON (model state held at iter200
since iter200 itself was not kept for any of the three combos).
  combo1 -> iter197 (commit 9bc22da482)
  combo2 -> iter198 (commit 2e8cbdc9f5)
  combo3 -> iter197 (commit 2c4803d72c)

TabPFN-2.5 = baselines/run_tabpfn.py output for the same combo, same 10 seeds,
same scaffold 8:2 split as the rest of the autoresearch_combos_v2 baselines.

The highest mean per (combo, metric) cell — comparing GateMol post vs TabPFN —
is rendered in bold.
"""

import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# (combo_tag, features, model_label, json_path)
ROWS = [
    ("combo1", "maccs+avalon+scage2+mole", "GateMol post (iter197)",
     "/home/minji/combos-v2-combo1/results/maccs+avalon+scage2+mole/holdout_eval/iter0197_9bc22da482.json"),
    ("combo1", "maccs+avalon+scage2+mole", "TabPFN-2.5",
     "/home/minji/autoresearch_combos_v2/results/maccs+avalon+scage2+mole/baselines/tabpfn.json"),
    ("combo2", "maccs+scage1+mole", "GateMol post (iter198)",
     "/home/minji/combos-v2-combo2/results/maccs+scage1+mole/holdout_eval/iter0198_2e8cbdc9f5.json"),
    ("combo2", "maccs+scage1+mole", "TabPFN-2.5",
     "/home/minji/autoresearch_combos_v2/results/maccs+scage1+mole/baselines/tabpfn.json"),
    ("combo3", "maccs+scage1+scage2+mole", "GateMol post (iter197)",
     "/home/minji/combos-v2-combo3/results/maccs+scage1+scage2+mole/holdout_eval/iter0197_2c4803d72c.json"),
    ("combo3", "maccs+scage1+scage2+mole", "TabPFN-2.5",
     "/home/minji/autoresearch_combos_v2/results/maccs+scage1+scage2+mole/baselines/tabpfn.json"),
]

SECTIONS = [
    ("Val", "val"),
    ("Holdout · nn05", "nn05"),
    ("Holdout · total", "total"),
]
METRICS = [("AUC", "roc_auc"), ("MCC", "mcc"), ("ACC", "accuracy")]

ID_HEADER_BG = "#7F8C8D"   # grey for identifier columns
GROUP_HEADER_BG = "#34495E"  # slate for metric group headers
VAL_HEADER_BG = "#5D6D7E"    # slightly lighter slate for Val
SUB_HEADER_BG = "#FFFFFF"
GRID = "#B0BEC5"
TEXT = "#1A1A1A"
MUTED = "#555"
CELL_LW = 1.1


def mean_std(xs):
    return st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0


def collect(path):
    d = json.load(open(path))
    per_seed = d["per_seed"]
    out = {"val": {}}
    for label, key in METRICS:
        out["val"][label] = mean_std([s["val"][key] for s in per_seed])
    for _, sub in SECTIONS:
        if sub == "val":
            continue
        out[sub] = {}
        for label, key in METRICS:
            out[sub][label] = mean_std([s["subsets"][sub][key] for s in per_seed])
    return out


def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def main():
    # data[i] = collected metrics for ROWS[i]
    data = [collect(path) for *_, path in ROWS]

    # ---- column widths (data units)
    # Generous widths so text never crowds the cell border.
    id_cols = ["Combo", "Features", "Model"]
    id_widths = [1.6, 5.6, 2.6]
    metric_w = 3.2

    col_widths = list(id_widths)
    for _ in SECTIONS:
        col_widths += [metric_w] * len(METRICS)
    total_w = sum(col_widths)
    x_left = [0.0]
    for w in col_widths:
        x_left.append(x_left[-1] + w)

    # ---- row heights
    title_h = 2.4
    group_header_h = 1.0
    sub_header_h = 0.75
    row_h = 0.85
    footer_h = 1.4
    n_rows = len(ROWS)

    total_h = title_h + group_header_h + sub_header_h + n_rows * row_h + footer_h

    # Aspect tuned to match Downloads/image(2).png visual density.
    fig_w = 24.0
    fig_h = fig_w * (total_h / total_w) * 1.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    # ---- title block
    y = total_h
    y -= 0.9
    ax.text(total_w / 2, y,
            "iter200 state · per-seed mean ± std (n=10)",
            ha="center", va="center", fontsize=17, fontweight="bold",
            color=TEXT)
    y -= 0.7
    ax.text(total_w / 2, y,
            "Combo1/2/3 — GateMol post (iter200 state) vs TabPFN-2.5 baseline · "
            "val + holdout subsets (nn05, total)",
            ha="center", va="center", fontsize=11, color=MUTED)
    y -= 0.55
    ax.text(total_w / 2, y,
            "GateMol post = last-kept holdout_eval at iter200 "
            "(combo1=iter197, combo2=iter198, combo3=iter197) · "
            "TabPFN-2.5 = baselines/tabpfn.json (same pool, same scaffold 8:2, same 10 seeds)",
            ha="center", va="center", fontsize=10, color=MUTED)

    # ---- group header row
    y_gh_top = total_h - title_h
    y_gh_bot = y_gh_top - group_header_h

    # identifier columns: span both group + sub header rows so the label sits centered.
    id_span_h = group_header_h + sub_header_h
    for i, name in enumerate(id_cols):
        x0 = x_left[i]; x1 = x_left[i + 1]
        ax.add_patch(Rectangle((x0, y_gh_top - id_span_h), x1 - x0, id_span_h,
                               facecolor=ID_HEADER_BG, edgecolor="white",
                               linewidth=1.2))
        ax.text((x0 + x1) / 2, y_gh_top - id_span_h / 2, name,
                ha="center", va="center", fontsize=12, color="white",
                fontweight="bold")

    # metric group headers
    col_cursor = len(id_cols)
    for si, (label, _) in enumerate(SECTIONS):
        span = len(METRICS)
        x0 = x_left[col_cursor]; x1 = x_left[col_cursor + span]
        color = VAL_HEADER_BG if label == "Val" else GROUP_HEADER_BG
        ax.add_patch(Rectangle((x0, y_gh_bot), x1 - x0, group_header_h,
                               facecolor=color, edgecolor="white",
                               linewidth=1.2))
        ax.text((x0 + x1) / 2, y_gh_bot + group_header_h / 2, label,
                ha="center", va="center", fontsize=12.5, color="white",
                fontweight="bold")
        col_cursor += span

    # ---- sub-header row (metric labels under each group)
    y_sh_top = y_gh_bot
    y_sh_bot = y_sh_top - sub_header_h
    col_cursor = len(id_cols)
    for _ in SECTIONS:
        for ml, _ in METRICS:
            x0 = x_left[col_cursor]; x1 = x_left[col_cursor + 1]
            ax.add_patch(Rectangle((x0, y_sh_bot), x1 - x0, sub_header_h,
                                   facecolor=SUB_HEADER_BG, edgecolor=GRID,
                                   linewidth=CELL_LW))
            ax.text((x0 + x1) / 2, y_sh_bot + sub_header_h / 2, ml,
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color=TEXT)
            col_cursor += 1

    # ---- precompute best (max mean) per (combo, metric) — comparing
    # GateMol post vs TabPFN within the same combo.
    # best_mean[(combo_tag, section_key, metric_label)] = max across the
    # two models for that combo.
    best_mean = {}
    combo_tags = sorted({tag for tag, *_ in ROWS})
    for ct in combo_tags:
        idxs = [i for i, (t, *_) in enumerate(ROWS) if t == ct]
        for _, sec in SECTIONS:
            for ml, _ in METRICS:
                best_mean[(ct, sec, ml)] = max(data[i][sec][ml][0] for i in idxs)

    # ---- data rows
    y_data_top = y_sh_bot
    prev_tag = None
    for ri, (tag, features, model_label, _) in enumerate(ROWS):
        y_top = y_data_top - ri * row_h
        y_bot = y_top - row_h

        # ID cells — only label the combo/features on the first row of each combo group
        first_of_group = tag != prev_tag
        prev_tag = tag
        id_vals = [
            (tag if first_of_group else "", "bold", "sans-serif"),
            (features if first_of_group else "", "normal", "monospace"),
            (model_label, "normal", "sans-serif"),
        ]
        for ci, (val, weight, family) in enumerate(id_vals):
            x0 = x_left[ci]; x1 = x_left[ci + 1]
            ax.add_patch(Rectangle((x0, y_bot), x1 - x0, row_h,
                                   facecolor="#FFFFFF", edgecolor=GRID,
                                   linewidth=CELL_LW))
            ax.text((x0 + x1) / 2, (y_top + y_bot) / 2, val,
                    ha="center", va="center", fontsize=11,
                    fontweight=weight, family=family, color=TEXT)

        # metric cells
        col_cursor = len(id_cols)
        for _, sec in SECTIONS:
            for ml, _ in METRICS:
                mean, std = data[ri][sec][ml]
                is_best = (mean == best_mean[(tag, sec, ml)])
                x0 = x_left[col_cursor]; x1 = x_left[col_cursor + 1]
                ax.add_patch(Rectangle((x0, y_bot), x1 - x0, row_h,
                                       facecolor="#FFFFFF", edgecolor=GRID,
                                       linewidth=CELL_LW))
                ax.text((x0 + x1) / 2, (y_top + y_bot) / 2,
                        fmt(mean, std),
                        ha="center", va="center", fontsize=11,
                        family="monospace", color=TEXT,
                        fontweight="bold" if is_best else "normal")
                col_cursor += 1

    # ---- footer
    foot_y = y_data_top - n_rows * row_h - 0.2
    ax.text(0.1, foot_y,
            "Format: mean ± population std across 10 seeds. "
            "Bold = higher mean within the same combo (GateMol post vs TabPFN-2.5).\n"
            "val metrics from the scaffold validation split (per-seed best epoch for GateMol post; "
            "TabPFN-2.5 has no training so val is a single inference per seed).\n"
            "holdout metrics computed per seed on each subset, then averaged.\n"
            "iter200 itself was not a keep iter for any GateMol combo, so the held state at iter200 = "
            "last-kept iter's holdout_eval (monotonic non-decreasing val_AUC ⇒ best val_AUC ≤ 200).",
            ha="left", va="top", fontsize=9.5, color=MUTED)

    out_dir = Path("/home/minji/autoresearch_combos_v2/results/comparison/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "iter200_summary_table.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
