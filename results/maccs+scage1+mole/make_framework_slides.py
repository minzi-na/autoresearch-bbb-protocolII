"""Generate two PPT-style framework slides for combo2 (maccs+scage1+mole):
  - bbb_autoresearch_phase1_combo2.png  (structure search)
  - bbb_autoresearch_phase2_combo2.png  (Optuna HPO)

Layout matches the two reference slides:
  Title (top-left, bold)
  Subtitle (Phase 1 / Phase 2 description)
  Section 1 — AutoResearch outer loop (left half)
    4-step boxes connected by down arrows
    Keep / Discard outcome boxes at the bottom
  Section 2 — Inner training/trial loop (right half top)
    3-4 stacked boxes
  File table (right half bottom)
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# Korean font
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_FONT_PATH)
plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).parent / "figures_framework"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- palette (matches the reference slides) ----
COL_TITLE = "#1A1A1A"
COL_BODY = "#222"
COL_MUTED = "#555"

OUTER_BOX_FILL = "#FBE9D8"
OUTER_BOX_EDGE = "#E5A87E"
INNER_BOX_FILL = "#E8DEF4"
INNER_BOX_EDGE = "#9C7BC4"
HEAD_FILL_OUTER = "#FFFFFF"
HEAD_EDGE_OUTER = "#C9722E"
HEAD_FILL_INNER = "#FFFFFF"
HEAD_EDGE_INNER = "#7E58B5"
KEEP_FILL = "#E6F4EA"
KEEP_EDGE = "#34A853"
DISCARD_FILL = "#FCE8E6"
DISCARD_EDGE = "#EA4335"
TABLE_HEAD_BG = "#F4F4F4"
TABLE_GRID = "#CFCFCF"


def rounded_box(ax, x, y, w, h, *, fc, ec, lw=1.4, radius=0.04):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.005,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)


def draw_arrow(ax, x0, y0, x1, y1, color="#5A5A5A", lw=1.5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>",
                                 mutation_scale=15,
                                 color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def step_box(ax, x, y, w, h, *, title, body, fc, ec):
    rounded_box(ax, x, y, w, h, fc=fc, ec=ec, lw=1.5)
    ax.text(x + 0.18, y + h - 0.18, title,
            ha="left", va="top", fontsize=11.5,
            fontweight="bold", color=COL_BODY)
    nlines = body.count("\n") + 1
    if nlines >= 3:
        body_fs, body_offset, lspc = 8.2, 0.42, 1.08
    elif nlines == 2:
        body_fs, body_offset, lspc = 8.8, 0.46, 1.10
    else:
        body_fs, body_offset, lspc = 9.8, 0.55, 1.10
    ax.text(x + 0.18, y + h - body_offset, body,
            ha="left", va="top", fontsize=body_fs,
            color=COL_BODY, linespacing=lspc)


def section_header(ax, x, y, w, h, num, text, edge):
    rounded_box(ax, x, y, w, h, fc="#FFFFFF", ec=edge, lw=2.2, radius=0.06)
    ax.text(x + 0.2, y + h / 2 + 0.05, f"{num}. {text}",
            ha="left", va="center", fontsize=14, fontweight="bold",
            color=COL_TITLE)


def outcome_box(ax, x, y, w, h, *, title, bullets, fc, ec):
    rounded_box(ax, x, y, w, h, fc=fc, ec=ec, lw=2.0, radius=0.05)
    ax.text(x + 0.18, y + h - 0.22, title,
            ha="left", va="top", fontsize=11.5, fontweight="bold",
            color=ec)
    for i, b in enumerate(bullets):
        ax.text(x + 0.28, y + h - 0.55 - i * 0.32, f"• {b}",
                ha="left", va="top", fontsize=9.5, color=COL_BODY)


def file_table(ax, x, y, w, h, rows):
    n_rows = len(rows)
    row_h = h / n_rows
    col_w0 = w * 0.40   # file name col widened (longer paths)
    name_fs = 8.5 if n_rows >= 9 else 9.5
    desc_fs = 8.5 if n_rows >= 9 else 9.3
    for i, (fname, desc) in enumerate(rows):
        ry = y + h - (i + 1) * row_h
        # name cell
        ax.add_patch(Rectangle((x, ry), col_w0, row_h,
                               facecolor=TABLE_HEAD_BG, edgecolor=TABLE_GRID,
                               linewidth=0.9))
        ax.text(x + 0.10, ry + row_h / 2, fname,
                ha="left", va="center", fontsize=name_fs, fontweight="bold",
                family="monospace", color=COL_BODY)
        # desc cell
        ax.add_patch(Rectangle((x + col_w0, ry), w - col_w0, row_h,
                               facecolor="#FFFFFF", edgecolor=TABLE_GRID,
                               linewidth=0.9))
        ax.text(x + col_w0 + 0.12, ry + row_h / 2, desc,
                ha="left", va="center", fontsize=desc_fs, color=COL_BODY)


def build_slide(phase, out_path):
    # 16:9 slide, units chosen so 1 unit ~ slide-relative coordinate
    fig_w, fig_h = 16.0, 9.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # ---- title + subtitle ----
    ax.text(0.4, 8.55, "BBB_AutoResearch Framework  ·  combo2 (maccs+scage1+mole)",
            ha="left", va="bottom", fontsize=22, fontweight="bold",
            color=COL_TITLE)
    if phase == 1:
        sub = ("Phase 1 : 모델 아키텍쳐 최적화  Structure search   ·   "
               "iter1 → iter200, best=iter198  (mean_val_AUC 0.8527 ± 0.0036)")
    else:
        sub = ("Phase 2 : Optuna 탐색 설계 최적화  hyperparameter search   ·   "
               "Phase-1 best architecture frozen   ·   "
               "iter1 → iter18 (ceiling 도달, 2 KEEP: iter3·iter11)   ·   "
               "val winner = iter11 trial#21 (0.852878)")
    ax.text(0.4, 8.15, sub,
            ha="left", va="bottom", fontsize=14, fontweight="bold",
            color=COL_BODY)

    # ---- Section 1: outer loop header (left) ----
    section_header(ax, 0.4, 7.30, 7.3, 0.55, 1,
                   "AutoResearch 외부 루프", HEAD_EDGE_OUTER)
    ax.text(0.4, 7.10,
            "코드 자체를 실험 단위로 바꾸고 결과가 좋으면 keep, 아니면 discard",
            ha="left", va="top", fontsize=10.5, color=COL_MUTED)

    # ---- Section 2: inner loop header (right) ----
    section_header(ax, 8.1, 7.30, 7.3, 0.55, 2,
                   "step 3 안에서 실행되는 내부 학습 루프", HEAD_EDGE_INNER)
    if phase == 1:
        right_sub = "모델이 스스로 가중치를 학습하는 단계"
    else:
        right_sub = "Optuna trial 단위로 하이퍼파라미터를 탐색하는 단계"
    ax.text(8.1, 7.10, right_sub,
            ha="left", va="top", fontsize=10.5, color=COL_MUTED)

    # ---- Outer-loop steps (left side, 4 stacked boxes) ----
    step_x = 0.6
    step_w = 6.9
    step_h = 0.80
    step_gap = 0.18
    step_y0 = 6.0

    if phase == 1:
        steps = [
            ("Step 1. Baseline 고정",
             "d_model, d_ffn, depth, dropout, lr, batch_size, 10 seeds 고정"),
            ("Step 2. Agent가 아키텍쳐 변경 제안",
             "ex) projection norm, SGU 구조, residual scaling, pooling, EMA"),
            ("Step 3. train.py 수정 후 evaluate_combo.py 실행",
             "10-seed scaffold 학습 → keep 이면 final_holdout_eval.py (mandatory)"),
            ("Step 4. 결과 파싱",
             "주 판단지표: mean_val_auc (scaffold)   |   보조 reporting: 5-subset holdout"),
        ]
        keep_bullets = [
            "git commit 유지 (results.tsv keep=True)",
            "final_holdout_eval.py 실행 → 5-subset holdout 기록",
            "best architecture 갱신",
        ]
        discard_bullets = [
            "'git revert <commit>' (soft only)",
            "git reset --hard 금지 — discard 도 history 보존",
            "다음 architectural change 로 이동",
        ]
        objective_label = "mean_val_auc 개선"
    else:
        steps = [
            ("Step 1. Best architecture 고정",
             "Phase-1 best (iter198) 를 best_train.py 에 freeze, build_and_train 공개"),
            ("Step 2. Agent가 Optuna 탐색 설계 변경 제안",
             "search space, sampler (TPE/CMA-ES), pruner, objective 집계 방식\n"
             "+ best_train.py BASE_CONFIG 확장 (LR schedule / label_smoothing / ema_decay 등 노출)"),
            ("Step 3. optuna_combo.py(+best_train.py) 수정 후 evaluate_hpo.py 실행",
             "BUDGET: n_trials=30 (iter5~, 이전 50), top_k=3\n"
             "search 5-seed × 30 epochs → confirm 10-seed × 50 epochs (inline holdout)"),
            ("Step 4. 결과 파싱",
             "1차: 5-seed search mean   |   최종: best_confirm_mean_val_auc (10-seed)"),
        ]
        keep_bullets = [
            "best_confirm_mean_val_auc > threshold_mean",
            "no std buffer (threshold = max P1 / P2 keeps)",
            "hpo/results.tsv keep=True + best HPO 갱신",
        ]
        discard_bullets = [
            "'git revert <commit>' (soft only)",
            "git reset --hard 금지 (history 그대로 보존)",
            "다음 탐색 설계 변경으로 이동",
        ]
        objective_label = "best_confirm_mean_val_auc 개선"

    step_centers = []
    for i, (title, body) in enumerate(steps):
        y = step_y0 - i * (step_h + step_gap)
        step_box(ax, step_x, y, step_w, step_h,
                 title=title, body=body,
                 fc=OUTER_BOX_FILL, ec=OUTER_BOX_EDGE)
        step_centers.append((step_x + step_w / 2, y + step_h))

    # arrows between outer-loop steps
    for i in range(len(steps) - 1):
        cx = step_x + step_w / 2
        y_top = step_y0 - i * (step_h + step_gap)
        y_bot = step_y0 - (i + 1) * (step_h + step_gap) + step_h
        draw_arrow(ax, cx, y_top, cx, y_bot + 0.02)

    # ---- outcome boxes (Keep / Discard) ----
    out_y_top = step_y0 - len(steps) * (step_h + step_gap) - 0.15
    out_w, out_h = 3.3, 1.55
    keep_x = step_x + 0.05
    disc_x = step_x + step_w - out_w - 0.05
    outcome_box(ax, keep_x, out_y_top - out_h, out_w, out_h,
                title="Keep", bullets=keep_bullets,
                fc=KEEP_FILL, ec=KEEP_EDGE)
    outcome_box(ax, disc_x, out_y_top - out_h, out_w, out_h,
                title="Discard", bullets=discard_bullets,
                fc=DISCARD_FILL, ec=DISCARD_EDGE)

    # labeled split from step 4 to outcomes
    last_cy = step_y0 - (len(steps) - 1) * (step_h + step_gap)  # bottom of step 4 area
    branch_top_y = last_cy - 0.02
    branch_low_y = out_y_top + 0.10  # stop above the outcome box edge (y grows upward)
    # left branch (Keep)
    arrow_x0_l = step_x + step_w / 2 - 0.5
    arrow_x1_l = keep_x + out_w / 2
    draw_arrow(ax, arrow_x0_l, branch_top_y, arrow_x1_l, branch_low_y,
               color="#34A853", lw=1.7)
    mid_x_l = (arrow_x0_l + arrow_x1_l) / 2
    mid_y_l = (branch_top_y + branch_low_y) / 2
    ax.text(mid_x_l - 0.15, mid_y_l + 0.05, objective_label,
            ha="right", va="center", fontsize=9.5, color="#34A853",
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    # right branch (Discard)
    arrow_x0_r = step_x + step_w / 2 + 0.5
    arrow_x1_r = disc_x + out_w / 2
    draw_arrow(ax, arrow_x0_r, branch_top_y, arrow_x1_r, branch_low_y,
               color="#EA4335", lw=1.7)
    mid_x_r = (arrow_x0_r + arrow_x1_r) / 2
    mid_y_r = (branch_top_y + branch_low_y) / 2
    ax.text(mid_x_r + 0.15, mid_y_r + 0.05, "개선 없음 또는 하락",
            ha="left", va="center", fontsize=9.5, color="#EA4335",
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    # ---- Inner loop boxes (right top) ----
    inner_x = 8.3
    inner_w = 6.8
    inner_h = 0.95
    inner_gap = 0.18
    inner_y0 = 6.0
    if phase == 1:
        inners = [
            ("Input",
             "feature: maccs (167) + scage1 (2048) + mole (768)  → 일괄 z-score 후 concat"),
            ("Forward / Backward",
             "예측값 계산 → BCE loss → gradient → AdamW step. 아키텍처는 고정한 채 weight 만 학습."),
            ("Validation / Early stopping",
             "scaffold val loss/AUC patience 추적 → best state 선택, 10-seed 평균으로 architecture 평가"),
        ]
    else:
        inners = [
            ("Input",
             "고정 architecture + trial-sampled hp\n"
             "기본: lr, wd, dropout, drop_path, mod_drop, head_dropout, batch, grad_clip\n"
             "확장: lr_schedule / label_smoothing / ema_decay 등 (best_train.py BASE_CONFIG 노출 시)"),
            ("Trial training",
             "sampled hp 로 모델 훈련 (search: 5-seed × 30 epochs)\n"
             "pruner iter별 가변 — MedianPruner / NopPruner / Successive 등 시도"),
            ("Trial objective",
             "5-seed scaffold validation ROC-AUC mean →\n"
             "TPESampler / CmaEsSampler 가 다음 trial 제안 (study 30 trials)"),
            ("Final reevaluation",
             "top-3 trial → 10-seed × 50 epochs scaffold confirm (best_confirm_mean_val_auc)\n"
             "+ inline holdout (nn05 / total)"),
        ]

    for i, (t, b) in enumerate(inners):
        y = inner_y0 - i * (inner_h + inner_gap)
        step_box(ax, inner_x, y, inner_w, inner_h,
                 title=t, body=b,
                 fc=INNER_BOX_FILL, ec=INNER_BOX_EDGE)
    for i in range(len(inners) - 1):
        cx = inner_x + inner_w / 2
        y_top = inner_y0 - i * (inner_h + inner_gap)
        y_bot = inner_y0 - (i + 1) * (inner_h + inner_gap) + inner_h
        draw_arrow(ax, cx, y_top, cx, y_bot + 0.02)

    # connector: from outer step 3 to inner loop input
    outer_step3_y = step_y0 - 2 * (step_h + step_gap) + step_h / 2
    inner_input_y = inner_y0 + inner_h / 2
    draw_arrow(ax, step_x + step_w + 0.02, outer_step3_y,
               inner_x - 0.02, inner_input_y, color="#5A5A5A", lw=1.6)

    # connector: from inner validation/final back to outer step 4
    inner_last_y = inner_y0 - (len(inners) - 1) * (inner_h + inner_gap) + inner_h / 2
    outer_step4_y = step_y0 - 3 * (step_h + step_gap) + step_h / 2
    draw_arrow(ax, inner_x - 0.02, inner_last_y,
               step_x + step_w + 0.02, outer_step4_y, color="#5A5A5A", lw=1.6)

    # ---- File table (right bottom) ----
    if phase == 1:
        rows = [
            ("prepare.py",            "고정 파일, 데이터 로딩 / 피처 생성 / split / 평가 함수 / 상수 정의"),
            ("train.py",              "수정 대상 — 모델 구조 + train_model/eval_model 변경"),
            ("evaluate_combo.py",     "10-seed scaffold 학습 + iter 행 TSV append, keep 판정"),
            ("final_holdout_eval.py", "keep 직후 mandatory, 5-subset holdout 평가 + JSON 저장"),
            ("program.md",            "Autoresearch agent 작동 문서 (loop rules)"),
            ("architecture_ideas.md", "구조 변경 카탈로그 + priority 표"),
            ("results.tsv",           "iter 결과 누적 로그 (커밋과 1:1 매핑)"),
            ("results/<combo>/weights/iter<N>/",    "keep 시 자동 저장 — best state per seed (.pt)"),
            ("results/<combo>/holdout_eval/iter<N>_<hash>.json", "5-subset holdout 지표 JSON"),
        ]
    else:
        rows = [
            ("prepare.py",        "고정 파일 (Phase 1과 동일 데이터/스플릿)"),
            ("best_train.py",     "model class frozen, BASE_CONFIG / train_model 확장 가능 (새 training-side HP 노출)"),
            ("optuna_combo.py",   "수정 대상 — suggest_config / sampler / pruner / objective 정의"),
            ("evaluate_hpo.py",   "frozen runner — study 실행, top-3 confirm, BUDGET (n_trials=30 iter5~)"),
            ("program_phase2.md", "Phase 2 agent 작동 문서 (Optuna loop rules)"),
            ("hpo/results.tsv",   "iter 결과 누적 로그 (keep 판정 + threshold 추적, 현재 2 keeps: iter3·iter11)"),
            ("hpo/study_*.json",  "Optuna study 직렬화 + top_k_confirm rows (per-trial holdout 포함)"),
            ("hpo/trials_*.tsv",  "trial 단위 raw 기록 (suggested hp + search mean_val_auc)"),
        ]
    table_x = 8.3
    table_y = 0.45
    table_w = 6.8
    table_h = 2.00
    file_table(ax, table_x, table_y, table_w, table_h, rows)

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    build_slide(1, OUT_DIR / "bbb_autoresearch_phase1_combo2.png")
    build_slide(2, OUT_DIR / "bbb_autoresearch_phase2_combo2.png")
