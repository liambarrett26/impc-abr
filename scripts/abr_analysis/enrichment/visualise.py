#!/usr/bin/env python3
"""
Visualisations for the enrichment analysis results.

Produces:
1. Sankey diagram (matplotlib): MP hierarchy flow from genes to enriched terms
2. Full vs filtered bar chart: circularity comparison at top-level
3. Dot plot of enriched sub-terms (acoustic vs non-acoustic)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import matplotlib
matplotlib.use("Agg")

from .config import RESULTS_DIR


# ── Colour scheme ──────────────────────────────────────────────────────

# Top-level category colours — used for both level 2 and level 3 nodes
TOP_LEVEL_COLOURS = {
    "hearing/vestibular/ear phenotype": "#5C6BC0",
    "behavior/neurological phenotype": "#1565C0",
    "nervous system phenotype": "#7B1FA2",
    "skeleton phenotype": "#00838F",
    "reproductive system phenotype": "#6D4C41",
    "homeostasis/metabolism phenotype": "#455A64",
    "immune system phenotype": "#AD1457",
}

# Dot plot: acoustic vs non-acoustic
ACOUSTIC_COLOUR = "#B71C1C"
ACOUSTIC_FILL = "#FFCDD2"
NON_ACOUSTIC_COLOUR = "#1B5E20"
NON_ACOUSTIC_FILL = "#C8E6C9"


# ── Helpers ────────────────────────────────────────────────────────────

def _draw_flow(ax, x0, y0, h0, x1, y1, h1, colour, alpha=0.25):
    """Draw a curved flow band between two nodes using cubic Bezier."""
    mid_x = (x0 + x1) / 2
    verts = [
        (x0, y0), (mid_x, y0), (mid_x, y1), (x1, y1),
        (x1, y1 + h1), (mid_x, y1 + h1), (mid_x, y0 + h0), (x0, y0 + h0),
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    path = MplPath(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=colour, edgecolor="none",
                                alpha=alpha, zorder=1)
    ax.add_patch(patch)


def _draw_node(ax, x, y, w, h, colour, label=None, label_side="right",
               fontsize=8, fontcolour="black"):
    """Draw a rectangular node with optional label."""
    rect = mpatches.Rectangle(
        (x, y), w, h,
        facecolor=colour, edgecolor="white", linewidth=0.5, zorder=3,
    )
    ax.add_patch(rect)
    if label:
        if label_side == "right":
            ax.text(x + w + 0.008, y + h / 2, label,
                    va="center", ha="left", fontsize=fontsize,
                    color=fontcolour, zorder=4)
        elif label_side == "left":
            ax.text(x - 0.008, y + h / 2, label,
                    va="center", ha="right", fontsize=fontsize,
                    color=fontcolour, zorder=4)


def _title_fix(name):
    """Title-case with acronym preservation."""
    label = name.title()
    for acr in ["Cns", "Ldl", "Hdl", "Ppi", "Abr"]:
        label = label.replace(acr, acr.upper())
    return label


# ── Sankey diagram ─────────────────────────────────────────────────────

def create_sankey(hier_df, top_df, output_dir=None):
    """Create a Sankey-style diagram. Sub-terms coloured by parent category."""
    if output_dir is None:
        output_dir = RESULTS_DIR

    sig_hier = hier_df[hier_df["significant"]].copy()
    sig_top = top_df[top_df["significant"]].copy()

    # ── Select sub-terms: top N by significance per parent ─────────
    MAX_PER_CATEGORY = 6
    selected_parts = []
    # Build a map from parent index to top-level name for colouring
    parent_idx_to_name = {}
    for i, (_, top_row) in enumerate(sig_top.iterrows()):
        top_id = top_row["mp_term_id"]
        parent_idx_to_name[i] = top_row["mp_term_name"]
        branch = sig_hier[sig_hier["top_level_mp_id"] == top_id].copy()
        if len(branch) == 0:
            continue
        sel = branch.nsmallest(MAX_PER_CATEGORY, "p_adjusted").copy()
        sel["_parent_idx"] = i
        selected_parts.append(sel)

    if not selected_parts:
        print("No significant terms to visualise")
        return

    selected = pd.concat(selected_parts).reset_index(drop=True)

    # ── Layout geometry ────────────────────────────────────────────
    x_source = 0.08
    x_top = 0.32
    x_sub = 0.56
    node_w = 0.022

    n_top = len(sig_top)
    total_value = sig_top["a"].sum()

    # Source node
    source_h = 0.75
    source_y = (1.0 - source_h) / 2

    # Top-level nodes: sqrt-scaled heights
    top_gap = 0.025
    top_available = 0.85
    raw_heights = np.sqrt(sig_top["a"].values.astype(float))
    raw_heights = raw_heights / raw_heights.sum() * (top_available - top_gap * (n_top - 1))
    raw_heights = np.maximum(raw_heights, 0.018)

    top_nodes = []
    y_cursor = 0.07
    for idx, (_, row) in enumerate(sig_top.iterrows()):
        top_nodes.append((y_cursor, raw_heights[idx], row))
        y_cursor += raw_heights[idx] + top_gap

    # Sub-terms: globally sqrt-scaled heights, centred on parent
    sub_gap_within = 0.006
    sub_gap_between = 0.016
    min_sub_h = 0.007
    max_sub_h = 0.030

    from collections import defaultdict as _defaultdict
    parent_to_selected = _defaultdict(list)
    for _, row in selected.iterrows():
        parent_to_selected[row["_parent_idx"]].append(row)

    all_a_values = selected["a"].values.astype(float)
    global_sqrt_min = np.sqrt(all_a_values.min())
    global_sqrt_max = np.sqrt(all_a_values.max())

    def _scale_height(a_val):
        sqrt_a = np.sqrt(float(a_val))
        if global_sqrt_max == global_sqrt_min:
            t = 0.5
        else:
            t = (sqrt_a - global_sqrt_min) / (global_sqrt_max - global_sqrt_min)
        return min_sub_h + t * (max_sub_h - min_sub_h)

    sub_groups = []
    for parent_idx in sorted(parent_to_selected.keys()):
        rows = parent_to_selected[parent_idx]
        parent_y_pos, parent_h, _ = top_nodes[parent_idx]
        parent_mid = parent_y_pos + parent_h / 2
        heights = np.array([_scale_height(r["a"]) for r in rows])
        total_gap = sub_gap_within * (len(rows) - 1)
        total_h = heights.sum() + total_gap
        group_bottom = parent_mid - total_h / 2
        group = []
        y_cursor = group_bottom
        for i, row in enumerate(rows):
            group.append((y_cursor, heights[i], row))
            y_cursor += heights[i] + sub_gap_within
        sub_groups.append((parent_idx, group))

    # Collision avoidance
    all_groups_flat = []
    for gi, (pidx, group) in enumerate(sub_groups):
        all_groups_flat.append((gi, group[0][0], group[-1][0] + group[-1][1], group))

    for _ in range(10):
        changed = False
        for i in range(len(all_groups_flat) - 1):
            gi_a, bot_a, top_a, grp_a = all_groups_flat[i]
            gi_b, bot_b, top_b, grp_b = all_groups_flat[i + 1]
            overlap = top_a + sub_gap_between - bot_b
            if overlap > 0:
                shift = overlap / 2 + 0.002
                new_grp_a = [(y - shift, h, r) for y, h, r in grp_a]
                new_grp_b = [(y + shift, h, r) for y, h, r in grp_b]
                all_groups_flat[i] = (gi_a, new_grp_a[0][0],
                                      new_grp_a[-1][0] + new_grp_a[-1][1], new_grp_a)
                all_groups_flat[i + 1] = (gi_b, new_grp_b[0][0],
                                           new_grp_b[-1][0] + new_grp_b[-1][1], new_grp_b)
                changed = True
        if not changed:
            break

    # Clamp to bounds
    for i, (gi, bot, top, group) in enumerate(all_groups_flat):
        if bot < 0.02:
            shift = 0.02 - bot
            group = [(y + shift, h, r) for y, h, r in group]
            all_groups_flat[i] = (gi, group[0][0], group[-1][0] + group[-1][1], group)
        grp_top = group[-1][0] + group[-1][1]
        if grp_top > 0.93:
            shift = grp_top - 0.93
            group = [(y - shift, h, r) for y, h, r in group]
            all_groups_flat[i] = (gi, group[0][0], group[-1][0] + group[-1][1], group)

    sub_nodes = []
    for gi, bot, top, group in all_groups_flat:
        for y, h, row in group:
            sub_nodes.append((y, h, row))

    # ── Draw ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")

    # Flows: source → top-level
    source_y_offset = source_y
    for i, (ty, th, trow) in enumerate(top_nodes):
        flow_h_source = (trow["a"] / total_value) * source_h
        colour = TOP_LEVEL_COLOURS.get(trow["mp_term_name"], "#78909C")
        _draw_flow(ax, x_source + node_w, source_y_offset, flow_h_source,
                   x_top, ty, th, colour, alpha=0.18)
        source_y_offset += flow_h_source

    # Flows: top-level → sub-terms (coloured by parent)
    from collections import defaultdict
    parent_to_subs = defaultdict(list)
    for si, (sy, sh, srow) in enumerate(sub_nodes):
        parent_to_subs[srow["_parent_idx"]].append((si, sy, sh, srow))

    for parent_idx, sub_list in parent_to_subs.items():
        parent_y_pos, parent_h, parent_row = top_nodes[parent_idx]
        parent_colour = TOP_LEVEL_COLOURS.get(parent_row["mp_term_name"], "#78909C")

        a_values = np.array([srow["a"] for _, _, _, srow in sub_list], dtype=float)
        a_norm = a_values / a_values.sum()
        flow_heights = a_norm * parent_h

        y_cursor = parent_y_pos
        for i, (si, sy, sh, srow) in enumerate(sub_list):
            _draw_flow(ax, x_top + node_w, y_cursor, flow_heights[i],
                       x_sub, sy, sh, parent_colour, alpha=0.28)
            y_cursor += flow_heights[i]

    # Draw source node
    _draw_node(ax, x_source, source_y, node_w, source_h, "#37474F")

    # Draw top-level nodes
    for ty, th, trow in top_nodes:
        colour = TOP_LEVEL_COLOURS.get(trow["mp_term_name"], "#78909C")
        label = f"{_title_fix(trow['mp_term_name'].replace(' phenotype', ''))} (N={int(trow['a'])})"
        _draw_node(ax, x_top, ty, node_w, th, colour,
                   label=label, label_side="left", fontsize=11)

    # Draw sub-term nodes — coloured by parent
    for sy, sh, srow in sub_nodes:
        parent_name = parent_idx_to_name.get(srow["_parent_idx"], "")
        colour = TOP_LEVEL_COLOURS.get(parent_name, "#78909C")
        label = _title_fix(srow["mp_term_name"])
        if len(label) > 55:
            label = label[:52] + "..."
        label = f"{label} (N={int(srow['a'])})"
        _draw_node(ax, x_sub, sy, node_w, sh, colour,
                   label=label, label_side="right", fontsize=9.5)

    # ── Title and annotations ──────────────────────────────────────
    ax.text(0.5, 0.99, "MP Term Enrichment Hierarchy",
            ha="center", va="top", fontsize=16, fontweight="bold",
            transform=ax.transAxes)

    ax.text(x_source + node_w / 2, source_y + source_h + 0.025,
            "133 Hearing Loss Genes\n(BF \u2265 3)",
            ha="center", va="bottom", fontsize=11.5, fontweight="bold")

    ax.text(x_top + node_w / 2, 0.96, "Top-level MP category",
            ha="center", va="top", fontsize=16, color="#444", fontstyle="italic")
    ax.text(x_sub + node_w / 2, 0.96, "Enriched sub-terms",
            ha="center", va="top", fontsize=16, color="#444", fontstyle="italic")

    # No classification legend — sub-terms coloured by parent category

    plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02)

    for ext in ["png", "eps", "pdf"]:
        path = output_dir / f"sankey_enrichment.{ext}"
        fig.savefig(str(path), dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved Sankey to {path}")

    plt.close(fig)


# ── Full vs filtered bar chart ─────────────────────────────────────────

def create_circularity_comparison(top_df, top_filtered_df, output_dir=None):
    """Side-by-side bar chart: full vs acoustic-filtered top-level enrichment."""
    if output_dir is None:
        output_dir = RESULTS_DIR

    merged = top_df.merge(
        top_filtered_df, on="mp_term_id", suffixes=("_full", "_filtered"),
    )
    show = merged[merged["significant_full"] | merged["significant_filtered"]].copy()
    show = show.sort_values("odds_ratio_full", ascending=True)
    show["label"] = show["mp_term_name_full"].str.replace(" phenotype", "", regex=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(show))
    bar_height = 0.35

    ax.barh(y + bar_height / 2, show["odds_ratio_full"].values,
            height=bar_height, label="Full (all terms)",
            color="#1565C0", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.barh(y - bar_height / 2, show["odds_ratio_filtered"].values,
            height=bar_height, label="Filtered (acoustic terms removed)",
            color="#388E3C", alpha=0.85, edgecolor="white", linewidth=0.5)

    for i, (_, row) in enumerate(show.iterrows()):
        if row["significant_full"]:
            ax.text(row["odds_ratio_full"] + 0.15, i + bar_height / 2,
                    "*", ha="left", va="center", fontsize=14,
                    fontweight="bold", color="#1565C0")
        if row["significant_filtered"]:
            ax.text(row["odds_ratio_filtered"] + 0.15, i - bar_height / 2,
                    "*", ha="left", va="center", fontsize=14,
                    fontweight="bold", color="#388E3C")

    ax.axvline(x=1, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    ns_row = show[show["label"] == "nervous system"]
    if len(ns_row) > 0:
        y_positions = list(show.index)
        y_pos = y_positions.index(ns_row.index[0])
        ax.annotate("Entirely\ncircular",
                     xy=(1.5, y_pos - bar_height / 2),
                     fontsize=9, color="#D32F2F", fontweight="bold",
                     ha="left", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(show["label"].values, fontsize=11)
    ax.set_xlabel("Odds Ratio", fontsize=12)
    ax.set_title("Top-Level MP Enrichment: Full vs. Acoustic-Filtered",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_xlim(0, max(show["odds_ratio_full"].max(), 6) * 1.15)

    plt.tight_layout()
    for ext in ["png", "eps"]:
        path = output_dir / f"circularity_comparison.{ext}"
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        print(f"Saved circularity comparison to {path}")
    plt.close(fig)


# ── Dot plot ───────────────────────────────────────────────────────────

def create_dot_plot(hier_df, output_dir=None):
    """Dot plot of significant enriched sub-terms.
    Simplified to acoustic vs non-acoustic classification.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR

    sig = hier_df[hier_df["significant"]].copy()

    # Binary classification: acoustic vs non-acoustic
    sig = sig.copy()
    sig["is_acoustic"] = sig["classification"] == "acoustic_dependent"

    # Select terms to show
    acoustic = sig[sig["is_acoustic"]].nsmallest(8, "p_adjusted")
    non_acoustic = sig[~sig["is_acoustic"]].nsmallest(16, "p_adjusted")
    show = pd.concat([acoustic, non_acoustic]).drop_duplicates(subset="mp_term_id")

    # Order: acoustic at top, non-acoustic bottom
    show = show.copy()
    show["_order"] = show["is_acoustic"].map({True: 0, False: 1})
    show = show.sort_values(["_order", "fold_enrichment"], ascending=[False, True])

    size_scale = 6
    max_a = show["a"].max()
    row_height = 0.55
    fig_height = max(9, len(show) * row_height + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y = np.arange(len(show)) * 1.2

    for i, (_, row) in enumerate(show.iterrows()):
        if row["is_acoustic"]:
            edge, fill = ACOUSTIC_COLOUR, ACOUSTIC_FILL
        else:
            edge, fill = NON_ACOUSTIC_COLOUR, NON_ACOUSTIC_FILL
        ax.scatter(row["fold_enrichment"], y[i],
                   s=row["a"] * size_scale, c=fill, alpha=0.9,
                   edgecolors=edge, linewidths=1.5, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([_title_fix(n) for n in show["mp_term_name"].values],
                       fontsize=11)
    ax.set_xlabel("Fold Enrichment", fontsize=13)
    ax.set_title("Enriched MP Sub-Terms (FDR < 0.05)",
                 fontsize=15, fontweight="bold")

    # Legend
    legend_handles = []
    legend_handles.append(ax.scatter([], [], c=ACOUSTIC_FILL, s=80,
                                      edgecolors=ACOUSTIC_COLOUR, linewidths=1.5,
                                      label="Acoustic-dependent"))
    legend_handles.append(ax.scatter([], [], c=NON_ACOUSTIC_FILL, s=80,
                                      edgecolors=NON_ACOUSTIC_COLOUR, linewidths=1.5,
                                      label="Non-acoustic"))

    size_legend_values = [v for v in [5, 20, 50, 100] if v <= max_a + 10]
    for sv in size_legend_values:
        handle = ax.scatter([], [], c="#E0E0E0", s=sv * size_scale, alpha=0.5,
                            edgecolors="#757575", linewidths=0.5,
                            label=f"N = {sv}")
        legend_handles.append(handle)

    ax.legend(handles=legend_handles, loc="lower right", fontsize=11,
              framealpha=0.9, title="Classification / Gene Count",
              title_fontsize=12, labelspacing=1.2, handletextpad=1.5)

    ax.axvline(x=1, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "eps"]:
        path = output_dir / f"enrichment_dot_plot.{ext}"
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        print(f"Saved dot plot to {path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    hier = pd.read_csv(RESULTS_DIR / "hierarchical_enrichment.csv")
    top = pd.read_csv(RESULTS_DIR / "top_level_enrichment.csv")
    top_filt = pd.read_csv(RESULTS_DIR / "top_level_enrichment_no_acoustic.csv")

    print("Creating Sankey diagram...")
    create_sankey(hier, top)

    print("\nCreating circularity comparison chart...")
    create_circularity_comparison(top, top_filt)

    print("\nCreating dot plot...")
    create_dot_plot(hier)

    print("\nDone.")


if __name__ == "__main__":
    main()
