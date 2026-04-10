#!/usr/bin/env python3
"""
Visualisations for the enrichment analysis results.

Produces:
1. Sankey diagram (matplotlib): MP hierarchy flow from genes to enriched terms
2. Full vs filtered bar chart: circularity comparison at top-level
3. Dot plot of enriched sub-terms
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

CLASSIFICATION_COLOURS = {
    "acoustic_dependent": "#B71C1C",
    "vestibular": "#E65100",
    "independent": "#1B5E20",
    "unclear": "#616161",
}

CLASSIFICATION_LABELS = {
    "acoustic_dependent": "Acoustic-dependent",
    "vestibular": "Vestibular",
    "independent": "Independent",
    "unclear": "Unclear",
}

# Top-level: use blues/purples/teals to avoid red/amber/green clash
TOP_LEVEL_COLOURS = {
    "hearing/vestibular/ear phenotype": "#5C6BC0",
    "behavior/neurological phenotype": "#1565C0",
    "nervous system phenotype": "#7B1FA2",
    "skeleton phenotype": "#00838F",
    "reproductive system phenotype": "#6D4C41",
    "homeostasis/metabolism phenotype": "#455A64",
    "immune system phenotype": "#AD1457",
}


# ── Sankey diagram (matplotlib) ────────────────────────────────────────

def _draw_flow(ax, x0, y0, h0, x1, y1, h1, colour, alpha=0.25):
    """Draw a curved flow band between two nodes using cubic Bezier."""
    mid_x = (x0 + x1) / 2
    verts = [
        (x0, y0),
        (mid_x, y0),
        (mid_x, y1),
        (x1, y1),
        (x1, y1 + h1),
        (mid_x, y1 + h1),
        (mid_x, y0 + h0),
        (x0, y0 + h0),
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
        elif label_side == "inside":
            ax.text(x + w / 2, y + h / 2, label,
                    va="center", ha="center", fontsize=fontsize,
                    color="white", fontweight="bold", zorder=4)


def create_sankey(hier_df, top_df, output_dir=None):
    """Create a Sankey-style diagram using matplotlib."""
    if output_dir is None:
        output_dir = RESULTS_DIR

    sig_hier = hier_df[hier_df["significant"]].copy()
    sig_top = top_df[top_df["significant"]].copy()

    # ── Select sub-terms ───────────────────────────────────────────
    MAX_PER_CATEGORY = 6
    selected_parts = []
    for i, (_, top_row) in enumerate(sig_top.iterrows()):
        top_id = top_row["mp_term_id"]
        branch = sig_hier[sig_hier["top_level_mp_id"] == top_id].copy()
        if len(branch) == 0:
            continue
        av = branch[branch["classification"].isin(["acoustic_dependent", "vestibular"])]
        ind = branch[branch["classification"] == "independent"].head(
            MAX_PER_CATEGORY - len(av)
        )
        sel = pd.concat([av, ind]).head(MAX_PER_CATEGORY).copy()
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
    n_sub = len(selected)
    total_value = sig_top["a"].sum()

    # ── Compute node positions ─────────────────────────────────────
    # Source: use sqrt scaling so it's not absurdly tall
    source_h = 0.75
    source_y = (1.0 - source_h) / 2

    # Top-level: sqrt-scaled heights so small categories remain visible
    top_gap = 0.025
    top_available = 0.85
    raw_heights = np.sqrt(sig_top["a"].values.astype(float))
    raw_heights = raw_heights / raw_heights.sum() * (top_available - top_gap * (n_top - 1))
    min_h = 0.018
    raw_heights = np.maximum(raw_heights, min_h)

    top_nodes = []
    y_cursor = 0.07
    for idx, (_, row) in enumerate(sig_top.iterrows()):
        h = raw_heights[idx]
        top_nodes.append((y_cursor, h, row))
        y_cursor += h + top_gap

    # Sub-terms: sized by sqrt(a) normalised GLOBALLY, centred around parent
    sub_gap_within = 0.006
    sub_gap_between = 0.016
    min_sub_h = 0.007
    max_sub_h = 0.030

    # First: compute global sqrt(a) scaling so node sizes are comparable
    # across all groups
    from collections import defaultdict as _defaultdict
    parent_to_selected = _defaultdict(list)
    for _, row in selected.iterrows():
        parent_to_selected[row["_parent_idx"]].append(row)

    all_a_values = selected["a"].values.astype(float)
    global_sqrt_min = np.sqrt(all_a_values.min())
    global_sqrt_max = np.sqrt(all_a_values.max())

    def _scale_height(a_val):
        """Map a-value to node height using global sqrt scale."""
        sqrt_a = np.sqrt(float(a_val))
        if global_sqrt_max == global_sqrt_min:
            t = 0.5
        else:
            t = (sqrt_a - global_sqrt_min) / (global_sqrt_max - global_sqrt_min)
        return min_sub_h + t * (max_sub_h - min_sub_h)

    # Build groups with globally-scaled heights, centred on parent
    sub_groups = []
    for parent_idx in sorted(parent_to_selected.keys()):
        rows = parent_to_selected[parent_idx]
        parent_y_pos, parent_h, _ = top_nodes[parent_idx]
        parent_mid = parent_y_pos + parent_h / 2

        n_subs = len(rows)
        heights = np.array([_scale_height(r["a"]) for r in rows])

        total_gap = sub_gap_within * (n_subs - 1)
        total_h = heights.sum() + total_gap
        group_bottom = parent_mid - total_h / 2

        group = []
        y_cursor = group_bottom
        for i, row in enumerate(rows):
            group.append((y_cursor, heights[i], row))
            y_cursor += heights[i] + sub_gap_within
        sub_groups.append((parent_idx, group))

    # Second pass: resolve vertical overlaps between groups
    # Push groups apart if they collide
    all_groups_flat = []  # (group_idx, items)
    for gi, (pidx, group) in enumerate(sub_groups):
        group_top = group[-1][0] + group[-1][1]
        group_bottom = group[0][0]
        all_groups_flat.append((gi, group_bottom, group_top, group))

    # Sort by vertical position and push overlapping groups apart
    for iteration in range(10):
        changed = False
        for i in range(len(all_groups_flat) - 1):
            gi_a, bot_a, top_a, grp_a = all_groups_flat[i]
            gi_b, bot_b, top_b, grp_b = all_groups_flat[i + 1]
            overlap = top_a + sub_gap_between - bot_b
            if overlap > 0:
                shift = overlap / 2 + 0.002
                # Push A down, B up
                new_grp_a = [(y - shift, h, r) for y, h, r in grp_a]
                new_grp_b = [(y + shift, h, r) for y, h, r in grp_b]
                all_groups_flat[i] = (gi_a, new_grp_a[0][0],
                                      new_grp_a[-1][0] + new_grp_a[-1][1], new_grp_a)
                all_groups_flat[i + 1] = (gi_b, new_grp_b[0][0],
                                           new_grp_b[-1][0] + new_grp_b[-1][1], new_grp_b)
                changed = True
        if not changed:
            break

    # Clamp all groups to stay within plot bounds [0.03, 0.95]
    y_min_bound = 0.02
    y_max_bound = 0.93
    for i, (gi, bot, top, group) in enumerate(all_groups_flat):
        if bot < y_min_bound:
            shift = y_min_bound - bot
            group = [(y + shift, h, r) for y, h, r in group]
            all_groups_flat[i] = (gi, group[0][0],
                                  group[-1][0] + group[-1][1], group)
        grp_top = group[-1][0] + group[-1][1]
        if grp_top > y_max_bound:
            shift = grp_top - y_max_bound
            group = [(y - shift, h, r) for y, h, r in group]
            all_groups_flat[i] = (gi, group[0][0],
                                  group[-1][0] + group[-1][1], group)

    # Flatten into sub_nodes list
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

    # Flows: top-level → sub-terms (stacked within each parent)
    # For each parent, normalise its sub-term 'a' values so flow heights
    # stack to exactly fill the parent node height, then draw bottom-to-top
    from collections import defaultdict
    parent_to_subs = defaultdict(list)
    for si, (sy, sh, srow) in enumerate(sub_nodes):
        parent_to_subs[srow["_parent_idx"]].append((si, sy, sh, srow))

    for parent_idx, sub_list in parent_to_subs.items():
        parent_y_pos, parent_h, parent_row = top_nodes[parent_idx]

        # Normalise a-values to fill parent height
        a_values = np.array([srow["a"] for _, _, _, srow in sub_list], dtype=float)
        a_norm = a_values / a_values.sum()
        flow_heights = a_norm * parent_h

        # Stack from bottom of parent node upward
        y_cursor = parent_y_pos
        for i, (si, sy, sh, srow) in enumerate(sub_list):
            fh = flow_heights[i]
            colour = CLASSIFICATION_COLOURS.get(srow["classification"], "#757575")
            _draw_flow(ax, x_top + node_w, y_cursor, fh,
                       x_sub, sy, sh, colour, alpha=0.28)
            y_cursor += fh

    # Draw source node
    _draw_node(ax, x_source, source_y, node_w, source_h, "#37474F")

    # Draw top-level nodes
    for ty, th, trow in top_nodes:
        colour = TOP_LEVEL_COLOURS.get(trow["mp_term_name"], "#78909C")
        label = f"{trow['mp_term_name'].replace(' phenotype', '').title()} (N={int(trow['a'])})"
        _draw_node(ax, x_top, ty, node_w, th, colour,
                   label=label, label_side="left", fontsize=11)

    # Draw sub-term nodes with labels to the right (including N=X)
    for sy, sh, srow in sub_nodes:
        colour = CLASSIFICATION_COLOURS.get(srow["classification"], "#757575")
        label = srow["mp_term_name"].title()
        # Fix acronyms that title() breaks
        for acr in ["Cns", "Ldl", "Hdl", "Ppi"]:
            label = label.replace(acr, acr.upper())
        if len(label) > 55:
            label = label[:52] + "..."
        label = f"{label} (N={int(srow['a'])})"
        _draw_node(ax, x_sub, sy, node_w, sh, colour,
                   label=label, label_side="right", fontsize=9.5)

    # ── Title ──────────────────────────────────────────────────────
    ax.text(0.5, 0.99, "MP Term Enrichment Hierarchy",
            ha="center", va="top", fontsize=16, fontweight="bold",
            transform=ax.transAxes)

    # Source label — above the node
    ax.text(x_source + node_w / 2, source_y + source_h + 0.025,
            "133 Hearing Loss Genes\n(BF \u2265 3)",
            ha="center", va="bottom", fontsize=11.5, fontweight="bold")

    # Column headers
    ax.text(x_top + node_w / 2, 0.96, "Top-level MP category",
            ha="center", va="top", fontsize=16, color="#444", fontstyle="italic")
    ax.text(x_sub + node_w / 2, 0.96, "Enriched sub-terms",
            ha="center", va="top", fontsize=16, color="#444", fontstyle="italic")

    # ── Classification legend — top right, with proper squares ─────
    legend_x = 0.92
    legend_y = 0.95
    legend_items = [
        ("Acoustic-dependent", CLASSIFICATION_COLOURS["acoustic_dependent"]),
        ("Vestibular", CLASSIFICATION_COLOURS["vestibular"]),
        ("Independent", CLASSIFICATION_COLOURS["independent"]),
    ]
    ax.text(legend_x, legend_y, "Sub-term classification",
            ha="right", va="top", fontsize=11.5, fontweight="bold",
            transform=ax.transAxes)
    for i, (label, colour) in enumerate(legend_items):
        y_leg = legend_y - 0.045 * (i + 1)
        ax.add_patch(mpatches.Rectangle(
            (legend_x + 0.01, y_leg - 0.01), 0.015, 0.02,
            facecolor=colour, edgecolor="none",
            transform=ax.transAxes, zorder=5, clip_on=False,
        ))
        ax.text(legend_x, y_leg, label,
                ha="right", va="center", fontsize=11,
                transform=ax.transAxes)

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
    """Dot plot of significant enriched sub-terms."""
    if output_dir is None:
        output_dir = RESULTS_DIR

    sig = hier_df[hier_df["significant"]].copy()

    terms_to_show = []
    for classification in ["acoustic_dependent", "vestibular", "independent"]:
        subset = sig[sig["classification"] == classification].nsmallest(
            12 if classification == "independent" else 6, "p_adjusted",
        )
        terms_to_show.append(subset)

    show = pd.concat(terms_to_show).drop_duplicates(subset="mp_term_id")

    # Order: acoustic at top, vestibular middle, independent bottom
    class_order = {"acoustic_dependent": 0, "vestibular": 1, "independent": 2}
    show = show.copy()
    show["_class_order"] = show["classification"].map(class_order)
    show = show.sort_values(
        ["_class_order", "fold_enrichment"], ascending=[False, True]
    )

    # Title-case labels with acronym fixes
    def _title_fix(name):
        label = name.title()
        for acr in ["Cns", "Ldl", "Hdl", "Ppi", "Abr"]:
            label = label.replace(acr, acr.upper())
        return label

    size_scale = 6
    max_a = show["a"].max()
    row_height = 0.55
    fig_height = max(9, len(show) * row_height + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y = np.arange(len(show)) * 1.2

    # Light fill with deep hue outline
    CLASSIFICATION_FILL = {
        "acoustic_dependent": "#FFCDD2",  # light red
        "vestibular": "#FFE0B2",          # light orange
        "independent": "#C8E6C9",         # light green
        "unclear": "#E0E0E0",
    }

    for i, (_, row) in enumerate(show.iterrows()):
        edge_colour = CLASSIFICATION_COLOURS.get(row["classification"], "#757575")
        fill_colour = CLASSIFICATION_FILL.get(row["classification"], "#E0E0E0")
        ax.scatter(row["fold_enrichment"], y[i],
                   s=row["a"] * size_scale, c=fill_colour, alpha=0.9,
                   edgecolors=edge_colour, linewidths=1.5,
                   zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([_title_fix(n) for n in show["mp_term_name"].values],
                       fontsize=11)
    ax.set_xlabel("Fold Enrichment", fontsize=13)
    ax.set_title("Enriched MP Sub-Terms (FDR < 0.05)",
                 fontsize=15, fontweight="bold")

    # Combined legend
    legend_handles = []
    for classification in ["acoustic_dependent", "vestibular", "independent"]:
        if classification in show["classification"].values:
            edge = CLASSIFICATION_COLOURS[classification]
            fill = CLASSIFICATION_FILL[classification]
            label = CLASSIFICATION_LABELS[classification]
            handle = ax.scatter([], [], c=fill, s=80, label=label,
                                edgecolors=edge, linewidths=1.5)
            legend_handles.append(handle)

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
