#!/usr/bin/env python3
"""
Test: stage 3 node sizing and vertical centering around parent nodes.
Uses 3 parents with different sub-term counts and a-value ranges.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import matplotlib
matplotlib.use("Agg")


def draw_flow(ax, x0, y0, h0, x1, y1, h1, colour, alpha=0.28):
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
    patch = mpatches.PathPatch(MplPath(verts, codes),
                                facecolor=colour, edgecolor="none",
                                alpha=alpha, zorder=1)
    ax.add_patch(patch)


def draw_node(ax, x, y, w, h, colour):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.002",
        facecolor=colour, edgecolor="white", linewidth=0.5, zorder=3)
    ax.add_patch(rect)


def layout_sub_nodes(parent_y, parent_h, sub_a_values, sub_gap,
                     total_available_h, min_node_h=0.008):
    """Compute y-positions and heights for sub-term nodes,
    centred vertically around the parent node midpoint.

    Args:
        parent_y: bottom y of parent node
        parent_h: height of parent node
        sub_a_values: array of 'a' values for sub-terms
        sub_gap: gap between sub-term nodes
        total_available_h: max total height for this group's sub-terms + gaps
        min_node_h: minimum node height

    Returns:
        list of (y, h) tuples for each sub-term
    """
    n = len(sub_a_values)
    a = np.array(sub_a_values, dtype=float)

    # Scale heights: sqrt to compress range, then normalise
    sqrt_a = np.sqrt(a)
    total_gap = sub_gap * (n - 1)
    available_for_nodes = total_available_h - total_gap
    heights = sqrt_a / sqrt_a.sum() * available_for_nodes
    heights = np.maximum(heights, min_node_h)

    # Recompute total height
    total_h = heights.sum() + total_gap

    # Centre the group around the parent midpoint
    parent_mid = parent_y + parent_h / 2
    group_bottom = parent_mid - total_h / 2

    positions = []
    y_cursor = group_bottom
    for h in heights:
        positions.append((y_cursor, h))
        y_cursor += h + sub_gap

    return positions


def main():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_parent = 0.15
    x_sub = 0.50
    node_w = 0.025

    # Three test parents at different y positions
    parents = [
        ("skeleton (N=48)", 0.70, 0.18, "#00838F",
         [("decreased bone mineral density", 18, "#388E3C"),
          ("abnormal bone mineral density", 21, "#388E3C"),
          ("abnormal bone structure", 34, "#388E3C"),
          ("abnormal skeleton morphology", 46, "#388E3C")]),

        ("behavior/neurological (N=95)", 0.35, 0.22, "#1565C0",
         [("decreased startle reflex", 39, "#D32F2F"),
          ("abnormal startle reflex", 51, "#D32F2F"),
          ("absent pinna reflex", 15, "#D32F2F"),
          ("head bobbing", 6, "#F57C00"),
          ("hyperactivity", 31, "#388E3C"),
          ("abnormal gait", 11, "#388E3C")]),

        ("immune system (N=44)", 0.08, 0.12, "#AD1457",
         [("abnormal leukocyte cell number", 33, "#388E3C"),
          ("increased leukocyte cell number", 26, "#388E3C"),
          ("increased granulocyte number", 16, "#388E3C")]),
    ]

    for name, py, ph, pcolour, subs in parents:
        # Draw parent
        draw_node(ax, x_parent, py, node_w, ph, pcolour)
        ax.text(x_parent - 0.01, py + ph / 2, name,
                ha="right", va="center", fontsize=9)

        # Layout sub-term nodes
        sub_a = [a for _, a, _ in subs]
        group_available_h = ph * 1.8  # allow group to be taller than parent
        positions = layout_sub_nodes(py, ph, sub_a,
                                      sub_gap=0.012,
                                      total_available_h=group_available_h,
                                      min_node_h=0.010)

        # Normalise flow heights within parent
        a_arr = np.array(sub_a, dtype=float)
        a_norm = a_arr / a_arr.sum()
        flow_heights = a_norm * ph

        # Draw flows (stacked within parent)
        y_cursor = py
        for i, ((sy, sh), (sname, a_val, scolour)) in enumerate(zip(positions, subs)):
            fh = flow_heights[i]
            draw_flow(ax, x_parent + node_w, y_cursor, fh,
                      x_sub, sy, sh, scolour, alpha=0.28)
            y_cursor += fh

            # Draw sub-term node
            draw_node(ax, x_sub, sy, node_w, sh, scolour)
            ax.text(x_sub + node_w + 0.01, sy + sh / 2,
                    f"{sname} (N={a_val})",
                    ha="left", va="center", fontsize=8)

    ax.set_title("Test: Sized & Centred Stage 3 Nodes",
                 fontsize=14, fontweight="bold")

    out = "/Users/liambarrett/github/impc-abr/scripts/abr_analysis/enrichment/results/test_stage3_layout.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
