#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Control Audiograms by Phenotyping Centre

Plots the mean wild-type (control) ABR audiogram for each phenotyping centre,
with 95% confidence bands, to illustrate the between-centre variation in control
data that persists despite a shared standard operating procedure.

Each control mouse contributes a 5-dimensional audiogram (thresholds at 6, 12,
18, 24, and 30 kHz). Only controls with complete data at all five frequencies
are used. By default all controls are included; use --procedure to restrict to
a single procedure version (e.g. IMPC_ABR_001).

Follows the project plotting conventions used by enhanced_abr_plotter.py:
Nord-inspired palette, dB SPL y-axis fixed to -10..100, dual PNG (1200 DPI) +
EPS (vector) output.

Usage:
    python scripts/abr_utils/plot_control_audiograms_by_centre.py \
        --data data/processed/abr_full_data.csv \
        --output docs/figures/control_audiograms_by_centre

Author: Liam Barrett
Version: 1.0.0
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FREQ_COLS = [
    "6kHz-evoked ABR Threshold",
    "12kHz-evoked ABR Threshold",
    "18kHz-evoked ABR Threshold",
    "24kHz-evoked ABR Threshold",
    "30kHz-evoked ABR Threshold",
]
FREQ_LABELS = ["6", "12", "18", "24", "30"]

# Colourblind-friendly qualitative palette (Paul Tol) extended to 11 entries
# so every centre is visually distinguishable.
CENTRE_PALETTE = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
    "#AA3377", "#BBBBBB", "#EE8866", "#44BB99", "#AA4499",
    "#332288",
]

# Background / text styling, mirroring enhanced_abr_plotter.py
BACKGROUND = "#fafbfc"
TEXT_DARK = "#2e3440"
GREY = "#6b7280"
GREY_LIGHT = "#9ca3af"


def _apply_style():
    plt.style.use("default")
    plt.rcParams["figure.facecolor"] = BACKGROUND
    plt.rcParams["axes.facecolor"] = BACKGROUND
    plt.rcParams["text.color"] = TEXT_DARK
    plt.rcParams["axes.labelcolor"] = TEXT_DARK
    plt.rcParams["xtick.color"] = TEXT_DARK
    plt.rcParams["ytick.color"] = TEXT_DARK
    plt.rcParams["axes.edgecolor"] = GREY
    plt.rcParams["grid.color"] = GREY_LIGHT
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["font.size"] = 11


def load_controls(data_path, procedure=None):
    """Load controls with complete 5-frequency data, optionally one procedure."""
    df = pd.read_csv(data_path, low_memory=False)
    ctrl = df[df["biological_sample_group"] == "control"].copy()
    if procedure is not None:
        ctrl = ctrl[ctrl["procedure_stable_id"] == procedure].copy()
    ctrl = ctrl.dropna(subset=FREQ_COLS)
    return ctrl


def summarise_by_centre(ctrl):
    """Return per-centre mean, 95% CI half-width, and n across frequencies."""
    summaries = {}
    for centre, grp in ctrl.groupby("phenotyping_center"):
        vals = grp[FREQ_COLS].to_numpy(dtype=float)
        n = vals.shape[0]
        mean = vals.mean(axis=0)
        sem = stats.sem(vals, axis=0)
        # 95% CI of the mean
        ci = sem * stats.t.ppf(0.975, df=n - 1) if n > 1 else np.zeros_like(mean)
        summaries[centre] = {"mean": mean, "ci": ci, "n": n}
    # Order centres by total control n (largest first) for a stable legend
    return dict(sorted(summaries.items(), key=lambda kv: -kv[1]["n"]))


def plot_faceted(summaries, output_base, procedure=None, ncols=4):
    """One subplot per centre, black/grey, no colour or legend.

    Each subplot is titled with its centre. Dots and the mean line are black;
    the 95% CI band is grey. Axes are shared for direct visual comparison.
    """
    _apply_style()
    x = np.arange(len(FREQ_LABELS))
    centres = list(summaries.keys())
    n = len(centres)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, centre in zip(axes, centres):
        s = summaries[centre]
        ax.fill_between(
            x, s["mean"] - s["ci"], s["mean"] + s["ci"],
            color=GREY_LIGHT, alpha=0.5, linewidth=0, zorder=1,
        )
        ax.plot(
            x, s["mean"], color="black", linewidth=1.6, marker="o",
            markersize=4, markeredgecolor="white", markeredgewidth=0.5,
            zorder=3,
        )
        ax.set_title(f"{centre} (n={s['n']})", fontsize=10, color=TEXT_DARK)
        ax.set_ylim(-10, 100)
        ax.set_yticks(np.arange(-10, 101, 20))
        ax.set_xticks(x)
        ax.set_xticklabels(FREQ_LABELS)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Hide any unused panels
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.supxlabel("Frequency (kHz)", fontsize=11)
    fig.supylabel("ABR Threshold (dB SPL)", fontsize=11)
    suptitle = "Control audiograms by phenotyping centre"
    if procedure is not None:
        suptitle += f"  ({procedure})"
    fig.suptitle(suptitle, fontsize=13, color=TEXT_DARK)
    fig.tight_layout()

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    eps_path = output_base.with_suffix(".eps")
    fig.savefig(png_path, dpi=1200, bbox_inches="tight", facecolor=BACKGROUND)
    fig.savefig(eps_path, format="eps", bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"Saved: {png_path} (PNG, 1200 DPI)")
    print(f"Saved: {eps_path} (EPS, vector)")


def plot(summaries, output_base, procedure=None):
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(FREQ_LABELS))

    for i, (centre, s) in enumerate(summaries.items()):
        colour = CENTRE_PALETTE[i % len(CENTRE_PALETTE)]
        ax.fill_between(
            x, s["mean"] - s["ci"], s["mean"] + s["ci"],
            color=colour, alpha=0.12, linewidth=0, zorder=1,
        )
        ax.plot(
            x, s["mean"], color=colour, linewidth=2.0, marker="o",
            markersize=5, markeredgecolor="white", markeredgewidth=0.6,
            label=f"{centre} (n={s['n']})", zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl} kHz" for lbl in FREQ_LABELS])
    ax.set_xlabel("Frequency")
    ax.set_ylabel("ABR Threshold (dB SPL)")
    ax.set_ylim(-10, 100)
    ax.set_yticks(np.arange(-10, 101, 10))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6)

    title = "Control audiograms by phenotyping centre"
    if procedure is not None:
        title += f"  ({procedure})"
    ax.set_title(title, fontsize=13, color=TEXT_DARK, pad=12)

    ax.legend(
        title="Centre", loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=False, fontsize=9, title_fontsize=10,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    eps_path = output_base.with_suffix(".eps")
    fig.savefig(png_path, dpi=1200, bbox_inches="tight", facecolor=BACKGROUND)
    fig.savefig(eps_path, format="eps", bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"Saved: {png_path} (PNG, 1200 DPI)")
    print(f"Saved: {eps_path} (EPS, vector)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/abr_full_data.csv",
                        help="Path to ABR data CSV.")
    parser.add_argument("--output",
                        default="docs/figures/control_audiograms_by_centre",
                        help="Output path base (no extension).")
    parser.add_argument("--procedure", default=None,
                        help="Restrict to one procedure_stable_id "
                             "(e.g. IMPC_ABR_001). Default: all controls.")
    parser.add_argument("--faceted", action="store_true",
                        help="One subplot per centre (black/grey, no legend) "
                             "instead of a single overlaid panel.")
    args = parser.parse_args()

    ctrl = load_controls(args.data, procedure=args.procedure)
    summaries = summarise_by_centre(ctrl)

    total_n = sum(s["n"] for s in summaries.values())
    print(f"Controls (complete 5-freq): {total_n} across "
          f"{len(summaries)} centres"
          + (f" [{args.procedure}]" if args.procedure else ""))
    for centre, s in summaries.items():
        print(f"  {centre:14s} n={s['n']:5d}  "
              f"mean={np.array2string(s['mean'], precision=1)}")

    if args.faceted:
        plot_faceted(summaries, args.output, procedure=args.procedure)
    else:
        plot(summaries, args.output, procedure=args.procedure)


if __name__ == "__main__":
    main()
