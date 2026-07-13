#!/usr/bin/env python3
"""
Generate complete lollipop plot of ARI bootstrap stability for ALL tied covariance models (k=3-12).
Manually extracts metrics from failed pipelines that still have stability scores.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Set publication-quality defaults
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["lines.linewidth"] = 2.5

# Color scheme - professional blue
BLUE_COLOR = "#2E86AB"


def load_all_tied_models(results_dir):
    """Load ALL tied covariance models, including those with failed analysis."""
    results_dir = Path(results_dir)

    data = []

    # Check k=3 to k=12
    for k in range(3, 13):
        model_dir = results_dir / f"gmm_k{k}_tied"
        metrics_file = model_dir / "metrics.json"

        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

                data.append(
                    {
                        "n_components": metrics["n_components"],
                        "stability_score": metrics["stability_score"],
                        "bic": metrics["bic"],
                        "silhouette": metrics["silhouette"],
                    }
                )

                print(f"✓ k={k}: ARI = {metrics['stability_score']:.4f}")

            except Exception as e:
                print(f"✗ k={k}: Error loading metrics - {e}")
        else:
            print(f"✗ k={k}: No metrics.json found")

    df = pd.DataFrame(data).sort_values("n_components")
    return df


def plot_lollipop_horizontal(df, output_dir):
    """Create horizontal lollipop plot with ARI on Y-axis."""
    fig, ax = plt.subplots(figsize=(10, 7))

    k_values = df["n_components"].values
    ari_scores = df["stability_score"].values

    # Create lollipop stems (vertical lines from 0 to ARI value)
    for k, score in zip(k_values, ari_scores):
        ax.plot([k, k], [0, score], color=BLUE_COLOR, linewidth=3, alpha=0.8, zorder=2)

    # Create lollipop heads (markers)
    ax.scatter(
        k_values,
        ari_scores,
        s=250,
        color=BLUE_COLOR,
        edgecolor="black",
        linewidth=2,
        zorder=3,
        alpha=0.9,
    )

    # Add value labels above markers
    for k, score in zip(k_values, ari_scores):
        ax.text(
            k,
            score + 0.04,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Formatting
    ax.set_xlabel("Number of Clusters (k)", fontweight="bold", fontsize=14)
    ax.set_ylabel("Bootstrap Stability (ARI)", fontweight="bold", fontsize=14)
    ax.set_title(
        "Cluster Stability: Tied Covariance Models",
        fontweight="bold",
        pad=20,
        fontsize=16,
    )

    # Set axis limits
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(2.5, 12.5)

    # Set x-axis ticks for all k values
    ax.set_xticks(k_values)

    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.8, zorder=1)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save with high resolution
    output_path_png = output_dir / "ari_stability_lollipop_complete.png"
    output_path_eps = output_dir / "ari_stability_lollipop_complete.eps"

    fig.savefig(output_path_png, dpi=1200, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path_eps, format="eps", bbox_inches="tight")
    plt.close(fig)

    print(f"\n✓ Saved: {output_path_png.name} (1200 dpi)")
    print(f"✓ Saved: {output_path_eps.name}")

    return output_path_png, output_path_eps


def main():
    # Configuration
    results_dir = Path(__file__).resolve().parent / "results" / "june_23_2025"
    output_dir = results_dir / "ari_stability_plots"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COMPLETE ARI LOLLIPOP PLOT (k=3-12, Tied Covariance)")
    print("=" * 60 + "\n")

    # Load all tied models
    print("Loading tied covariance model metrics...")
    print("-" * 40)
    df = load_all_tied_models(results_dir)
    print("-" * 40)

    if len(df) == 0:
        print("\n✗ No tied covariance models found!")
        return

    print(f"\n✓ Loaded {len(df)} tied covariance models")
    print(f"  k-range: {df['n_components'].min()}-{df['n_components'].max()}\n")

    # Summary statistics
    print("STABILITY SUMMARY:")
    print("-" * 40)
    print(
        f"  Best: k={df.loc[df['stability_score'].idxmax(), 'n_components']:.0f} (ARI = {df['stability_score'].max():.4f})"
    )
    print(
        f"  Worst: k={df.loc[df['stability_score'].idxmin(), 'n_components']:.0f} (ARI = {df['stability_score'].min():.4f})"
    )
    print(f"  Mean: {df['stability_score'].mean():.4f}")
    print(f"  Median: {df['stability_score'].median():.4f}")
    high_stability = (df["stability_score"] >= 0.8).sum()
    print(f"  High stability (≥0.8): {high_stability}/{len(df)} models")
    print("-" * 40)

    # Generate plot
    print("\nGenerating complete lollipop plot...")
    plot_lollipop_horizontal(df, output_dir)

    # Save CSV summary
    summary_path = output_dir / "ari_stability_complete_summary.csv"
    df[["n_components", "stability_score", "bic", "silhouette"]].to_csv(
        summary_path, index=False
    )
    print(f"✓ Saved: {summary_path.name}")

    print("\n" + "=" * 60)
    print("✓ Complete plot generation finished!")
    print(f"✓ Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
