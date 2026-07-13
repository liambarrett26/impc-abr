#!/usr/bin/env python3
"""
Aggregate results from parallel GMM model runs.

This script collects results from all trained models, performs model selection,
and generates final analysis reports.
"""

import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import analysis components
from analysis import AudiometricAnalyzer


class ModelAggregator:
    """
    Aggregates results from parallel GMM model training runs.

    Collects metrics, performs model selection, and generates comparative analysis.
    """

    def __init__(self, results_dir: str, output_dir: str, log_level: str = "INFO"):
        """
        Initialize aggregator.

        Args:
            results_dir: Directory containing individual model results
            output_dir: Directory for aggregated outputs
            log_level: Logging level
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging(log_level)

        # Storage for collected results
        self.model_results = []
        self.best_model = None

    def _setup_logging(self, log_level: str):
        """Setup logging for aggregation."""
        log_file = self.output_dir / "aggregation.log"

        # Create logger
        self.logger = logging.getLogger("ModelAggregator")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def collect_model_results(self) -> List[Dict[str, Any]]:
        """
        Collect results from all completed model runs.

        Returns:
            List of model result dictionaries
        """
        self.logger.info("Collecting results from parallel model runs")

        model_dirs = [
            d
            for d in self.results_dir.iterdir()
            if d.is_dir() and d.name.startswith("gmm_k")
        ]

        self.logger.info(f"Found {len(model_dirs)} model directories")

        for model_dir in sorted(model_dirs):
            self.logger.debug(f"Processing model directory: {model_dir.name}")

            # Check if model has metrics (more reliable than completed.txt)
            metrics_path = model_dir / "metrics.json"
            if not metrics_path.exists():
                self.logger.warning(f"Skipping model without metrics: {model_dir.name}")
                continue

            try:
                # Load metrics

                with open(metrics_path, "r") as f:
                    metrics = json.load(f)

                # Create result entry
                result = {
                    "model_name": model_dir.name,
                    "model_dir": str(model_dir),
                    "n_components": metrics["n_components"],
                    "covariance_type": metrics["covariance_type"],
                    "bic": metrics["bic"],
                    "aic": metrics["aic"],
                    "silhouette": metrics["silhouette"],
                    "log_likelihood": metrics["log_likelihood"],
                    "stability_score": metrics["stability_score"],
                    "training_time": metrics.get("training_time", 0),
                    "timestamp": metrics.get("timestamp", ""),
                }

                # Check for analysis results
                analysis_path = model_dir / "analysis_results.json"
                if analysis_path.exists():
                    with open(analysis_path, "r") as f:
                        analysis = json.load(f)
                        result["has_analysis"] = True
                        result["n_clusters"] = analysis.get(
                            "n_clusters", metrics["n_components"]
                        )
                        result["cluster_sizes"] = analysis.get("cluster_sizes", {})
                else:
                    result["has_analysis"] = False

                self.model_results.append(result)
                self.logger.info(
                    f"Collected results for {model_dir.name}: "
                    f"k={result['n_components']}, cov={result['covariance_type']}, "
                    f"BIC={result['bic']:.2f}, Silhouette={result['silhouette']:.3f}"
                )

            except Exception as e:
                self.logger.error(f"Error loading results from {model_dir.name}: {e}")
                continue

        self.logger.info(f"Collected results from {len(self.model_results)} models")
        return self.model_results

    def select_best_model(
        self, weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Select best model using weighted criteria.

        Args:
            weights: Custom weights for selection criteria

        Returns:
            Best model information
        """
        if not self.model_results:
            raise ValueError("No model results to select from")

        self.logger.info("Performing model selection")

        # Default weights
        if weights is None:
            weights = {"bic": 0.4, "aic": 0.2, "silhouette": 0.2, "stability": 0.2}

        # Extract metrics
        bic_scores = np.array([r["bic"] for r in self.model_results])
        aic_scores = np.array([r["aic"] for r in self.model_results])
        sil_scores = np.array([r["silhouette"] for r in self.model_results])
        stab_scores = np.array([r["stability_score"] for r in self.model_results])

        # Normalize (lower is better for BIC/AIC, higher for silhouette/stability)
        bic_norm = 1 - (bic_scores - bic_scores.min()) / (
            bic_scores.max() - bic_scores.min() + 1e-8
        )
        aic_norm = 1 - (aic_scores - aic_scores.min()) / (
            aic_scores.max() - aic_scores.min() + 1e-8
        )
        sil_norm = (sil_scores - sil_scores.min()) / (
            sil_scores.max() - sil_scores.min() + 1e-8
        )
        stab_norm = (stab_scores - stab_scores.min()) / (
            stab_scores.max() - stab_scores.min() + 1e-8
        )

        # Calculate combined scores
        combined_scores = (
            weights["bic"] * bic_norm
            + weights["aic"] * aic_norm
            + weights["silhouette"] * sil_norm
            + weights["stability"] * stab_norm
        )

        # Find best model
        best_idx = np.argmax(combined_scores)
        self.best_model = self.model_results[best_idx].copy()
        self.best_model["combined_score"] = float(combined_scores[best_idx])
        self.best_model["selection_weights"] = weights

        # Add normalized scores for transparency
        self.best_model["normalized_scores"] = {
            "bic": float(bic_norm[best_idx]),
            "aic": float(aic_norm[best_idx]),
            "silhouette": float(sil_norm[best_idx]),
            "stability": float(stab_norm[best_idx]),
        }

        self.logger.info(f"Selected best model: {self.best_model['model_name']}")
        self.logger.info(f"Combined score: {self.best_model['combined_score']:.3f}")

        return self.best_model

    def generate_separate_metric_plots(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate separate plots for each metric with improved formatting.

        Args:
            df: DataFrame with model results

        Returns:
            Dictionary mapping plot names to file paths
        """
        plot_files = {}

        # Define metrics and their properties
        metrics = [
            {
                "column": "bic",
                "title": "Bayesian Information Criterion",
                "ylabel": "BIC Score",
                "color": "#1f77b4",  # Blue
                "note": "(lower is better)",
            },
            {
                "column": "aic",
                "title": "Akaike Information Criterion",
                "ylabel": "AIC Score",
                "color": "#ff7f0e",  # Orange
                "note": "(lower is better)",
            },
            {
                "column": "silhouette",
                "title": "Silhouette Coefficient",
                "ylabel": "Silhouette Score",
                "color": "#2ca02c",  # Green
                "note": "(higher is better)",
            },
            {
                "column": "stability_score",
                "title": "Bootstrap Stability",
                "ylabel": "Stability Score",
                "color": "#d62728",  # Red
                "note": "(higher is better)",
            },
        ]

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create x positions and labels with proper ordering
            x = range(len(df))
            bars = ax.bar(x, df[metric["column"]], color=metric["color"], alpha=0.8)

            # Set x-axis labels with proper formatting
            ax.set_xticks(x)
            ax.set_xticklabels(
                [
                    f"k{r['n_components']}\n{r['covariance_type'][:4]}"
                    for _, r in df.iterrows()
                ],
                rotation=45,
                ha="right",
            )

            # Labels and title
            ax.set_ylabel(metric["ylabel"], fontsize=12)
            ax.set_title(f"{metric['title']}", fontsize=14, pad=20)
            ax.grid(True, alpha=0.3)

            # Value labels on bars removed as requested

            # Improve layout
            plt.tight_layout()

            # Save as PNG with high DPI
            png_path = self.output_dir / f"model_comparison_{metric['column']}.png"
            fig.savefig(png_path, dpi=1200, bbox_inches="tight", facecolor="white")
            plot_files[f'model_comparison_{metric["column"]}_png'] = str(png_path)

            plt.close(fig)

        return plot_files

    def generate_comparison_plots(self) -> Dict[str, str]:
        """
        Generate plots comparing all models.

        Returns:
            Dictionary mapping plot names to file paths
        """
        self.logger.info("Generating model comparison plots")

        plot_files = {}

        # Convert results to DataFrame for easier plotting
        df = pd.DataFrame(self.model_results)

        self.logger.info(f"Creating plots for {len(df)} models")
        self.logger.info(f"Models included: {list(df['model_name'])}")

        # Sort by n_components first, then by covariance type for proper ordering
        df = df.sort_values(["n_components", "covariance_type"])

        # Log the sorted order
        self.logger.info("Plot order after sorting:")
        for _, row in df.iterrows():
            self.logger.info(
                f"  {row['model_name']}: k={row['n_components']}, cov={row['covariance_type']}"
            )

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Model selection criteria comparison with individual colors
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Define colors for consistency with separate plots
        colors = {
            "bic": "#1f77b4",  # Blue
            "aic": "#ff7f0e",  # Orange
            "silhouette": "#2ca02c",  # Green
            "stability_score": "#d62728",  # Red
        }

        # BIC scores
        ax = axes[0, 0]
        x = range(len(df))
        bars = ax.bar(x, df["bic"], color=colors["bic"], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"k{r['n_components']}\n{r['covariance_type'][:4]}"
                for _, r in df.iterrows()
            ],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel("BIC Score")
        ax.set_title("Bayesian Information Criterion")
        ax.grid(True, alpha=0.3)

        # AIC scores
        ax = axes[0, 1]
        bars = ax.bar(x, df["aic"], color=colors["aic"], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"k{r['n_components']}\n{r['covariance_type'][:4]}"
                for _, r in df.iterrows()
            ],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel("AIC Score")
        ax.set_title("Akaike Information Criterion")
        ax.grid(True, alpha=0.3)

        # Silhouette scores
        ax = axes[1, 0]
        bars = ax.bar(x, df["silhouette"], color=colors["silhouette"], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"k{r['n_components']}\n{r['covariance_type'][:4]}"
                for _, r in df.iterrows()
            ],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Coefficient")
        ax.grid(True, alpha=0.3)

        # Stability scores
        ax = axes[1, 1]
        bars = ax.bar(
            x, df["stability_score"], color=colors["stability_score"], alpha=0.8
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"k{r['n_components']}\n{r['covariance_type'][:4]}"
                for _, r in df.iterrows()
            ],
            rotation=45,
            ha="right",
        )
        ax.set_ylabel("Stability Score")
        ax.set_title("Bootstrap Stability")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        comparison_path_png = self.output_dir / "model_comparison_metrics.png"
        comparison_path_eps = self.output_dir / "model_comparison_metrics.eps"
        fig.savefig(comparison_path_png, dpi=1200, bbox_inches="tight")
        fig.savefig(comparison_path_eps, format="eps", bbox_inches="tight")
        plt.close(fig)
        plot_files["model_comparison_png"] = str(comparison_path_png)
        plot_files["model_comparison_eps"] = str(comparison_path_eps)

        # Generate separate plots for each metric
        separate_plot_files = self.generate_separate_metric_plots(df)
        plot_files.update(separate_plot_files)

        # 2. Score evolution by k
        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by covariance type
        for cov_type in df["covariance_type"].unique():
            cov_df = df[df["covariance_type"] == cov_type].sort_values("n_components")
            ax.plot(
                cov_df["n_components"],
                cov_df["bic"],
                marker="o",
                label=f"{cov_type} - BIC",
                linewidth=2,
            )

        ax.set_xlabel("Number of Components (k)")
        ax.set_ylabel("BIC Score")
        ax.set_title("BIC Score Evolution by Number of Components")
        ax.legend()
        ax.grid(True, alpha=0.3)

        evolution_path_png = self.output_dir / "bic_evolution.png"
        evolution_path_eps = self.output_dir / "bic_evolution.eps"
        fig.savefig(evolution_path_png, dpi=1200, bbox_inches="tight")
        fig.savefig(evolution_path_eps, format="eps", bbox_inches="tight")
        plt.close(fig)
        plot_files["bic_evolution_png"] = str(evolution_path_png)
        plot_files["bic_evolution_eps"] = str(evolution_path_eps)

        # 3. Training time comparison
        if "training_time" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            bars = ax.bar(x, df["training_time"])
            ax.set_xticks(x)
            ax.set_xticklabels(
                [
                    f"k{r['n_components']}\n{r['covariance_type'][:4]}"
                    for _, r in df.iterrows()
                ],
                rotation=45,
            )
            ax.set_ylabel("Training Time (seconds)")
            ax.set_title("Model Training Time Comparison")
            ax.grid(True, alpha=0.3)

            time_path_png = self.output_dir / "training_time_comparison.png"
            time_path_eps = self.output_dir / "training_time_comparison.eps"
            fig.savefig(time_path_png, dpi=1200, bbox_inches="tight")
            fig.savefig(time_path_eps, format="eps", bbox_inches="tight")
            plt.close(fig)
            plot_files["training_time_png"] = str(time_path_png)
            plot_files["training_time_eps"] = str(time_path_eps)

        self.logger.info(f"Generated {len(plot_files)} comparison plots")
        return plot_files

    def generate_summary_report(self) -> str:
        """
        Generate comprehensive summary report.

        Returns:
            Path to generated report
        """
        self.logger.info("Generating summary report")

        report_path = self.output_dir / "model_selection_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("GMM MODEL SELECTION SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total models evaluated: {len(self.model_results)}\n\n")

            # Best model summary
            if self.best_model:
                f.write("SELECTED BEST MODEL\n")
                f.write("-" * 40 + "\n")
                f.write(f"Model: {self.best_model['model_name']}\n")
                f.write(f"Components: {self.best_model['n_components']}\n")
                f.write(f"Covariance: {self.best_model['covariance_type']}\n")
                f.write(f"Combined score: {self.best_model['combined_score']:.3f}\n\n")

                f.write("Metrics:\n")
                f.write(f"  BIC: {self.best_model['bic']:.2f}\n")
                f.write(f"  AIC: {self.best_model['aic']:.2f}\n")
                f.write(f"  Silhouette: {self.best_model['silhouette']:.3f}\n")
                f.write(f"  Stability: {self.best_model['stability_score']:.3f}\n")
                f.write(
                    f"  Training time: {self.best_model.get('training_time', 0):.1f}s\n\n"
                )

                f.write("Normalized scores:\n")
                for metric, score in self.best_model["normalized_scores"].items():
                    f.write(f"  {metric}: {score:.3f}\n")
                f.write("\n")

            # All models ranking
            f.write("ALL MODELS RANKING\n")
            f.write("-" * 40 + "\n")

            # Sort by BIC (primary criterion)
            sorted_models = sorted(self.model_results, key=lambda x: x["bic"])

            f.write(
                f"{'Rank':<6}{'Model':<20}{'BIC':<12}{'Silhouette':<12}{'Stability':<12}\n"
            )
            f.write("-" * 62 + "\n")

            for i, model in enumerate(sorted_models, 1):
                is_best = (
                    model["model_name"] == self.best_model["model_name"]
                    if self.best_model
                    else False
                )
                marker = " *" if is_best else ""
                f.write(
                    f"{i:<6}{model['model_name']:<20}"
                    f"{model['bic']:<12.2f}{model['silhouette']:<12.3f}"
                    f"{model['stability_score']:<12.3f}{marker}\n"
                )

            f.write("\n* = Selected best model\n\n")

            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")

            bic_values = [m["bic"] for m in self.model_results]
            sil_values = [m["silhouette"] for m in self.model_results]
            stab_values = [m["stability_score"] for m in self.model_results]

            f.write(f"BIC range: {min(bic_values):.2f} - {max(bic_values):.2f}\n")
            f.write(
                f"Silhouette range: {min(sil_values):.3f} - {max(sil_values):.3f}\n"
            )
            f.write(
                f"Stability range: {min(stab_values):.3f} - {max(stab_values):.3f}\n"
            )

            if "training_time" in self.model_results[0]:
                times = [m["training_time"] for m in self.model_results]
                f.write(f"Total training time: {sum(times):.1f}s\n")
                f.write(f"Average training time: {np.mean(times):.1f}s\n")

            f.write("\n" + "=" * 80 + "\n")

        self.logger.info(f"Report saved to {report_path}")
        return str(report_path)

    def save_aggregated_results(self) -> Dict[str, str]:
        """
        Save all aggregated results.

        Returns:
            Dictionary mapping result types to file paths
        """
        saved_files = {}

        # Save model comparison data
        comparison_df = pd.DataFrame(self.model_results)
        comparison_path = self.output_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        saved_files["model_comparison"] = str(comparison_path)

        # Save selection results
        if self.best_model:
            selection_path = self.output_dir / "model_selection_results.json"
            selection_data = {
                "best_model": self.best_model,
                "all_models": self.model_results,
                "selection_timestamp": datetime.now().isoformat(),
            }
            with open(selection_path, "w") as f:
                json.dump(selection_data, f, indent=2)
            saved_files["selection_results"] = str(selection_path)

            # Copy best model files to main results directory
            best_model_dir = Path(self.best_model["model_dir"])

            # Copy key files with clear naming
            files_to_copy = [
                ("model.pkl", "best_model.pkl"),
                ("cluster_labels.npy", "best_cluster_labels.npy"),
                ("cluster_probabilities.npy", "best_cluster_probabilities.npy"),
                ("metrics.json", "best_model_metrics.json"),
                ("analysis_results.json", "best_model_analysis.json"),
            ]

            for src_name, dst_name in files_to_copy:
                src_path = best_model_dir / src_name
                if src_path.exists():
                    dst_path = self.output_dir / dst_name
                    if src_name.endswith(".pkl"):
                        with open(src_path, "rb") as f_src:
                            with open(dst_path, "wb") as f_dst:
                                f_dst.write(f_src.read())
                    elif src_name.endswith(".npy"):
                        np.save(dst_path, np.load(src_path))
                    else:
                        with open(src_path, "r") as f_src:
                            with open(dst_path, "w") as f_dst:
                                f_dst.write(f_src.read())
                    saved_files[f"best_{src_name}"] = str(dst_path)

        self.logger.info(f"Saved {len(saved_files)} aggregated result files")
        return saved_files


def main():
    """Main entry point for result aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate results from parallel GMM model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Arguments
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing individual model results",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for aggregated results",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="JSON string with custom selection weights",
    )

    args = parser.parse_args()

    # Parse custom weights if provided
    weights = None
    if args.weights:
        try:
            weights = json.loads(args.weights)
        except:
            print(f"Warning: Could not parse weights '{args.weights}', using defaults")

    # Initialize aggregator
    aggregator = ModelAggregator(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        log_level=args.log_level,
    )

    try:
        # Collect results
        results = aggregator.collect_model_results()

        if not results:
            print("Error: No valid model results found")
            return 1

        # Select best model
        best_model = aggregator.select_best_model(weights)

        # Generate comparison plots
        plot_files = aggregator.generate_comparison_plots()

        # Generate report
        report_path = aggregator.generate_summary_report()

        # Save aggregated results
        saved_files = aggregator.save_aggregated_results()

        print(f"\nAggregation completed successfully")
        print(f"Best model: {best_model['model_name']}")
        print(f"Results saved to: {args.output_dir}")

        return 0

    except Exception as e:
        print(f"Error during aggregation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
