#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot All Bootstrap Results

Batch process all bootstrapped models with the enhanced plotter.

Usage:
    python scripts/abr_utils/plot_all_bootstrap_results.py results/bootstrap_incremental_Pank2
"""

import sys
import subprocess
from pathlib import Path

def plot_all_models(bootstrap_dir):
    """Plot all models in a bootstrap results directory."""
    bootstrap_path = Path(bootstrap_dir)

    if not bootstrap_path.exists():
        print(f"ERROR: Directory not found: {bootstrap_path}")
        return

    # Find all model directories (those containing trace.nc)
    model_dirs = []
    for item in bootstrap_path.rglob("trace.nc"):
        model_dir = item.parent
        model_dirs.append(model_dir)

    print(f"Found {len(model_dirs)} model directories to plot:")

    for model_dir in model_dirs:
        print(f"\n📊 Plotting: {model_dir.name}")

        try:
            # Run enhanced plotter
            result = subprocess.run([
                "python", "scripts/abr_utils/enhanced_abr_plotter.py", str(model_dir)
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print(f"  ✅ SUCCESS: Plots created in {model_dir}/enhanced_plots/")
            else:
                print(f"  ❌ FAILED: {result.stderr}")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")

    print(f"\n🎨 Batch plotting complete! Processed {len(model_dirs)} models.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/plot_all_bootstrap_results.py <bootstrap_directory>")
        print("Example: python scripts/plot_all_bootstrap_results.py results/bootstrap_incremental_Pank2")
        sys.exit(1)

    plot_all_models(sys.argv[1])