#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced ABR Visualization Script

This script loads pre-trained Bayesian ABR models and creates enhanced visualizations
with improved aesthetics and separate plots for each analysis component.

Author: Liam Barrett
Version: 1.0.0
"""

import argparse
import json
import re
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

# Nord theme color palette for modern styling
NORD_COLORS = {
    'nord_blue': '#5e81ac',
    'nord_blue_light': '#8ba4cc',
    'nord_blue_dark': '#4c6889',
    'nord_red': '#BF616A',
    'nord_red_light': '#D08A8F',
    'sage_green': '#7ba05b',
    'sage_green_light': '#93bc85',
    'warm_orange': '#e89611',
    'warm_orange_light': '#efb776',
    'modern_teal': '#14b8a6',
    'modern_teal_light': '#2dd4bf',
    'nord_grey': '#6b7280',
    'nord_grey_light': '#9ca3af',
    'nord_grey_dark': '#374151',
    'background': '#fafbfc',
    'text_dark': '#2e3440'
}


class EnhancedABRPlotter:
    """Enhanced plotter for Bayesian ABR analysis results."""

    def __init__(self, model_directory):
        """Initialize plotter with model directory path."""
        self.model_dir = Path(model_directory)
        self.trace = None
        self.model_spec = None
        self.freq_labels = None
        self.gene_info = None

        # Set modern plotting style
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = NORD_COLORS['background']
        plt.rcParams['axes.facecolor'] = NORD_COLORS['background']
        plt.rcParams['text.color'] = NORD_COLORS['text_dark']
        plt.rcParams['axes.labelcolor'] = NORD_COLORS['text_dark']
        plt.rcParams['xtick.color'] = NORD_COLORS['text_dark']
        plt.rcParams['ytick.color'] = NORD_COLORS['text_dark']
        plt.rcParams['axes.edgecolor'] = NORD_COLORS['nord_grey']
        plt.rcParams['grid.color'] = NORD_COLORS['nord_grey_light']
        plt.rcParams['grid.alpha'] = 0.3

    def load_model(self):
        """Load model trace and specifications."""
        # Load trace
        trace_path = self.model_dir / "trace.nc"
        if not trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")

        self.trace = az.from_netcdf(trace_path)

        # Load model specifications
        spec_path = self.model_dir / "model_spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"Model spec file not found: {spec_path}")

        with open(spec_path, 'r', encoding='utf-8') as f:
            self.model_spec = json.load(f)

        # Extract frequency labels
        freq_list = self.model_spec['frequencies']
        self.freq_labels = [freq.split()[0].replace('kHz-evoked', '') for freq in freq_list]

        # Parse gene information from directory path
        self._parse_gene_info()

    def _parse_gene_info(self):
        """Parse gene, allele, zygosity, center, and gender info from directory name."""
        # Extract from the directory structure
        # Format: Gene_Allele_Zygosity_Center or similar
        dir_name = self.model_dir.name

        # Split by underscore, but handle complex allele names
        if '_' in dir_name:
            parts = dir_name.split('_')
            gene = parts[0]

            # Handle complex allele names that may contain underscores
            if len(parts) >= 4:
                # Assume: Gene_Allele_Zygosity_Center
                allele = parts[1]
                zygosity = parts[2]
                center = '_'.join(parts[3:])
            else:
                allele = parts[1] if len(parts) > 1 else 'Unknown'
                zygosity = parts[2] if len(parts) > 2 else 'Unknown'
                center = 'Unknown'
        else:
            # Fallback for non-standard naming
            gene = dir_name
            allele = 'Unknown'
            zygosity = 'Unknown'
            center = 'Unknown'

        # Detect gender splits from directory name (if present)
        gender = 'All'
        if 'male' in dir_name.lower() and 'female' not in dir_name.lower():
            gender = 'Male'
        elif 'female' in dir_name.lower():
            gender = 'Female'

        self.gene_info = {
            'gene': gene,
            'allele': allele,
            'zygosity': zygosity,
            'center': center,
            'gender': gender
        }

    def _get_title_prefix(self):
        """Generate title prefix with gene information."""
        gene = self.gene_info['gene']
        allele = self.gene_info['allele']
        zygosity = self.gene_info['zygosity']
        center = self.gene_info['center']
        gender = self.gene_info['gender']
        return f"{gene} | {allele} | {zygosity} | {center} | {gender}"

    def _save_dual_format(self, fig, base_path):
        """Save figure in both PNG (1200 DPI) and EPS (vector) formats."""
        base_path = Path(base_path)

        # Save PNG at 1200 DPI
        png_path = base_path.with_suffix('.png')
        fig.savefig(png_path, dpi=1200, bbox_inches='tight',
                   facecolor=NORD_COLORS['background'])

        # Save EPS at 1200 DPI equivalent
        eps_path = base_path.with_suffix('.eps')
        fig.savefig(eps_path, format='eps', bbox_inches='tight',
                   facecolor=NORD_COLORS['background'])

        print(f"Saved: {png_path} (PNG, 1200 DPI)")
        print(f"Saved: {eps_path} (EPS, vector format)")

    def plot_abr_profiles(self, save_path=None):
        """Create enhanced ABR profiles plot."""
        # Extract raw data from trace observed_data
        control_data = np.column_stack([self.trace.observed_data[f'controls_{i}'].values for i in range(5)])
        mutant_data = np.column_stack([self.trace.observed_data[f'mutants_{i}'].values for i in range(5)])

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(self.freq_labels))

        # Control profiles - Nord blue theme
        control_mean = np.mean(control_data, axis=0)
        control_std = np.std(control_data, axis=0)
        ax.fill_between(x, control_mean - 2*control_std, control_mean + 2*control_std,
                       alpha=0.3, color=NORD_COLORS['nord_grey_light'],
                       label='Control 95% CI')
        ax.plot(x, control_mean, color=NORD_COLORS['nord_grey_dark'],
                linewidth=2.5, label='Control Mean')

        # Mutant profiles - warm orange theme
        mutant_mean = np.mean(mutant_data, axis=0)
        mutant_std = np.std(mutant_data, axis=0)
        ax.fill_between(x, mutant_mean - 2*mutant_std, mutant_mean + 2*mutant_std,
                       alpha=0.3, color=NORD_COLORS['nord_red_light'],
                       label='Mutant 95% CI')
        ax.plot(x, mutant_mean, color=NORD_COLORS['nord_red'],
                linewidth=2.5, label='Mutant Mean')

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels([f'{label} kHz' for label in self.freq_labels])
        ax.set_ylabel('ABR Threshold (dB SPL)')
        ax.set_ylim(-10, 100)
        ax.set_yticks(np.arange(-10, 101, 10))
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{self._get_title_prefix()} - ABR Profiles')

        # Move legend outside plot area
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            self._save_dual_format(fig, save_path)

        return fig

    def plot_posterior_distribution(self, save_path=None):
        """Create enhanced posterior distribution plot with HDI lines."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get posterior samples
        p_hl_samples = self.trace.posterior['p_hearing_loss'].values.flatten()

        # Plot posterior distribution with nord light blue
        ax.hist(p_hl_samples, bins=50, density=True, alpha=0.7,
                color=NORD_COLORS['nord_blue_light'],
                edgecolor=NORD_COLORS['nord_blue'], linewidth=1.2)

        # Get HDI values from model spec
        mean_val = self.model_spec['parameters']['p_hearing_loss']['mean']
        hdi_lower = self.model_spec['parameters']['p_hearing_loss']['hdi_3%']
        hdi_upper = self.model_spec['parameters']['p_hearing_loss']['hdi_97%']

        # Add vertical lines for mean and HDI with specified Nord colors
        ax.axvline(mean_val, color=NORD_COLORS['nord_grey_dark'], linestyle='-',
                  linewidth=3, label=f'Mean: {mean_val:.3f}')
        ax.axvline(hdi_lower, color=NORD_COLORS['nord_grey'], linestyle='--',
                  linewidth=2.5, label=f'HDI Lower: {hdi_lower:.3f}')
        ax.axvline(hdi_upper, color=NORD_COLORS['nord_grey'], linestyle='--',
                  linewidth=2.5, label=f'HDI Upper: {hdi_upper:.3f}')

        # Formatting
        ax.set_xlabel('Probability of Hearing Loss')
        ax.set_ylabel('Density')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{self._get_title_prefix()} - Posterior Probability of Hearing Loss')

        # Move legend outside plot area
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            self._save_dual_format(fig, save_path)

        return fig

    def plot_effect_size(self, save_path=None):
        """Create enhanced effect size forest plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract effect size data from trace
        hl_shift_samples = self.trace.posterior['hl_shift']

        # Calculate summary statistics for each frequency
        means = []
        hdi_lowers = []
        hdi_uppers = []

        for i in range(5):
            samples = hl_shift_samples.isel(hl_shift_dim_0=i).values.flatten()
            means.append(np.mean(samples))
            hdi = az.hdi(samples, hdi_prob=0.94)  # 94% HDI to match the 3% and 97% from model spec
            hdi_lowers.append(hdi[0])
            hdi_uppers.append(hdi[1])

        # Create horizontal forest plot (frequency on x-axis, effect size on y-axis)
        x_pos = np.arange(5)
        freq_labels_khz = [f'{label}' for label in self.freq_labels]

        # Plot means and error bars with Nord blue theme
        ax.errorbar(x_pos, means, yerr=[np.array(means) - np.array(hdi_lowers),
                                       np.array(hdi_uppers) - np.array(means)],
                   fmt='o', color=NORD_COLORS['nord_blue'], capsize=6,
                   capthick=2.5, markersize=10, linewidth=2.5)

        # Add horizontal line at zero
        ax.axhline(0, color=NORD_COLORS['nord_grey_dark'], linestyle='-',
                  alpha=0.7, linewidth=1.5)

        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{label} kHz' for label in freq_labels_khz])
        ax.set_ylabel('Hearing Loss Effect Size (dB)')
        ax.set_xlabel('Frequency')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(f'{self._get_title_prefix()} - Hearing Loss Effect Size')

        plt.tight_layout()

        if save_path:
            self._save_dual_format(fig, save_path)

        return fig

    def create_all_plots(self, output_dir=None):
        """Create all three enhanced plots."""
        if output_dir is None:
            output_dir = self.model_dir / "enhanced_plots"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # Create plots
        print("Creating ABR profiles plot...")
        fig1 = self.plot_abr_profiles(output_dir / "abr_profiles_enhanced.png")
        plt.close(fig1)

        print("Creating posterior distribution plot...")
        fig2 = self.plot_posterior_distribution(output_dir / "posterior_distribution_enhanced.png")
        plt.close(fig2)

        print("Creating effect size plot...")
        fig3 = self.plot_effect_size(output_dir / "effect_size_enhanced.png")
        plt.close(fig3)

        print(f"All plots saved to: {output_dir}")

        return output_dir


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Create enhanced visualizations from saved Bayesian ABR models"
    )
    parser.add_argument(
        "model_directory",
        help="Path to directory containing trace.nc and model_spec.json files"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for plots (default: model_directory/enhanced_plots)"
    )

    args = parser.parse_args()

    try:
        # Create plotter and load model
        plotter = EnhancedABRPlotter(args.model_directory)
        plotter.load_model()

        # Create all plots
        output_dir = plotter.create_all_plots(args.output_dir)

        print(f"Successfully created enhanced plots in: {output_dir}")

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()