#!/bin/env python
# -*- coding: utf-8 -*-
# abr_analysis/models/bayesian.py

"""
Bayesian analysis module for Auditory Brainstem Response (ABR) data.

This module implements a Bayesian statistical approach to analyse ABR profiles,
treating them as multivariate observations rather than individual frequency measurements.
It provides tools for fitting models, calculating evidence for hearing loss,
and visualising results with appropriate uncertainty quantification.

Author: Liam Barrett
Version: 1.0.1
"""


from pathlib import Path
import json
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

class BayesianABRAnalysis:
    """
    Bayesian model for analysing Auditory Brainstem Response (ABR) data.
    
    This class implements a mixture model approach that compares ABR profiles
    from mutant mice against control distributions. It estimates the probability
    of hearing loss and quantifies the evidence using Bayes factors, while also
    characterising the pattern and magnitude of hearing threshold shifts.
    
    Attributes:
        model: PyMC model object
        trace: Inference data containing posterior samples
        freq_cols: Column names for frequency measurements
    """
    def __init__(self):
        self.model = None
        self.trace = None
        self.freq_cols = None

    def fit(self, control_profiles, mutant_profiles, freq_cols):
        """Fit Bayesian model to control and mutant data."""
        self.freq_cols = freq_cols
        n_dims = control_profiles.shape[1]
        n_mutants = len(mutant_profiles)

        with pm.Model() as self.model:
            # Prior for control mean
            control_mean = pm.Normal('control_mean',
                                   mu=np.mean(control_profiles, axis=0),
                                   sigma=10,
                                   shape=n_dims)

            # Prior for control standard deviation (one per dimension)
            control_sd = pm.HalfNormal('control_sd',
                                     sigma=10,
                                     shape=n_dims)

            # Control covariance matrix with diagonal dominance
            # Not currently implemented
            #control_cov = pm.LKJCholeskyCov(
            #    'control_cov',
            #    n=n_dims,
            #    eta=2.0,  # Weakly favors diagonal matrix
            #    sd_dist=pm.HalfNormal.dist(sigma=20)
            #)

            # Prior for probability of hearing loss
            p_hearing_loss = pm.Beta('p_hearing_loss', alpha=1, beta=3)

            # Likelihood for controls using independent normal distributions
            for i in range(n_dims):
                pm.Normal(f'controls_{i}',
                         mu=control_mean[i],
                         sigma=control_sd[i],
                         observed=control_profiles[:, i])

            # Prior for hearing loss effect size (shift from control mean)
            hl_shift = pm.Normal('hl_shift',
                               mu=20,  # Expected shift for hearing loss
                               sigma=5,
                               shape=n_dims)

            # Define hearing loss mean
            hl_mean = pm.Deterministic('hl_mean',
                                     control_mean + hl_shift)

            # Binary indicator for each mutant
            z = pm.Bernoulli('z',
                            p=p_hearing_loss,
                            shape=n_mutants)

            # Model for mutants using independent normal distributions
            for i in range(n_dims):
                means_i = pm.math.switch(z,
                                       hl_mean[i],
                                       control_mean[i])

                pm.Normal(f'mutants_{i}',
                         mu=means_i,
                         sigma=control_sd[i],
                         observed=mutant_profiles[:, i])

            # Sample from posterior
            self.trace = pm.sample(
                2000,
                tune=1000,
                target_accept=0.9,
                return_inferencedata=True
            )

    def calculate_bayes_factor(self):
        """Calculate Bayes factor for hearing loss vs normal hearing."""
        p_hl = self.trace.posterior['p_hearing_loss'].values.flatten()

        # Estimate the odds ratio of p_hl > 0.5 vs p_hl <= 0.5
        # Count samples above and below 0.5
        n_samples = len(p_hl)
        n_hearing_loss = np.sum(p_hl > 0.5)
        n_normal = n_samples - n_hearing_loss

        # Calculate posterior odds
        posterior_odds = n_hearing_loss / max(n_normal, 1)  # Avoid division by zero

        # Calculate prior odds (under Beta(1,1), it's 1.0)
        prior_odds = 1.0

        # Bayes factor is the ratio of posterior odds to prior odds
        bayes_factor = posterior_odds / prior_odds

        return bayes_factor

    def get_summary_statistics(self):
        """Get summary statistics from the posterior."""
        summary = az.summary(self.trace,
                           var_names=['p_hearing_loss', 'hl_shift'])
        return summary

    def plot_results(self, control_profiles, mutant_profiles, gene_name=None):
        """Create comprehensive visualization of results."""
        fig = plt.figure(figsize=(15, 10))

        # 1. Raw Data Visualization
        ax1 = plt.subplot(221)
        self._plot_profiles(ax1, control_profiles, mutant_profiles, gene_name)

        # 2. Posterior Distribution of p_hearing_loss
        ax2 = plt.subplot(222)
        self._plot_posterior_distributions(ax2)

        # 3. Hearing Loss Effect Size
        ax3 = plt.subplot(223)
        self._plot_effect_size(ax3)

        # 4. Evidence Summary
        ax4 = plt.subplot(224)
        self._plot_evidence_summary(ax4)

        plt.tight_layout()
        return fig

    def _plot_profiles(self, ax, control_profiles, mutant_profiles, gene_name):
        """Plot raw ABR profiles with uncertainty."""
        x = np.arange(len(self.freq_cols))

        # Control profiles
        control_mean = np.mean(control_profiles, axis=0)
        control_std = np.std(control_profiles, axis=0)
        ax.fill_between(x, control_mean - 2*control_std, control_mean + 2*control_std,
                       alpha=0.2, color='blue', label='Control 95% CI')
        ax.plot(x, control_mean, 'b-', label='Control Mean')

        # Mutant profiles
        mutant_mean = np.mean(mutant_profiles, axis=0)
        mutant_std = np.std(mutant_profiles, axis=0)
        ax.fill_between(x, mutant_mean - 2*mutant_std, mutant_mean + 2*mutant_std,
                       alpha=0.2, color='red', label='Mutant 95% CI')
        ax.plot(x, mutant_mean, 'r-', label='Mutant Mean')

        ax.set_xticks(x)
        ax.set_xticklabels([col.split()[0] for col in self.freq_cols], rotation=45)
        ax.set_ylabel('ABR Threshold (dB SPL)')
        ax.set_title(f'ABR Profiles{f" - {gene_name}" if gene_name else ""}')
        ax.legend()

    def _plot_posterior_distributions(self, ax):
        """Plot posterior distribution of hearing loss probability."""
        az.plot_posterior(
            self.trace,
            var_names=['p_hearing_loss'],
            ax=ax
        )
        ax.set_title('Posterior Probability of Hearing Loss')

    def _plot_effect_size(self, ax):
        """Plot hearing loss effect size with uncertainty."""
        az.plot_forest(
            self.trace,
            var_names=['hl_shift'],
            ax=ax,
            combined=True
        )
        ax.set_title('Hearing Loss Effect Size')

    def _plot_evidence_summary(self, ax):
        """Plot summary of statistical evidence."""
        summary = self.get_summary_statistics()
        bf = self.calculate_bayes_factor()

        # Get probability of hearing loss and HDI
        p_hl = summary.loc['p_hearing_loss', 'mean']
        ci_lower = summary.loc['p_hearing_loss', 'hdi_3%']
        ci_upper = summary.loc['p_hearing_loss', 'hdi_97%']

        # Interpret Bayes factor
        if bf > 100:
            evidence = "Extreme"
        elif bf > 30:
            evidence = "Very Strong"
        elif bf > 10:
            evidence = "Strong"
        elif bf > 3:
            evidence = "Substantial"
        else:
            evidence = "Weak"

        txt = (
            f"Evidence for Hearing Loss:\n"
            f"Bayes Factor = {bf:.1f} ({evidence})\n\n"
            f"P(Hearing Loss | Data) = {p_hl:.3f}\n"
            f"95% HDI: [{ci_lower:.3f}, {ci_upper:.3f}]"
        )

        ax.text(0.5, 0.5, txt,
                ha='center', va='center',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        ax.axis('off')

    def get_model_specification(self):
        """Return a dictionary with model specifications for documentation."""
        if not hasattr(self, 'trace') or self.trace is None:
            return {"error": "Model has not been fitted yet"}

        summary = self.get_summary_statistics()
        spec = {
            "model_type": "Bayesian ABR Analysis",
            "parameters": {
                "p_hearing_loss": {
                    "mean": float(summary.loc['p_hearing_loss', 'mean']),
                    "hdi_3%": float(summary.loc['p_hearing_loss', 'hdi_3%']),
                    "hdi_97%": float(summary.loc['p_hearing_loss', 'hdi_97%'])
                },
                "bayes_factor": float(self.calculate_bayes_factor())
            },
            "frequencies": self.freq_cols,
            "hearing_loss_shifts": {}
        }

        # Add effect sizes for each frequency
        for i, freq in enumerate(self.freq_cols):
            freq_name = freq.split()[0]
            spec["hearing_loss_shifts"][freq_name] = {
                "mean": float(summary.loc[f'hl_shift[{i}]', 'mean']),
                "hdi_3%": float(summary.loc[f'hl_shift[{i}]', 'hdi_3%']),
                "hdi_97%": float(summary.loc[f'hl_shift[{i}]', 'hdi_97%'])
            }

        return spec

    def save_model(self, directory):
        """Save the model trace and specifications to the given directory."""
        if not hasattr(self, 'trace') or self.trace is None:
            print(f"Warning: No trace to save for model in directory {directory}")
            return False

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        try:
            # Save trace
            trace_path = directory / "trace.nc"
            self.trace.to_netcdf(trace_path)
            print(f"Successfully saved model trace to {trace_path}")

            # Save model specifications
            spec = self.get_model_specification()
            spec_path = directory / "model_spec.json"
            with open(spec_path, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)
            print(f"Successfully saved model specifications to {spec_path}")

            # Create README.md
            readme_path = directory / "README.md"
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write("# Bayesian ABR Analysis Model\n\n")
                f.write("## Probability of Hearing Loss\n")
                f.write(f"- Mean: {spec['parameters']['p_hearing_loss']['mean']:.3f}\n")
                hdi_low = spec['parameters']['p_hearing_loss']['hdi_3%']
                hdi_high = spec['parameters']['p_hearing_loss']['hdi_97%']
                f.write(f"- 95% HDI: [{hdi_low:.3f}, {hdi_high:.3f}]\n\n")
                f.write(f"## Bayes Factor: {spec['parameters']['bayes_factor']:.2f}\n\n")
                f.write("## Frequency-specific Hearing Loss Shifts\n")

                for freq, values in spec["hearing_loss_shifts"].items():
                    f.write(f"### {freq} kHz\n")
                    f.write(f"- Mean shift: {values['mean']:.2f} dB\n")
                    f.write(f"- 95% HDI: [{values['hdi_3%']:.2f}, {values['hdi_97%']:.2f}] dB\n\n")
            print(f"Successfully saved model README to {readme_path}")

            return True
        except Exception as e:
            print(f"Error saving model to {directory}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False