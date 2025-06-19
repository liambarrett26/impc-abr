#!/usr/bin/env python3
"""
Cluster number optimization for audiometric phenotype discovery.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class ClusterOptimizer:
    """Optimize the number of clusters for deep clustering."""
    
    def __init__(self, config: Dict):
        """Initialize cluster optimizer."""
        self.config = config
        self.k_range = tuple(config.get('k_range', [6, 18]))
        self.metrics = config.get('metrics', ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'elbow'])
        self.consensus_method = config.get('consensus_method', 'median')
        self.save_analysis = config.get('save_analysis', True)
        
    def extract_latent_features(self, model, dataloader, device):
        """Extract latent representations from pretrained model."""
        model.eval()
        latent_features = []
        
        logger.info("Extracting latent features for cluster optimization...")
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i % 10 == 0:
                    logger.info(f"Processing batch {i}/{len(dataloader)}")
                    
                features = batch['features'].to(device)
                output = model(features)
                latent_z = output['latent_z'].cpu().numpy()
                latent_features.append(latent_z)
        
        features_array = np.vstack(latent_features)
        logger.info(f"Extracted {features_array.shape[0]} latent features of dimension {features_array.shape[1]}")
        
        return features_array
    
    def compute_clustering_metrics(self, features: np.ndarray) -> pd.DataFrame:
        """Compute clustering metrics for different k values."""
        logger.info(f"Computing clustering metrics for k in range {self.k_range}")
        
        metrics_data = {
            'k': [],
            'inertia': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        for k in range(self.k_range[0], self.k_range[1] + 1):
            logger.info(f"Testing k={k}...")
            
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            # Basic metrics
            metrics_data['k'].append(k)
            metrics_data['inertia'].append(kmeans.inertia_)
            
            # Advanced metrics (require k > 1)
            if k > 1:
                metrics_data['silhouette'].append(silhouette_score(features, labels))
                metrics_data['davies_bouldin'].append(davies_bouldin_score(features, labels))
                metrics_data['calinski_harabasz'].append(calinski_harabasz_score(features, labels))
            else:
                metrics_data['silhouette'].append(0)
                metrics_data['davies_bouldin'].append(float('inf'))
                metrics_data['calinski_harabasz'].append(0)
        
        return pd.DataFrame(metrics_data)
    
    def find_elbow_point(self, values: np.ndarray) -> int:
        """Find elbow point using line distance method."""
        # Normalize values for comparison
        x = np.arange(len(values))
        y = np.array(values)
        y_norm = (y - y.min()) / (y.max() - y.min())
        
        # Line from first to last point
        coords = np.vstack([x, y_norm]).T
        first_point = coords[0]
        last_point = coords[-1]
        line_vec = last_point - first_point
        line_vec_norm = line_vec / np.linalg.norm(line_vec)
        
        # Find point with maximum distance from line
        distances = []
        for i in range(len(coords)):
            point_vec = coords[i] - first_point
            proj_length = np.dot(point_vec, line_vec_norm)
            proj = first_point + proj_length * line_vec_norm
            distance = np.linalg.norm(coords[i] - proj)
            distances.append(distance)
        
        return np.argmax(distances)
    
    def get_optimal_k_per_metric(self, metrics_df: pd.DataFrame) -> Dict[str, int]:
        """Get optimal k for each metric."""
        optimal_ks = {}
        
        if 'elbow' in self.metrics:
            elbow_idx = self.find_elbow_point(metrics_df['inertia'].values)
            optimal_ks['elbow'] = metrics_df['k'].iloc[elbow_idx]
        
        if 'silhouette' in self.metrics:
            optimal_ks['silhouette'] = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']
        
        if 'davies_bouldin' in self.metrics:
            optimal_ks['davies_bouldin'] = metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'k']
            
        if 'calinski_harabasz' in self.metrics:
            optimal_ks['calinski_harabasz'] = metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'k']
        
        return optimal_ks
    
    def get_consensus_k(self, optimal_ks: Dict[str, int]) -> int:
        """Get consensus optimal k using specified method."""
        k_values = list(optimal_ks.values())
        
        if self.consensus_method == 'median':
            consensus_k = int(np.median(k_values))
        elif self.consensus_method == 'mode':
            from scipy import stats
            consensus_k = int(stats.mode(k_values)[0][0])
        elif self.consensus_method == 'mean':
            consensus_k = int(np.round(np.mean(k_values)))
        else:
            # Default to median
            consensus_k = int(np.median(k_values))
        
        return consensus_k
    
    def plot_analysis(self, metrics_df: pd.DataFrame, optimal_ks: Dict[str, int], 
                     consensus_k: int, save_path: Optional[Path] = None):
        """Plot cluster optimization analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Inertia (Elbow curve)
        ax = axes[0, 0]
        ax.plot(metrics_df['k'], metrics_df['inertia'], 'bo-', linewidth=2, markersize=8)
        if 'elbow' in optimal_ks:
            ax.axvline(x=optimal_ks['elbow'], color='r', linestyle='--', linewidth=2,
                      label=f'Elbow at k={optimal_ks["elbow"]}')
        ax.axvline(x=consensus_k, color='orange', linestyle='-', linewidth=3,
                  label=f'Consensus k={consensus_k}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax.set_ylabel('Inertia', fontsize=12)
        ax.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Silhouette Score
        ax = axes[0, 1]
        ax.plot(metrics_df['k'], metrics_df['silhouette'], 'go-', linewidth=2, markersize=8)
        if 'silhouette' in optimal_ks:
            ax.axvline(x=optimal_ks['silhouette'], color='r', linestyle='--', linewidth=2,
                      label=f'Best at k={optimal_ks["silhouette"]}')
        ax.axvline(x=consensus_k, color='orange', linestyle='-', linewidth=3,
                  label=f'Consensus k={consensus_k}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title('Silhouette Analysis (Higher is Better)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Davies-Bouldin Score
        ax = axes[1, 0]
        ax.plot(metrics_df['k'], metrics_df['davies_bouldin'], 'ro-', linewidth=2, markersize=8)
        if 'davies_bouldin' in optimal_ks:
            ax.axvline(x=optimal_ks['davies_bouldin'], color='b', linestyle='--', linewidth=2,
                      label=f'Best at k={optimal_ks["davies_bouldin"]}')
        ax.axvline(x=consensus_k, color='orange', linestyle='-', linewidth=3,
                  label=f'Consensus k={consensus_k}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax.set_ylabel('Davies-Bouldin Score', fontsize=12)
        ax.set_title('Davies-Bouldin Score (Lower is Better)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calinski-Harabasz Score
        ax = axes[1, 1]
        ax.plot(metrics_df['k'], metrics_df['calinski_harabasz'], 'mo-', linewidth=2, markersize=8)
        if 'calinski_harabasz' in optimal_ks:
            ax.axvline(x=optimal_ks['calinski_harabasz'], color='r', linestyle='--', linewidth=2,
                      label=f'Best at k={optimal_ks["calinski_harabasz"]}')
        ax.axvline(x=consensus_k, color='orange', linestyle='-', linewidth=3,
                  label=f'Consensus k={consensus_k}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax.set_ylabel('Calinski-Harabasz Score', fontsize=12)
        ax.set_title('Calinski-Harabasz Score (Higher is Better)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Cluster Number Optimization - Consensus: k={consensus_k}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Analysis plot saved to {save_path}")
        
        return fig
    
    def optimize(self, model, dataloader, device, save_dir: Optional[Path] = None) -> int:
        """
        Run cluster number optimization.
        
        Returns:
            Optimal number of clusters
        """
        logger.info("Starting cluster number optimization...")
        
        # Extract latent features
        latent_features = self.extract_latent_features(model, dataloader, device)
        
        # Compute metrics
        metrics_df = self.compute_clustering_metrics(latent_features)
        
        # Find optimal k for each metric
        optimal_ks = self.get_optimal_k_per_metric(metrics_df)
        
        # Get consensus
        consensus_k = self.get_consensus_k(optimal_ks)
        
        # Log results
        logger.info("Cluster optimization results:")
        for metric, k in optimal_ks.items():
            logger.info(f"  {metric}: k={k}")
        logger.info(f"  Consensus ({self.consensus_method}): k={consensus_k}")
        
        # Save analysis if requested
        if self.save_analysis and save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            metrics_df.to_csv(save_dir / 'cluster_optimization_metrics.csv', index=False)
            
            # Save plot
            plot_path = save_dir / 'cluster_optimization_analysis.png'
            self.plot_analysis(metrics_df, optimal_ks, consensus_k, plot_path)
            
            # Save summary
            summary = {
                'consensus_k': consensus_k,
                'optimal_k_per_metric': optimal_ks,
                'consensus_method': self.consensus_method,
                'k_range': list(self.k_range),  # Convert tuple to list for YAML serialization
                'metrics_used': self.metrics
            }
            
            import yaml
            with open(save_dir / 'cluster_optimization_summary.yaml', 'w') as f:
                yaml.dump(summary, f, default_flow_style=False)
            
            logger.info(f"Cluster optimization analysis saved to {save_dir}")
        
        return consensus_k