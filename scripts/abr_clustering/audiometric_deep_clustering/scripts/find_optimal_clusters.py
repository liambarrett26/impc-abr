#!/usr/bin/env python3
"""
Find optimal number of clusters for audiometric phenotype data.
Run this after VAE pretraining to determine the best k for clustering.
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist
import yaml
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.full_model import create_model
from data.dataloader import create_data_module
from utils.logging import setup_logging


def extract_latent_features(model, dataloader, device):
    """Extract latent representations from pretrained model."""
    model.eval()
    latent_features = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            output = model(features)
            latent_z = output['latent_z'].cpu().numpy()
            latent_features.append(latent_z)
    
    return np.vstack(latent_features)


def compute_elbow_metrics(features, k_range=(5, 20)):
    """Compute clustering metrics for different k values."""
    metrics = {
        'k': [],
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }
    
    for k in range(k_range[0], k_range[1] + 1):
        print(f"Testing k={k}...")
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Compute metrics
        metrics['k'].append(k)
        metrics['inertia'].append(kmeans.inertia_)
        
        if k > 1:  # Silhouette requires at least 2 clusters
            metrics['silhouette'].append(silhouette_score(features, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(features, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(features, labels))
        else:
            metrics['silhouette'].append(0)
            metrics['davies_bouldin'].append(float('inf'))
            metrics['calinski_harabasz'].append(0)
    
    return pd.DataFrame(metrics)


def find_elbow_point(values):
    """Find elbow point using kneedle algorithm."""
    # Normalize values
    x = np.arange(len(values))
    y = np.array(values)
    y_norm = (y - y.min()) / (y.max() - y.min())
    
    # Compute distances from line between first and last point
    coords = np.vstack([x, y_norm]).T
    first_point = coords[0]
    last_point = coords[-1]
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
    distances = []
    for i in range(len(coords)):
        point_vec = coords[i] - first_point
        proj_length = np.dot(point_vec, line_vec_norm)
        proj = first_point + proj_length * line_vec_norm
        distance = np.linalg.norm(coords[i] - proj)
        distances.append(distance)
    
    # Elbow is the point with maximum distance
    elbow_idx = np.argmax(distances)
    return elbow_idx


def plot_elbow_analysis(metrics_df, save_path=None):
    """Plot elbow curves and other metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Inertia (Elbow curve)
    ax = axes[0, 0]
    ax.plot(metrics_df['k'], metrics_df['inertia'], 'bo-')
    elbow_idx = find_elbow_point(metrics_df['inertia'])
    elbow_k = metrics_df['k'].iloc[elbow_idx]
    ax.axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow at k={elbow_k}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Silhouette Score (higher is better)
    ax = axes[0, 1]
    ax.plot(metrics_df['k'], metrics_df['silhouette'], 'go-')
    best_k = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']
    ax.axvline(x=best_k, color='r', linestyle='--', label=f'Best at k={best_k}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Analysis (Higher is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Davies-Bouldin Score (lower is better)
    ax = axes[1, 0]
    ax.plot(metrics_df['k'], metrics_df['davies_bouldin'], 'ro-')
    best_k = metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'k']
    ax.axvline(x=best_k, color='b', linestyle='--', label=f'Best at k={best_k}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Davies-Bouldin Score')
    ax.set_title('Davies-Bouldin Score (Lower is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calinski-Harabasz Score (higher is better)
    ax = axes[1, 1]
    ax.plot(metrics_df['k'], metrics_df['calinski_harabasz'], 'mo-')
    best_k = metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'k']
    ax.axvline(x=best_k, color='r', linestyle='--', label=f'Best at k={best_k}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Calinski-Harabasz Score')
    ax.set_title('Calinski-Harabasz Score (Higher is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    return fig


def get_optimal_k(metrics_df):
    """Determine optimal k based on multiple metrics."""
    # Get best k for each metric
    elbow_k = metrics_df['k'].iloc[find_elbow_point(metrics_df['inertia'])]
    silhouette_k = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']
    davies_k = metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'k']
    calinski_k = metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'k']
    
    print("\nOptimal k by different metrics:")
    print(f"  Elbow method: k={elbow_k}")
    print(f"  Silhouette score: k={silhouette_k}")
    print(f"  Davies-Bouldin score: k={davies_k}")
    print(f"  Calinski-Harabasz score: k={calinski_k}")
    
    # Consensus: use mode or median
    k_values = [elbow_k, silhouette_k, davies_k, calinski_k]
    optimal_k = int(np.median(k_values))
    
    print(f"\nConsensus optimal k: {optimal_k}")
    
    return optimal_k


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal number of clusters for audiometric data'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to pretrained VAE checkpoint')
    parser.add_argument('--config', type=str, default='config/model_config_large.yaml',
                       help='Model configuration file')
    parser.add_argument('--data-config', type=str, default='config/training_config_contrastive.yaml',
                       help='Data configuration file')
    parser.add_argument('--k-min', type=int, default=5,
                       help='Minimum number of clusters to test')
    parser.add_argument('--k-max', type=int, default=20,
                       help='Maximum number of clusters to test')
    parser.add_argument('--output-dir', type=str, default='cluster_analysis',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger = setup_logging(
        log_level='INFO',
        log_dir=output_dir,
        experiment_name='cluster_optimization'
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load configs
    with open(args.config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Create model and load checkpoint
    logger.info("Loading pretrained model...")
    model = create_model(model_config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    logger.info("Loading data...")
    data_module = create_data_module(data_config['dataset'])
    data_module.setup(data_config['dataset']['data_path'])
    train_loader = data_module.train_dataloader()
    
    # Extract latent features
    logger.info("Extracting latent features...")
    latent_features = extract_latent_features(model, train_loader, device)
    logger.info(f"Extracted features shape: {latent_features.shape}")
    
    # Compute metrics for different k values
    logger.info(f"Testing k values from {args.k_min} to {args.k_max}...")
    metrics_df = compute_elbow_metrics(latent_features, (args.k_min, args.k_max))
    
    # Save metrics
    metrics_df.to_csv(output_dir / 'clustering_metrics.csv', index=False)
    logger.info(f"Metrics saved to {output_dir / 'clustering_metrics.csv'}")
    
    # Plot analysis
    plot_path = output_dir / 'elbow_analysis.png'
    plot_elbow_analysis(metrics_df, save_path=plot_path)
    
    # Get optimal k
    optimal_k = get_optimal_k(metrics_df)
    
    # Save recommendation
    recommendation = {
        'optimal_k': int(optimal_k),
        'metrics_summary': metrics_df.to_dict('records'),
        'analysis': {
            'elbow_k': int(metrics_df['k'].iloc[find_elbow_point(metrics_df['inertia'])]),
            'best_silhouette_k': int(metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']),
            'best_davies_k': int(metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'k']),
            'best_calinski_k': int(metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'k'])
        }
    }
    
    with open(output_dir / 'optimal_k_recommendation.yaml', 'w') as f:
        yaml.dump(recommendation, f)
    
    logger.info(f"\nRecommended number of clusters: {optimal_k}")
    logger.info(f"Results saved to {output_dir}")
    
    return optimal_k


if __name__ == '__main__':
    main()