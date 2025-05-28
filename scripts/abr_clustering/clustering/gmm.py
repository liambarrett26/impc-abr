#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GMM Clustering Module for ABR Audiogram Analysis

This module implements Gaussian Mixture Model (GMM) clustering for dimensionality
reduction and pattern identification in Auditory Brainstem Response (ABR) audiograms.
It provides functionality to identify clusters of hearing loss patterns and analyze
their characteristics.

author: Liam Barrett
version: 1.0.0
"""

import warnings
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA

class AudiogramGMM:
    """
    Apply GMM clustering to audiogram data for pattern identification.
    
    This class handles the preprocessing, clustering, and visualization of audiogram
    data using Gaussian Mixture Models. It helps identify distinct patterns of
    hearing loss and can be used to classify new audiograms into these patterns.
    
    Attributes:
        n_clusters (int): Number of clusters to fit.
        gmm (GaussianMixture): The sklearn GaussianMixture object.
        scaler (StandardScaler): The sklearn StandardScaler object.
        pca (PCA): Optional PCA for dimensionality reduction before clustering.
        cluster_labels (numpy.ndarray): Cluster assignments for training data.
        cluster_centers (numpy.ndarray): Cluster centers in original space.
        frequency_labels (list): Labels for the frequency columns.
        fitted (bool): Whether the GMM has been fitted.
        use_pca (bool): Whether to use PCA preprocessing.
        pca_components (int): Number of PCA components if using PCA.
    """
    
    def __init__(self, n_clusters=5, use_pca=False, pca_components=3, 
                 covariance_type='full', random_state=42):
        """
        Initialize the AudiogramGMM with the specified parameters.
        
        Parameters:
            n_clusters (int, optional): Number of clusters to fit. Defaults to 5.
            use_pca (bool, optional): Whether to use PCA preprocessing. Defaults to False.
            pca_components (int, optional): Number of PCA components. Defaults to 3.
            covariance_type (str, optional): Type of covariance matrix. Defaults to 'full'.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        self.n_clusters = n_clusters
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components) if use_pca else None
        self.cluster_labels = None
        self.cluster_centers = None
        self.cluster_probabilities = None
        self.frequency_labels = None
        self.fitted = False
        
    def fit(self, audiograms, frequency_labels=None):
        """
        Fit the GMM model to the audiogram data.
        
        Parameters:
            audiograms (numpy.ndarray or pandas.DataFrame): 
                Audiogram data with shape (n_samples, n_frequencies).
            frequency_labels (list, optional): 
                Labels for the frequency columns, used for visualization.
                
        Returns:
            self: The fitted AudiogramGMM object.
        """
        # Convert to numpy array if DataFrame
        if isinstance(audiograms, pd.DataFrame):
            if frequency_labels is None:
                frequency_labels = audiograms.columns
            audiograms = audiograms.values
        
        # Store frequency labels
        self.frequency_labels = frequency_labels
        if self.frequency_labels is None:
            self.frequency_labels = [f"Freq_{i}" for i in range(audiograms.shape[1])]
            
        # Standardize the data
        audiograms_scaled = self.scaler.fit_transform(audiograms)
        
        # Apply PCA if requested
        if self.use_pca:
            audiograms_processed = self.pca.fit_transform(audiograms_scaled)
        else:
            audiograms_processed = audiograms_scaled
        
        # Fit GMM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gmm.fit(audiograms_processed)
        
        # Get cluster assignments and probabilities
        self.cluster_labels = self.gmm.predict(audiograms_processed)
        self.cluster_probabilities = self.gmm.predict_proba(audiograms_processed)
        
        # Calculate cluster centers in original space
        self._calculate_cluster_centers(audiograms)
        
        self.fitted = True
        return self
    
    def predict(self, audiograms):
        """
        Predict cluster assignments for new audiograms.
        
        Parameters:
            audiograms (numpy.ndarray or pandas.DataFrame): 
                Audiogram data with shape (n_samples, n_frequencies).
                
        Returns:
            tuple: (cluster_labels, cluster_probabilities)
            
        Raises:
            ValueError: If the GMM model has not been fitted.
        """
        if not self.fitted:
            raise ValueError("GMM model needs to be fitted before prediction.")
        
        # Convert to numpy array if DataFrame
        if isinstance(audiograms, pd.DataFrame):
            audiograms = audiograms.values
            
        # Preprocess data
        audiograms_scaled = self.scaler.transform(audiograms)
        
        if self.use_pca:
            audiograms_processed = self.pca.transform(audiograms_scaled)
        else:
            audiograms_processed = audiograms_scaled
            
        # Predict clusters
        labels = self.gmm.predict(audiograms_processed)
        probabilities = self.gmm.predict_proba(audiograms_processed)
        
        return labels, probabilities
    
    def _calculate_cluster_centers(self, original_audiograms):
        """
        Calculate cluster centers in the original audiogram space.
        
        Parameters:
            original_audiograms (numpy.ndarray): Original audiogram data.
        """
        self.cluster_centers = []
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                cluster_center = np.mean(original_audiograms[cluster_mask], axis=0)
            else:
                # If no points assigned to cluster, use overall mean
                cluster_center = np.mean(original_audiograms, axis=0)
            self.cluster_centers.append(cluster_center)
            
        self.cluster_centers = np.array(self.cluster_centers)
    
    def calculate_metrics(self, audiograms):
        """
        Calculate clustering quality metrics.
        
        Parameters:
            audiograms (numpy.ndarray or pandas.DataFrame): 
                Audiogram data used for fitting.
                
        Returns:
            dict: Dictionary containing various clustering metrics.
        """
        if not self.fitted:
            raise ValueError("GMM model needs to be fitted before calculating metrics.")
            
        # Convert to numpy array if DataFrame
        if isinstance(audiograms, pd.DataFrame):
            audiograms = audiograms.values
            
        # Preprocess data
        audiograms_scaled = self.scaler.transform(audiograms)
        
        if self.use_pca:
            audiograms_processed = self.pca.transform(audiograms_scaled)
        else:
            audiograms_processed = audiograms_scaled
        
        metrics = {}
        
        # Silhouette score
        if len(set(self.cluster_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(
                audiograms_processed, self.cluster_labels
            )
        else:
            metrics['silhouette_score'] = np.nan
            
        # Calinski-Harabasz score
        if len(set(self.cluster_labels)) > 1:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                audiograms_processed, self.cluster_labels
            )
        else:
            metrics['calinski_harabasz_score'] = np.nan
        
        # AIC and BIC
        metrics['aic'] = self.gmm.aic(audiograms_processed)
        metrics['bic'] = self.gmm.bic(audiograms_processed)
        
        # Log likelihood
        metrics['log_likelihood'] = self.gmm.score(audiograms_processed)
        
        return metrics
    
    def plot_cluster_audiograms(self, save_path=None, show_confidence=True):
        """
        Plot the average audiogram for each cluster.
        
        Parameters:
            save_path (str or Path, optional): 
                Path to save the plot. If None, the plot is displayed.
            show_confidence (bool, optional): 
                Whether to show confidence intervals. Defaults to True.
                
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if not self.fitted:
            raise ValueError("GMM model needs to be fitted before plotting.")
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color palette for clusters
        colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, self.n_clusters))
        
        for cluster_id in range(self.n_clusters):
            cluster_center = self.cluster_centers[cluster_id]
            cluster_size = np.sum(self.cluster_labels == cluster_id)
            
            # Plot cluster center
            ax.plot(self.frequency_labels, cluster_center, 
                    color=colors[cluster_id], marker='o', linewidth=2,
                    label=f'Cluster {cluster_id} (n={cluster_size})')
            
            # Add confidence intervals if requested
            if show_confidence:
                cluster_mask = self.cluster_labels == cluster_id
                if np.sum(cluster_mask) > 1:
                    # Calculate standard error
                    cluster_data = self.cluster_centers[cluster_mask] if hasattr(self, '_original_data') else None
                    if cluster_data is not None:
                        cluster_std = np.std(cluster_data, axis=0)
                        cluster_se = cluster_std / np.sqrt(cluster_size)
                        
                        ax.fill_between(
                            range(len(self.frequency_labels)),
                            cluster_center - 1.96 * cluster_se,
                            cluster_center + 1.96 * cluster_se,
                            color=colors[cluster_id], alpha=0.2
                        )
        
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Threshold (dB SPL)')
        ax.set_title('Cluster Average Audiograms')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        ax.set_xticks(range(len(self.frequency_labels)))
        ax.set_xticklabels(self.frequency_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_cluster_scatter(self, audiograms, save_path=None):
        """
        Plot a 2D scatter plot of clusters using PCA projection.
        
        Parameters:
            audiograms (numpy.ndarray or pandas.DataFrame): 
                Original audiogram data.
            save_path (str or Path, optional): 
                Path to save the plot. If None, the plot is displayed.
                
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        if not self.fitted:
            raise ValueError("GMM model needs to be fitted before plotting.")
            
        # Convert to numpy array if DataFrame
        if isinstance(audiograms, pd.DataFrame):
            audiograms = audiograms.values
            
        # Always use PCA for 2D visualization
        audiograms_scaled = self.scaler.transform(audiograms)
        pca_2d = PCA(n_components=2)
        audiograms_2d = pca_2d.fit_transform(audiograms_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color palette for clusters
        colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, self.n_clusters))
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size > 0:
                ax.scatter(
                    audiograms_2d[cluster_mask, 0],
                    audiograms_2d[cluster_mask, 1],
                    c=[colors[cluster_id]], 
                    alpha=0.6,
                    s=50,
                    label=f'Cluster {cluster_id} (n={cluster_size})'
                )
        
        ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('GMM Clusters in PCA Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def analyze_cluster_patterns(self):
        """
        Analyze and classify the patterns of each cluster.
        
        Returns:
            pandas.DataFrame: DataFrame with cluster analysis results.
        """
        if not self.fitted:
            raise ValueError("GMM model needs to be fitted before analysis.")
            
        results = []
        
        for cluster_id in range(self.n_clusters):
            cluster_center = self.cluster_centers[cluster_id]
            cluster_size = np.sum(self.cluster_labels == cluster_id)
            
            # Analyze pattern characteristics
            pattern_analysis = self._classify_audiogram_pattern(cluster_center)
            
            results.append({
                'cluster_id': cluster_id,
                'cluster_size': cluster_size,
                'cluster_percentage': cluster_size / len(self.cluster_labels) * 100,
                'pattern_type': pattern_analysis['pattern_type'],
                'severity': pattern_analysis['severity'],
                'mean_threshold': np.mean(cluster_center),
                'threshold_range': np.max(cluster_center) - np.min(cluster_center),
                'high_freq_bias': np.mean(cluster_center[-2:]) - np.mean(cluster_center[:2]),
                'center_thresholds': cluster_center.tolist()
            })
            
        return pd.DataFrame(results)
    
    def _classify_audiogram_pattern(self, audiogram):
        """
        Classify the pattern of an audiogram.
        
        Parameters:
            audiogram (numpy.ndarray): Audiogram thresholds.
            
        Returns:
            dict: Pattern classification results.
        """
        # Calculate pattern characteristics
        mean_threshold = np.mean(audiogram)
        low_freq_mean = np.mean(audiogram[:2])  # 6, 12 kHz
        high_freq_mean = np.mean(audiogram[-2:])  # 24, 30 kHz
        
        # Determine severity
        if mean_threshold < 30:
            severity = 'Normal'
        elif mean_threshold < 45:
            severity = 'Mild'
        elif mean_threshold < 70:
            severity = 'Moderate'
        else:
            severity = 'Severe'
        
        # Determine pattern type
        freq_diff = high_freq_mean - low_freq_mean
        
        if abs(freq_diff) < 15:
            pattern_type = 'Flat'
        elif freq_diff > 15:
            pattern_type = 'High-frequency'
        else:
            pattern_type = 'Low-frequency'
            
        return {
            'pattern_type': pattern_type,
            'severity': severity,
            'frequency_bias': freq_diff
        }
    
    def get_gene_cluster_mapping(self, gene_labels):
        """
        Create a mapping of genes to their cluster assignments.
        
        Parameters:
            gene_labels (array-like): Gene labels corresponding to audiograms.
            
        Returns:
            pandas.DataFrame: DataFrame with gene-cluster mappings.
        """
        if not self.fitted:
            raise ValueError("GMM model needs to be fitted before creating mapping.")
            
        if len(gene_labels) != len(self.cluster_labels):
            raise ValueError("Gene labels must have same length as audiogram data.")
            
        # Create DataFrame with gene-cluster mappings
        mapping_df = pd.DataFrame({
            'gene_symbol': gene_labels,
            'cluster_id': self.cluster_labels,
            'max_probability': np.max(self.cluster_probabilities, axis=1)
        })
        
        # Add cluster probabilities for each cluster
        for i in range(self.n_clusters):
            mapping_df[f'cluster_{i}_prob'] = self.cluster_probabilities[:, i]
        
        return mapping_df
    
    def save_model(self, file_path):
        """
        Save the GMM model to a file.
        
        Parameters:
            file_path (str or Path): Path to save the model.
        """
        if not self.fitted:
            raise ValueError("GMM model needs to be fitted before saving.")
            
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'n_clusters': self.n_clusters,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components,
            'gmm': self.gmm,
            'scaler': self.scaler,
            'pca': self.pca,
            'cluster_labels': self.cluster_labels,
            'cluster_centers': self.cluster_centers,
            'cluster_probabilities': self.cluster_probabilities,
            'frequency_labels': self.frequency_labels,
            'fitted': self.fitted
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, file_path):
        """
        Load a saved GMM model from a file.
        
        Parameters:
            file_path (str or Path): Path to the saved model.
            
        Returns:
            AudiogramGMM: The loaded model.
        """
        file_path = Path(file_path)
        
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        model = cls(
            n_clusters=model_data['n_clusters'],
            use_pca=model_data['use_pca'],
            pca_components=model_data['pca_components']
        )
        
        # Restore model state
        model.gmm = model_data['gmm']
        model.scaler = model_data['scaler']
        model.pca = model_data['pca']
        model.cluster_labels = model_data['cluster_labels']
        model.cluster_centers = model_data['cluster_centers']
        model.cluster_probabilities = model_data['cluster_probabilities']
        model.frequency_labels = model_data['frequency_labels']
        model.fitted = model_data['fitted']
        
        return model


def select_optimal_clusters(audiograms, max_clusters=10, metric='bic'):
    """
    Select optimal number of clusters using model selection criteria.
    
    Parameters:
        audiograms (numpy.ndarray or pandas.DataFrame): Audiogram data.
        max_clusters (int, optional): Maximum number of clusters to test. Defaults to 10.
        metric (str, optional): Metric to use ('bic', 'aic', 'silhouette'). Defaults to 'bic'.
        
    Returns:
        tuple: (optimal_n_clusters, scores_df)
    """
    if isinstance(audiograms, pd.DataFrame):
        audiograms = audiograms.values
    
    scores = []
    cluster_range = range(2, max_clusters + 1)
    
    for n_clusters in cluster_range:
        gmm = AudiogramGMM(n_clusters=n_clusters)
        gmm.fit(audiograms)
        
        metrics = gmm.calculate_metrics(audiograms)
        metrics['n_clusters'] = n_clusters
        scores.append(metrics)
    
    scores_df = pd.DataFrame(scores)
    
    # Select optimal based on metric
    if metric == 'bic':
        optimal_idx = scores_df['bic'].idxmin()
    elif metric == 'aic':
        optimal_idx = scores_df['aic'].idxmin()
    elif metric == 'silhouette':
        optimal_idx = scores_df['silhouette_score'].idxmax()
    else:
        raise ValueError("Metric must be 'bic', 'aic', or 'silhouette'")
    
    optimal_n_clusters = scores_df.loc[optimal_idx, 'n_clusters']
    
    return optimal_n_clusters, scores_df
