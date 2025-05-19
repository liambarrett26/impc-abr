#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PCA Module for ABR Audiogram Analysis

This module implements Principal Component Analysis (PCA) for dimensionality
reduction of Auditory Brainstem Response (ABR) audiograms. It provides
functionality to identify major patterns of variation in audiogram shapes
and visualize these patterns.

author: Liam Barrett
version: 1.0.0
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class AudiogramPCA:
    """
    Apply PCA to audiogram data for dimensionality reduction and pattern analysis.
    
    This class handles the preprocessing, analysis, and visualization of audiogram
    data using Principal Component Analysis (PCA). It helps identify major modes of
    variation in audiogram shapes and can be used to project new audiograms onto
    these principal components.
    
    Attributes:
        n_components (int): Number of principal components to compute.
        pca (PCA): The sklearn PCA object.
        scaler (StandardScaler): The sklearn StandardScaler object.
        components (numpy.ndarray): The principal components.
        explained_variance_ratio (numpy.ndarray): The variance explained by each component.
        frequency_labels (list): Labels for the frequency columns.
        feature_names (list): Names of the features used in the PCA.
        fitted (bool): Whether the PCA has been fitted.
    """
    
    def __init__(self, n_components=2):
        """
        Initialize the AudiogramPCA with the specified number of components.
        
        Parameters:
            n_components (int, optional): Number of principal components to compute.
                Defaults to 2.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.components = None
        self.explained_variance_ratio = None
        self.frequency_labels = None
        self.feature_names = None
        self.fitted = False
        
    def fit(self, audiograms, frequency_labels=None):
        """
        Fit the PCA model to the audiogram data.
        
        Parameters:
            audiograms (numpy.ndarray or pandas.DataFrame): 
                Audiogram data with shape (n_samples, n_frequencies).
            frequency_labels (list, optional): 
                Labels for the frequency columns, used for visualization.
                
        Returns:
            self: The fitted AudiogramPCA object.
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
        
        self.feature_names = self.frequency_labels
            
        # Standardize the data
        audiograms_scaled = self.scaler.fit_transform(audiograms)
        
        # Fit PCA
        self.pca.fit(audiograms_scaled)
        
        # Store components and explained variance
        self.components = self.pca.components_
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        
        self.fitted = True
        return self
    
    def transform(self, audiograms):
        """
        Transform audiograms into the PCA space.
        
        Parameters:
            audiograms (numpy.ndarray or pandas.DataFrame): 
                Audiogram data with shape (n_samples, n_frequencies).
                
        Returns:
            numpy.ndarray: Transformed audiograms in PCA space.
            
        Raises:
            ValueError: If the PCA model has not been fitted.
        """
        if not self.fitted:
            raise ValueError("PCA model needs to be fitted before transform.")
        
        # Convert to numpy array if DataFrame
        if isinstance(audiograms, pd.DataFrame):
            audiograms = audiograms.values
            
        # Standardize and transform
        audiograms_scaled = self.scaler.transform(audiograms)
        return self.pca.transform(audiograms_scaled)
    
    def fit_transform(self, audiograms, frequency_labels=None):
        """
        Fit the PCA model and transform audiograms in one step.
        
        Parameters:
            audiograms (numpy.ndarray or pandas.DataFrame): 
                Audiogram data with shape (n_samples, n_frequencies).
            frequency_labels (list, optional): 
                Labels for the frequency columns.
                
        Returns:
            numpy.ndarray: Transformed audiograms in PCA space.
        """
        self.fit(audiograms, frequency_labels)
        return self.transform(audiograms)
    
    def inverse_transform(self, pca_coordinates):
        """
        Transform PCA coordinates back to original audiogram space.
        
        Parameters:
            pca_coordinates (numpy.ndarray): 
                Coordinates in PCA space with shape (n_samples, n_components).
                
        Returns:
            numpy.ndarray: Audiograms in original space.
            
        Raises:
            ValueError: If the PCA model has not been fitted.
        """
        if not self.fitted:
            raise ValueError("PCA model needs to be fitted before inverse_transform.")
            
        # Inverse PCA transform then inverse scaling
        audiograms_scaled = self.pca.inverse_transform(pca_coordinates)
        return self.scaler.inverse_transform(audiograms_scaled)
    
    def plot_explained_variance(self, save_path=None):
        """
        Plot the explained variance ratio for each principal component.
        
        Parameters:
            save_path (str or Path, optional): 
                Path to save the plot. If None, the plot is displayed.
                
        Returns:
            matplotlib.figure.Figure: The created figure.
            
        Raises:
            ValueError: If the PCA model has not been fitted.
        """
        if not self.fitted:
            raise ValueError("PCA model needs to be fitted before plotting.")
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar plot of explained variance
        ax.bar(range(1, self.n_components + 1), 
               self.explained_variance_ratio, 
               alpha=0.8, 
               color='skyblue')
        
        # Line plot of cumulative explained variance
        cum_var = np.cumsum(self.explained_variance_ratio)
        ax2 = ax.twinx()
        ax2.plot(range(1, self.n_components + 1), 
                 cum_var, 
                 'r-', 
                 marker='o', 
                 markersize=8, 
                 linewidth=2)
        
        # Formatting
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.grid(False)
        ax.set_xticks(range(1, self.n_components + 1))
        ax.set_title('Explained Variance by Principal Component')
        
        # Add value annotations
        for i, v in enumerate(self.explained_variance_ratio):
            ax.text(i + 1, v + 0.01, f'{v:.2f}', ha='center')
            
        for i, v in enumerate(cum_var):
            ax2.text(i + 1, v + 0.01, f'{v:.2f}', ha='center', color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def plot_components(self, save_path=None):
        """
        Plot the loadings (weight vectors) for each principal component.
        
        Parameters:
            save_path (str or Path, optional): 
                Path to save the plot. If None, the plot is displayed.
                
        Returns:
            matplotlib.figure.Figure: The created figure.
            
        Raises:
            ValueError: If the PCA model has not been fitted.
        """
        if not self.fitted:
            raise ValueError("PCA model needs to be fitted before plotting.")
            
        fig, axes = plt.subplots(nrows=self.n_components, figsize=(10, 3*self.n_components))
        
        # Ensure axes is iterable even with a single component
        if self.n_components == 1:
            axes = [axes]
            
        # Plot each principal component
        for i, (ax, component) in enumerate(zip(axes, self.components)):
            var_explained = self.explained_variance_ratio[i]
            
            # Bar plot of component loadings
            ax.bar(self.frequency_labels, component, color='skyblue', alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            
            # Add titles and labels
            ax.set_title(f'PC{i+1} ({var_explained:.2%} explained variance)')
            ax.set_ylabel('Loading')
            
            # Rotate x labels for better readability
            ax.set_xticklabels(self.frequency_labels, rotation=45, ha='right')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def plot_component_audiograms(self, n_std=2, save_path=None):
        """
        Plot the effect of each principal component on an audiogram.
        
        Creates a visualization of how each principal component affects the 
        audiogram shape when varied from -n_std to +n_std standard deviations.
        
        Parameters:
            n_std (float, optional): Number of standard deviations to vary the component.
                Defaults to 2.
            save_path (str or Path, optional): 
                Path to save the plot. If None, the plot is displayed.
                
        Returns:
            matplotlib.figure.Figure: The created figure.
            
        Raises:
            ValueError: If the PCA model has not been fitted.
        """
        if not self.fitted:
            raise ValueError("PCA model needs to be fitted before plotting.")
            
        fig, axes = plt.subplots(nrows=self.n_components, figsize=(10, 3*self.n_components))
        
        # Ensure axes is iterable even with a single component
        if self.n_components == 1:
            axes = [axes]
            
        # Calculate standard deviation of each PC
        std_devs = np.sqrt(self.pca.explained_variance_)
        
        # Get the mean audiogram from the scaler
        mean_audiogram = self.scaler.mean_
        
        # Plot each principal component's effect
        for i, (ax, component) in enumerate(zip(axes, self.components)):
            var_explained = self.explained_variance_ratio[i]
            std = std_devs[i]
            
            # Create a zero vector in PCA space
            pca_coordinates_zero = np.zeros(self.n_components)
            
            # Generate points along PCA component i
            variations = np.linspace(-n_std*std, n_std*std, 5)
            
            # Plot each variation
            for variation in variations:
                # Set the position along component i
                pca_coordinates = pca_coordinates_zero.copy()
                pca_coordinates[i] = variation
                
                # Transform back to audiogram space (single sample)
                audiogram = self.inverse_transform(pca_coordinates.reshape(1, -1))[0]
                
                # Plot with increasing intensity as variation increases
                intensity = abs(variation) / (n_std*std)
                color = 'red' if variation > 0 else 'blue'
                alpha = 0.3 + 0.7 * intensity
                
                ax.plot(self.frequency_labels, audiogram, 
                        color=color, alpha=alpha, marker='o', 
                        label=f'{variation:.1f} std' if round(variation, 1) != 0 else 'Mean')
            
            # Plot mean audiogram
            ax.plot(self.frequency_labels, mean_audiogram, 
                    color='black', linewidth=2, marker='o', label='Mean')
            
            # Add titles and labels
            ax.set_title(f'PC{i+1} Effect ({var_explained:.2%} explained variance)')
            ax.set_ylabel('Threshold (dB SPL)')
            
            # Rotate x labels for better readability
            ax.set_xticklabels(self.frequency_labels, rotation=45, ha='right')
            
            # Add legend
            ax.legend(loc='best')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def plot_extreme_audiograms(self, audiograms, pca_coords, component=1, n_examples=5, save_path=None):
        """
        Plot audiograms with extreme values along a specific principal component.
        
        Parameters:
            audiograms (numpy.ndarray or pandas.DataFrame): 
                Original audiogram data.
            pca_coords (numpy.ndarray): 
                Coordinates in PCA space.
            component (int, optional): 
                Principal component to analyze (1-indexed). Defaults to 1.
            n_examples (int, optional): 
                Number of examples to plot for each extreme. Defaults to 5.
            save_path (str or Path, optional): 
                Path to save the plot. If None, the plot is displayed.
                
        Returns:
            matplotlib.figure.Figure: The created figure.
            
        Raises:
            ValueError: If the PCA model has not been fitted or if component is invalid.
        """
        if not self.fitted:
            raise ValueError("PCA model needs to be fitted before plotting.")
            
        # Check component is valid
        if component < 1 or component > self.n_components:
            raise ValueError(f"Component must be between 1 and {self.n_components}")
            
        # Convert to numpy array if DataFrame
        if isinstance(audiograms, pd.DataFrame):
            audiograms_array = audiograms.values
        else:
            audiograms_array = audiograms
            
        # Find extreme examples along this component
        pc_idx = component - 1  # Convert to 0-indexed
        sorted_indices = np.argsort(pca_coords[:, pc_idx])
        low_indices = sorted_indices[:n_examples]
        high_indices = sorted_indices[-n_examples:]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot low extreme audiograms
        for i, idx in enumerate(low_indices):
            ax.plot(self.frequency_labels, audiograms_array[idx], 'b-', alpha=0.7, 
                    marker='o', label=f'Low PC{component} #{i+1}')
            
        # Plot high extreme audiograms
        for i, idx in enumerate(high_indices):
            ax.plot(self.frequency_labels, audiograms_array[idx], 'r-', alpha=0.7, 
                    marker='o', label=f'High PC{component} #{i+1}')
            
        # Calculate and plot mean audiogram
        mean_audiogram = np.mean(audiograms_array, axis=0)
        ax.plot(self.frequency_labels, mean_audiogram, 'k-', linewidth=2, 
                marker='s', label='Mean Audiogram')
            
        # Add titles and labels
        var_explained = self.explained_variance_ratio[pc_idx]
        ax.set_title(f'Extreme Audiograms along PC{component} ({var_explained:.2%} explained variance)')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Threshold (dB SPL)')
        
        # Rotate x labels for better readability
        ax.set_xticklabels(self.frequency_labels, rotation=45, ha='right')
        
        # Add legend (in two columns for readability)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        
        # Invert y-axis (lower thresholds = better hearing)
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def save_model(self, file_path):
        """
        Save the PCA model to a file.
        
        Parameters:
            file_path (str or Path): Path to save the model.
            
        Raises:
            ValueError: If the PCA model has not been fitted.
        """
        if not self.fitted:
            raise ValueError("PCA model needs to be fitted before saving.")
            
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a dictionary with all the model data
        model_data = {
            'n_components': self.n_components,
            'pca': self.pca,
            'scaler': self.scaler,
            'components': self.components,
            'explained_variance_ratio': self.explained_variance_ratio,
            'frequency_labels': self.frequency_labels,
            'feature_names': self.feature_names,
            'fitted': self.fitted
        }
        
        # Save to file
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, file_path):
        """
        Load a saved PCA model from a file.
        
        Parameters:
            file_path (str or Path): Path to the saved model.
            
        Returns:
            AudiogramPCA: The loaded model.
        """
        file_path = Path(file_path)
        
        # Load from file
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        model = cls(n_components=model_data['n_components'])
        
        # Restore model state
        model.pca = model_data['pca']
        model.scaler = model_data['scaler']
        model.components = model_data['components']
        model.explained_variance_ratio = model_data['explained_variance_ratio']
        model.frequency_labels = model_data['frequency_labels']
        model.feature_names = model_data['feature_names']
        model.fitted = model_data['fitted']
        
        return model