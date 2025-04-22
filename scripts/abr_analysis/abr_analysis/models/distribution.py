# abr_analysis/models/distribution.py
from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EmpiricalCovariance

class BaseDistribution(ABC):
    """Abstract base class for ABR profile distributions."""
    
    @abstractmethod
    def fit(self, profiles):
        """Fit distribution to control profiles."""
        pass
    
    @abstractmethod
    def score(self, profile):
        """Calculate log probability of profile."""
        pass

class RobustMultivariateGaussian(BaseDistribution):
    """Robust Multivariate Gaussian distribution model with error handling."""
    
    def __init__(self, min_cov_value=1e-8):
        self.mean = None
        self.cov = None
        self.min_cov_value = min_cov_value
        self.scaler = None
        self._cov_estimator = EmpiricalCovariance(assume_centered=False)
    
    def _validate_and_clean_data(self, profiles):
        """Validate and clean input data."""
        if len(profiles.shape) != 2:
            raise ValueError("Profiles must be 2D array")
        
        # Remove any rows with NaN or inf values
        clean_profiles = profiles[~np.any(np.isnan(profiles) | np.isinf(profiles), axis=1)]
        
        if len(clean_profiles) == 0:
            raise ValueError("No valid profiles after cleaning")
            
        return clean_profiles
    
    def _preprocess_data(self, profiles):
        """Scale the data to help with numerical stability."""
        # Simple standardization
        self.scaler = {
            'mean': np.mean(profiles, axis=0),
            'std': np.std(profiles, axis=0)
        }
        self.scaler['std'][self.scaler['std'] == 0] = 1  # Avoid division by zero
        
        scaled_profiles = (profiles - self.scaler['mean']) / self.scaler['std']
        return scaled_profiles
    
    def fit(self, profiles):
        """Fit MVN to control profiles with robust covariance estimation."""
        clean_profiles = self._validate_and_clean_data(profiles)
        scaled_profiles = self._preprocess_data(clean_profiles)
        
        # Calculate mean
        self.mean = np.mean(scaled_profiles, axis=0)
        
        # Use robust covariance estimation
        try:
            self._cov_estimator.fit(scaled_profiles)
            self.cov = self._cov_estimator.covariance_
            
            # Ensure matrix is positive definite
            min_eig = np.min(np.real(np.linalg.eigvals(self.cov)))
            if min_eig < self.min_cov_value:
                self.cov += (self.min_cov_value - min_eig) * np.eye(self.cov.shape[0])
                
        except Exception as e:
            print(f"Warning: Error in covariance estimation: {e}")
            print("Falling back to diagonal covariance matrix")
            self.cov = np.diag(np.var(scaled_profiles, axis=0))
    
    def score(self, profile):
        """Calculate log probability of profile."""
        if np.any(np.isnan(profile)) or np.any(np.isinf(profile)):
            return -np.inf
        
        # Scale the profile
        scaled_profile = (profile - self.scaler['mean']) / self.scaler['std']
        
        try:
            log_prob = stats.multivariate_normal.logpdf(
                scaled_profile,
                mean=self.mean,
                cov=self.cov,
                allow_singular=True
            )
            return log_prob
        except Exception as e:
            print(f"Warning: Error calculating log probability: {e}")
            return -np.inf
    
    def mahalanobis(self, profile):
        """Calculate Mahalanobis distance for a profile."""
        if np.any(np.isnan(profile)) or np.any(np.isinf(profile)):
            return np.inf
        
        # Scale the profile
        scaled_profile = (profile - self.scaler['mean']) / self.scaler['std']
        
        try:
            # Calculate Mahalanobis distance
            diff = scaled_profile - self.mean
            inv_cov = np.linalg.pinv(self.cov)  # Use pseudoinverse for stability
            dist = np.sqrt(diff @ inv_cov @ diff)
            return dist
        except Exception as e:
            print(f"Warning: Error calculating Mahalanobis distance: {e}")
            return np.inf