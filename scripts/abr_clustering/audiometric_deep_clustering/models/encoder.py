"""
Frequency-aware encoder for ContrastiveVAE-DEC model.

This module implements the encoder component that transforms 18-dimensional
audiometric features into a lower-dimensional latent representation, with
specialized attention mechanisms for ABR frequency relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ABRFeatureEncoder(nn.Module):
    """
    Specialized encoder for ABR threshold features with frequency-aware processing.

    Handles the first 6 dimensions (ABR thresholds) with attention mechanisms
    that can learn frequency relationships.
    """

    def __init__(self, config: Dict):
        """
        Initialize ABR feature encoder.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config

        # ABR-specific parameters
        self.abr_dim = config['data']['abr_features']  # 6
        self.hidden_dim = config['encoder']['abr_encoder']['hidden_dim']  # 16
        self.use_batch_norm = config['encoder']['abr_encoder']['use_batch_norm']

        # ABR feature processing layers
        self.abr_projection = nn.Linear(self.abr_dim, self.hidden_dim)

        if self.use_batch_norm:
            self.abr_norm = nn.BatchNorm1d(self.hidden_dim)

        self.abr_activation = nn.ReLU()
        self.abr_dropout = nn.Dropout(config['encoder']['dropout_rate'])

        # Optional frequency attention
        self.use_attention = config['encoder']['attention']['enabled']
        if self.use_attention:
            self.attention = FrequencyAttention(
                input_dim=self.abr_dim,
                head_dim=config['encoder']['attention']['head_dim'],
                num_heads=config['encoder']['attention']['num_heads'],
                dropout=config['encoder']['attention']['dropout']
            )

    def forward(self, abr_features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for ABR features.

        Args:
            abr_features: ABR threshold features (batch_size, 6)

        Returns:
            Tuple of (encoded_features, attention_weights)
        """
        # Apply attention if enabled
        attention_weights = None
        if self.use_attention:
            abr_features, attention_weights = self.attention(abr_features)

        # Project ABR features
        x = self.abr_projection(abr_features)

        # Apply normalization if enabled
        if self.use_batch_norm:
            x = self.abr_norm(x)

        # Apply activation and dropout
        x = self.abr_activation(x)
        x = self.abr_dropout(x)

        return x, attention_weights


class MetadataEncoder(nn.Module):
    """
    Encoder for metadata features (continuous and categorical embeddings).

    Handles the remaining 12 dimensions (10 metadata + 2 PCA) with appropriate
    processing for mixed data types.
    """

    def __init__(self, config: Dict):
        """
        Initialize metadata encoder.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config

        # Metadata parameters
        self.metadata_dim = config['data']['metadata_features'] + config['data']['pca_features']  # 12
        self.hidden_dim = config['encoder']['metadata_encoder']['hidden_dim']  # 24

        # Metadata processing layers
        self.metadata_projection = nn.Linear(self.metadata_dim, self.hidden_dim)
        self.metadata_activation = nn.ReLU()
        self.metadata_dropout = nn.Dropout(config['encoder']['dropout_rate'])

    def forward(self, metadata_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for metadata features.

        Args:
            metadata_features: Metadata and PCA features (batch_size, 12)

        Returns:
            Encoded metadata features
        """
        x = self.metadata_projection(metadata_features)
        x = self.metadata_activation(x)
        x = self.metadata_dropout(x)

        return x


class FrequencyAttention(nn.Module):
    """
    Multi-head attention mechanism specialized for ABR frequency relationships.

    Learns to weight the importance of different frequency measurements
    and their interactions for hearing phenotype characterization.
    """

    def __init__(self, input_dim: int, head_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize frequency attention.

        Args:
            input_dim: Input feature dimension (6 for ABR)
            head_dim: Dimension per attention head
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        # Attention projections (working with single values per frequency)
        self.q_projection = nn.Linear(1, head_dim * num_heads)
        self.k_projection = nn.Linear(1, head_dim * num_heads)
        self.v_projection = nn.Linear(1, head_dim * num_heads)

        # Output projection (back to single value per frequency)
        self.output_projection = nn.Linear(head_dim * num_heads, 1)

        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply frequency attention.

        Args:
            x: Input features (batch_size, seq_len) where seq_len = 6 frequencies

        Returns:
            Tuple of (attended_features, attention_weights)
        """
        batch_size, seq_len = x.shape

        # Reshape input for attention: (batch_size, seq_len, 1) -> treat each frequency as a token
        x = x.unsqueeze(-1)  # (batch_size, 6, 1)

        # Project to queries, keys, values
        q = self.q_projection(x)  # (batch_size, 6, head_dim * num_heads)
        k = self.k_projection(x)
        v = self.v_projection(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.head_dim * self.num_heads
        )

        # Output projection and dropout
        output = self.output_projection(attended)
        output = self.output_dropout(output)

        # Squeeze back to original shape: (batch_size, seq_len)
        output = output.squeeze(-1)

        # Return mean attention weights across heads for interpretability
        mean_attention_weights = attention_weights.mean(dim=1)  # (batch_size, seq_len, seq_len)

        return output, mean_attention_weights


class ContrastiveVAEEncoder(nn.Module):
    """
    Complete encoder for ContrastiveVAE-DEC model.

    Combines ABR and metadata encoders, processes through main encoder layers,
    and outputs latent representations for both VAE and clustering components.
    """

    def __init__(self, config: Dict):
        """
        Initialize the complete encoder.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config

        # Initialize sub-encoders
        self.abr_encoder = ABRFeatureEncoder(config)
        self.metadata_encoder = MetadataEncoder(config)

        # Calculate combined feature dimension
        abr_hidden = config['encoder']['abr_encoder']['hidden_dim']  # 16
        metadata_hidden = config['encoder']['metadata_encoder']['hidden_dim']  # 24
        self.combined_dim = abr_hidden + metadata_hidden  # 40

        # Main encoder layers
        self.hidden_dims = config['encoder']['hidden_dims']  # [64, 32, 16]
        self.latent_dim = config['latent']['latent_dim']  # 10
        self.dropout_rate = config['encoder']['dropout_rate']

        # Build main encoder network
        self.encoder_layers = self._build_encoder_layers()

        # VAE heads (mean and log variance)
        self.mu_head = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.logvar_head = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(f"Initialized ContrastiveVAE encoder: "
                   f"input_dim={config['data']['total_features']}, "
                   f"latent_dim={self.latent_dim}, "
                   f"hidden_dims={self.hidden_dims}")

    def _build_encoder_layers(self) -> nn.ModuleList:
        """Build the main encoder network layers."""
        layers = nn.ModuleList()

        input_dim = self.combined_dim

        for hidden_dim in self.hidden_dims:
            layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if self.config['encoder'].get('use_batch_norm', True) else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            )
            layers.append(layer)
            input_dim = hidden_dim

        return layers

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            if self.config['architecture']['weight_init'] == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif self.config['architecture']['weight_init'] == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the encoder.

        Args:
            x: Input features (batch_size, 18)

        Returns:
            Dictionary containing:
                - latent_mu: Latent space mean (batch_size, latent_dim)
                - latent_logvar: Latent space log variance (batch_size, latent_dim)
                - latent_z: Sampled latent vector (batch_size, latent_dim)
                - attention_weights: Frequency attention weights (optional)
                - encoder_features: Intermediate encoder features
        """
        batch_size = x.shape[0]

        # Split input features
        abr_features = x[:, :6]  # First 6 dimensions
        metadata_features = x[:, 6:]  # Remaining 12 dimensions

        # Encode ABR features
        abr_encoded, attention_weights = self.abr_encoder(abr_features)

        # Encode metadata features
        metadata_encoded = self.metadata_encoder(metadata_features)

        # Combine encoded features
        combined_features = torch.cat([abr_encoded, metadata_encoded], dim=1)

        # Pass through main encoder layers
        encoder_features = combined_features
        for layer in self.encoder_layers:
            encoder_features = layer(encoder_features)

        # Generate latent parameters
        mu = self.mu_head(encoder_features)
        logvar = self.logvar_head(encoder_features)

        # Clamp log variance for numerical stability
        logvar = torch.clamp(
            logvar,
            min=self.config['latent']['min_logvar'],
            max=self.config['latent']['max_logvar']
        )

        # Sample latent vector using reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Prepare output dictionary
        output = {
            'latent_mu': mu,
            'latent_logvar': logvar,
            'latent_z': z,
            'encoder_features': encoder_features,
            'combined_features': combined_features,
            'abr_encoded': abr_encoded,
            'metadata_encoded': metadata_encoded
        }

        if attention_weights is not None:
            output['attention_weights'] = attention_weights

        return output

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE sampling.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space (deterministic).

        Args:
            x: Input features

        Returns:
            Latent mean vectors
        """
        with torch.no_grad():
            output = self.forward(x)
            return output['latent_mu']

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get attention weights for frequency analysis.

        Args:
            x: Input features

        Returns:
            Attention weights if attention is enabled
        """
        if not self.abr_encoder.use_attention:
            return None

        with torch.no_grad():
            output = self.forward(x)
            return output.get('attention_weights')

    def get_feature_importance(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze feature importance through gradient-based methods.

        Args:
            x: Input features

        Returns:
            Dictionary of feature importance scores
        """
        x.requires_grad_(True)
        output = self.forward(x)

        # Use latent norm as importance metric
        latent_norm = torch.norm(output['latent_z'], dim=1).sum()

        # Compute gradients
        gradients = torch.autograd.grad(latent_norm, x, create_graph=False)[0]

        # Calculate importance scores
        importance = torch.abs(gradients).mean(dim=0)

        return {
            'overall_importance': importance,
            'abr_importance': importance[:6],
            'metadata_importance': importance[6:]
        }