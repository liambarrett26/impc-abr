"""
Attention mechanisms for audiometric feature processing.

This module provides specialized attention mechanisms for ABR frequency
relationships and cross-modal attention between ABR and metadata features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging
import math

logger = logging.getLogger(__name__)


class FrequencyAttention(nn.Module):
    """
    Multi-head attention mechanism specialized for ABR frequency relationships.

    Learns to weight the importance of different frequency measurements
    and their interactions for hearing phenotype characterization.
    """

    def __init__(self, input_dim: int, head_dim: int, num_heads: int,
                 dropout: float = 0.1, use_positional_encoding: bool = True):
        """
        Initialize frequency attention.

        Args:
            input_dim: Input feature dimension (6 for ABR frequencies)
            head_dim: Dimension per attention head
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_positional_encoding: Whether to use frequency positional encoding
        """
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.use_positional_encoding = use_positional_encoding

        # Frequency-specific parameters
        self.frequencies = torch.tensor([6.0, 12.0, 18.0, 24.0, 30.0, 0.0])  # kHz, 0 for click

        # Input projection to create token embeddings
        self.input_projection = nn.Linear(1, head_dim * num_heads)

        # Attention projections
        self.q_projection = nn.Linear(head_dim * num_heads, head_dim * num_heads)
        self.k_projection = nn.Linear(head_dim * num_heads, head_dim * num_heads)
        self.v_projection = nn.Linear(head_dim * num_heads, head_dim * num_heads)

        # Output projection
        self.output_projection = nn.Linear(head_dim * num_heads, input_dim)

        # Positional encoding for frequencies
        if use_positional_encoding:
            self.freq_encoding = FrequencyPositionalEncoding(head_dim * num_heads)

        # Dropout layers
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(head_dim * num_heads)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply frequency attention to ABR features.

        Args:
            x: Input ABR features (batch_size, num_frequencies)

        Returns:
            Tuple of (attended_features, attention_weights)
        """
        batch_size, seq_len = x.shape

        # Reshape input: treat each frequency as a token
        # (batch_size, seq_len) -> (batch_size, seq_len, 1)
        x_tokens = x.unsqueeze(-1)

        # Project to token embeddings
        token_embeddings = self.input_projection(x_tokens)  # (batch_size, seq_len, embed_dim)

        # Add positional encoding for frequencies
        if self.use_positional_encoding:
            token_embeddings = self.freq_encoding(token_embeddings)

        # Apply layer norm
        token_embeddings = self.layer_norm(token_embeddings)

        # Project to queries, keys, values
        q = self.q_projection(token_embeddings)
        k = self.k_projection(token_embeddings)
        v = self.v_projection(token_embeddings)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply frequency-specific attention bias (optional)
        scores = self._apply_frequency_bias(scores)

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.head_dim * self.num_heads
        )

        # Output projection
        output = self.output_projection(attended)
        output = self.output_dropout(output)

        # Residual connection: add to original input
        output = output.squeeze(-1) + x

        # Return mean attention weights across heads for interpretability
        mean_attention_weights = attention_weights.mean(dim=1)  # (batch_size, seq_len, seq_len)

        return output, mean_attention_weights

    def _apply_frequency_bias(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency-specific attention bias.

        Encourages attention between adjacent frequencies and
        known frequency relationships in audiology.

        Args:
            scores: Attention scores (batch_size, num_heads, seq_len, seq_len)

        Returns:
            Biased attention scores
        """
        seq_len = scores.size(-1)
        device = scores.device

        # Create frequency relationship bias matrix
        bias = torch.zeros(seq_len, seq_len, device=device)

        # Adjacent frequency bias (stronger connections between adjacent frequencies)
        for i in range(seq_len - 1):
            bias[i, i + 1] = 0.1  # Encourage attention to next frequency
            bias[i + 1, i] = 0.1  # And previous frequency

        # Octave relationships (e.g., 6kHz and 12kHz, 12kHz and 24kHz)
        octave_pairs = [(0, 1), (1, 3)]  # (6-12kHz), (12-24kHz)
        for i, j in octave_pairs:
            if i < seq_len and j < seq_len:
                bias[i, j] = 0.05
                bias[j, i] = 0.05

        # Add bias to scores
        return scores + bias.unsqueeze(0).unsqueeze(0)


class FrequencyPositionalEncoding(nn.Module):
    """
    Positional encoding based on frequency values rather than sequence position.

    Uses the actual frequency values (6, 12, 18, 24, 30 kHz) to create
    meaningful positional embeddings that reflect acoustic relationships.
    """

    def __init__(self, embed_dim: int, max_freq: float = 30.0):
        """
        Initialize frequency positional encoding.

        Args:
            embed_dim: Embedding dimension
            max_freq: Maximum frequency for normalization
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_freq = max_freq

        # Frequency values for ABR measurements
        self.register_buffer('frequencies',
                           torch.tensor([6.0, 12.0, 18.0, 24.0, 30.0, 0.0]))  # 0 for click

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add frequency-based positional encoding.

        Args:
            x: Input embeddings (batch_size, seq_len, embed_dim)

        Returns:
            Embeddings with positional encoding added
        """
        batch_size, seq_len, embed_dim = x.shape

        # Normalize frequencies
        normalized_freqs = self.frequencies[:seq_len] / self.max_freq

        # Create positional encoding using sinusoidal functions
        pe = torch.zeros(seq_len, embed_dim, device=x.device)

        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=x.device) *
                           -(math.log(10000.0) / embed_dim))

        # Use normalized frequency instead of position
        for i in range(seq_len):
            freq = normalized_freqs[i]
            pe[i, 0::2] = torch.sin(freq * div_term)
            pe[i, 1::2] = torch.cos(freq * div_term)

        # Add positional encoding to input
        return x + pe.unsqueeze(0)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between ABR features and metadata.

    Allows ABR frequency information to attend to relevant metadata
    (age, genetic background, etc.) and vice versa.
    """

    def __init__(self, abr_dim: int, metadata_dim: int,
                 embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize cross-modal attention.

        Args:
            abr_dim: ABR feature dimension
            metadata_dim: Metadata feature dimension
            embed_dim: Embedding dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.abr_dim = abr_dim
        self.metadata_dim = metadata_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projection layers for different modalities
        self.abr_projection = nn.Linear(abr_dim, embed_dim)
        self.metadata_projection = nn.Linear(metadata_dim, embed_dim)

        # Cross-attention layers
        self.abr_to_meta_attention = MultiHeadCrossAttention(
            embed_dim, num_heads, dropout
        )
        self.meta_to_abr_attention = MultiHeadCrossAttention(
            embed_dim, num_heads, dropout
        )

        # Output projections
        self.abr_output = nn.Linear(embed_dim, abr_dim)
        self.metadata_output = nn.Linear(embed_dim, metadata_dim)

        # Layer normalization
        self.abr_norm = nn.LayerNorm(embed_dim)
        self.metadata_norm = nn.LayerNorm(embed_dim)

    def forward(self, abr_features: torch.Tensor,
                metadata_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply cross-modal attention.

        Args:
            abr_features: ABR features (batch_size, abr_dim)
            metadata_features: Metadata features (batch_size, metadata_dim)

        Returns:
            Tuple of (attended_abr, attended_metadata, attention_weights)
        """
        batch_size = abr_features.size(0)

        # Project to common embedding space
        abr_embed = self.abr_projection(abr_features.unsqueeze(1))  # (batch_size, 1, embed_dim)
        meta_embed = self.metadata_projection(metadata_features.unsqueeze(1))  # (batch_size, 1, embed_dim)

        # Apply layer normalization
        abr_embed = self.abr_norm(abr_embed)
        meta_embed = self.metadata_norm(meta_embed)

        # Cross-attention: ABR attends to metadata
        abr_attended, abr_to_meta_weights = self.abr_to_meta_attention(
            abr_embed, meta_embed, meta_embed
        )

        # Cross-attention: Metadata attends to ABR
        meta_attended, meta_to_abr_weights = self.meta_to_abr_attention(
            meta_embed, abr_embed, abr_embed
        )

        # Project back to original dimensions
        abr_output = self.abr_output(abr_attended.squeeze(1))
        metadata_output = self.metadata_output(meta_attended.squeeze(1))

        # Residual connections
        abr_output = abr_output + abr_features
        metadata_output = metadata_output + metadata_features

        attention_weights = {
            'abr_to_metadata': abr_to_meta_weights,
            'metadata_to_abr': meta_to_abr_weights
        }

        return abr_output, metadata_output, attention_weights


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention module.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """Initialize multi-head cross-attention."""
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projection layers
        self.q_projection = nn.Linear(embed_dim, embed_dim)
        self.k_projection = nn.Linear(embed_dim, embed_dim)
        self.v_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head cross-attention.

        Args:
            query: Query tensor (batch_size, seq_len_q, embed_dim)
            key: Key tensor (batch_size, seq_len_k, embed_dim)
            value: Value tensor (batch_size, seq_len_v, embed_dim)

        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]

        # Project to Q, K, V
        q = self.q_projection(query)
        k = self.k_projection(key)
        v = self.v_projection(value)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, v)

        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )

        # Output projection
        output = self.output_projection(attended)

        # Return mean attention weights across heads
        mean_attention_weights = attention_weights.mean(dim=1)

        return output, mean_attention_weights


class AttentionAnalyzer:
    """
    Utility class for analyzing attention patterns and interpretability.
    """

    def __init__(self):
        """Initialize attention analyzer."""
        pass

    def analyze_frequency_attention(self, attention_weights: torch.Tensor,
                                  frequency_labels: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Analyze frequency attention patterns.

        Args:
            attention_weights: Attention weight matrix (batch_size, seq_len, seq_len)
            frequency_labels: Labels for frequencies

        Returns:
            Dictionary of attention analysis results
        """
        if frequency_labels is None:
            frequency_labels = ['6kHz', '12kHz', '18kHz', '24kHz', '30kHz', 'Click']

        with torch.no_grad():
            # Average attention patterns across batch
            mean_attention = attention_weights.mean(dim=0)

            # Self-attention vs cross-attention
            diagonal = torch.diag(mean_attention)
            off_diagonal = mean_attention - torch.diag(diagonal)

            # Frequency band interactions
            low_freq_attn = mean_attention[:2, :2].mean()  # 6-12 kHz
            mid_freq_attn = mean_attention[2:4, 2:4].mean()  # 18-24 kHz
            high_freq_attn = mean_attention[4:, 4:].mean()  # 30 kHz + Click

            # Cross-band attention
            low_to_high = mean_attention[:2, 4:].mean()
            high_to_low = mean_attention[4:, :2].mean()

            return {
                'mean_attention_matrix': mean_attention,
                'self_attention_strength': diagonal.mean(),
                'cross_attention_strength': off_diagonal.mean(),
                'low_freq_attention': low_freq_attn,
                'mid_freq_attention': mid_freq_attn,
                'high_freq_attention': high_freq_attn,
                'low_to_high_attention': low_to_high,
                'high_to_low_attention': high_to_low,
                'attention_entropy': self._compute_attention_entropy(attention_weights)
            }

    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distributions."""
        # Entropy over the last dimension (attention targets)
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        return entropy.mean()

    def visualize_attention_heatmap(self, attention_weights: torch.Tensor,
                                   frequency_labels: List[str] = None) -> torch.Tensor:
        """
        Prepare attention weights for heatmap visualization.

        Args:
            attention_weights: Attention weights (batch_size, seq_len, seq_len)
            frequency_labels: Frequency labels

        Returns:
            Mean attention matrix for plotting
        """
        with torch.no_grad():
            mean_attention = attention_weights.mean(dim=0)
            return mean_attention