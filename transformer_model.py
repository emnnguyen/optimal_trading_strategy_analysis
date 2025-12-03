"""
Transformer Model for Stock Trading Signal Prediction

This module implements a transformer-based architecture for multi-class
classification (BUY/SELL/HOLD) of stock trading signals.

Architecture Features:
- Positional encoding for temporal sequence awareness
- Multi-head self-attention for temporal patterns
- Cross-attention between temporal and feature dimensions
- Layer normalization and dropout for regularization

Author: Trading ML System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    
    Adds positional information to the input embeddings so the transformer
    can understand the sequential nature of the data.
    
    Args:
        d_model: Dimension of the model embeddings
        max_len: Maximum sequence length to support
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class FeatureAttention(nn.Module):
    """
    Cross-attention module for attending to feature dimensions.
    
    This allows the model to learn which features are most important
    for predicting trading signals.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feature attention.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with feature attention applied
        """
        # Self-attention across the sequence
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block with self-attention and feed-forward layers.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward network dimension
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder block.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            src_mask: Optional attention mask
            src_key_padding_mask: Optional padding mask
            
        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(
            x, x, x, 
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TradingTransformer(nn.Module):
    """
    Complete Transformer model for trading signal prediction.
    
    This model takes sequences of stock features and predicts
    BUY/SELL/HOLD signals using transformer architecture.
    
    Args:
        num_features: Number of input features (75)
        seq_length: Length of input sequences (40)
        d_model: Model dimension (128)
        num_heads: Number of attention heads (8)
        num_layers: Number of encoder layers (4)
        d_ff: Feed-forward dimension (512)
        num_classes: Number of output classes (3 for BUY/HOLD/SELL)
        dropout: Dropout probability (0.1)
    """
    
    def __init__(
        self,
        num_features: int = 78,
        seq_length: int = 40,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_features = num_features
        self.seq_length = seq_length
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection: project features to model dimension
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_length + 10, dropout=dropout)
        
        # Feature attention layer (cross-attention on features)
        self.feature_attention = FeatureAttention(d_model, num_heads, dropout)
        
        # Stack of transformer encoder blocks
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Global pooling options
        self.global_attention = nn.Linear(d_model, 1)  # For attention-based pooling
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, num_features)
            return_attention: Whether to return attention weights for interpretability
            
        Returns:
            logits: Output logits of shape (batch_size, num_classes)
            attention_weights: Optional attention weights if return_attention=True
        """
        batch_size = x.size(0)
        
        # Project input features to model dimension
        # (batch_size, seq_length, num_features) -> (batch_size, seq_length, d_model)
        x = self.input_projection(x)
        
        # Transpose for transformer: (batch_size, seq_length, d_model) -> (seq_length, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply feature attention
        x = self.feature_attention(x)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        # Transpose back: (seq_length, batch_size, d_model) -> (batch_size, seq_length, d_model)
        x = x.transpose(0, 1)
        
        # Attention-based pooling
        attention_weights = F.softmax(self.global_attention(x).squeeze(-1), dim=-1)
        x = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
        
        # Classification
        logits = self.classifier(x)
        
        if return_attention:
            return logits, attention_weights
        return logits, None
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, num_features)
            
        Returns:
            Softmax probabilities of shape (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, num_features)
            
        Returns:
            Predicted class indices of shape (batch_size,)
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)


class ModelConfig:
    """Configuration class for model hyperparameters."""
    
    def __init__(
        self,
        num_features: int = 75,
        seq_length: int = 40,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        num_classes: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        num_epochs: int = 100,
        early_stopping_patience: int = 10
    ):
        self.num_features = num_features
        self.seq_length = seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.num_classes = num_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


def create_model(config: Optional[ModelConfig] = None) -> TradingTransformer:
    """
    Factory function to create a TradingTransformer model.
    
    Args:
        config: Optional ModelConfig. If None, uses default configuration.
        
    Returns:
        Initialized TradingTransformer model
    """
    if config is None:
        config = ModelConfig()
    
    model = TradingTransformer(
        num_features=config.num_features,
        seq_length=config.seq_length,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        num_classes=config.num_classes,
        dropout=config.dropout
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    config = ModelConfig()
    model = create_model(config)
    
    # Print model summary
    print(f"Model Configuration:")
    print(f"  - Input features: {config.num_features}")
    print(f"  - Sequence length: {config.seq_length}")
    print(f"  - Model dimension: {config.d_model}")
    print(f"  - Attention heads: {config.num_heads}")
    print(f"  - Encoder layers: {config.num_layers}")
    print(f"  - Feed-forward dim: {config.d_ff}")
    print(f"  - Output classes: {config.num_classes}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, config.seq_length, config.num_features)
    
    model.eval()
    with torch.no_grad():
        logits, attention = model(x, return_attention=True)
        probs = F.softmax(logits, dim=-1)
    
    print(f"\nTest forward pass:")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output logits shape: {logits.shape}")
    print(f"  - Output probs shape: {probs.shape}")
    print(f"  - Attention weights shape: {attention.shape}")
    print(f"  - Sample probabilities: {probs[0].numpy()}")
