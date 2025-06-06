"""
Multitask Neural Network Architecture for Genome-Wide Association Studies (GWAS)

This module implements a deep learning architecture designed for simultaneous analysis
of multiple phenotypic traits in genomic data, incorporating attention mechanisms
for task-specific feature weighting.

Author: [Your Name]
Institution: [Your Institution]
Date: [Current Date]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class AttentionPooling(nn.Module):
    """
    Attention-based pooling mechanism for task-specific feature selection.
    
    This module computes attention weights for each task, allowing the model
    to focus on different genomic features for different phenotypic traits.
    
    Args:
        input_dim (int): Dimension of input features
        num_tasks (int): Number of phenotypic traits to predict
        temperature (float): Temperature parameter for softmax normalization
    """
    
    def __init__(self, input_dim: int, num_tasks: int, temperature: float = 1.0):
        super(AttentionPooling, self).__init__()
        self.temperature = temperature
        self.attention_weights = nn.Linear(input_dim, num_tasks)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of attention pooling.
        
        Args:
            features: Input features of shape (batch_size, feature_dim)
            
        Returns:
            Tuple of (attention_weights, normalized_features)
        """
        # Apply layer normalization for stability
        normalized_features = self.layer_norm(features)
        
        # Compute attention weights
        attention_logits = self.attention_weights(normalized_features) / self.temperature
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        return attention_weights, normalized_features


class TaskSpecificHead(nn.Module):
    """
    Task-specific prediction head with residual connections.
    
    Each phenotypic trait has its own dedicated neural network head
    for specialized feature processing and prediction.
    
    Args:
        input_dim (int): Dimension of shared features
        hidden_dim (int): Dimension of hidden layers
        output_dim (int): Dimension of output (typically 1 for binary traits)
        dropout_rate (float): Dropout probability for regularization
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float):
        super(TaskSpecificHead, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Residual connection for input dimension matching
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        main_output = self.layers(x)
        residual_output = self.residual(x)
        return torch.sigmoid(main_output + residual_output)


class SharedFeatureExtractor(nn.Module):
    """
    Shared feature extraction backbone for genomic data.
    
    This module learns common genomic representations that are useful
    across multiple phenotypic traits, implementing a deep architecture
    with batch normalization and skip connections.
    
    Args:
        input_dim (int): Number of genetic variants (SNPs)
        hidden_dims (List[int]): Dimensions of hidden layers
        dropout_rate (float): Dropout probability for regularization
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float):
        super(SharedFeatureExtractor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
        # Initialize weights using He initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using He initialization for ReLU networks."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared genomic features."""
        return self.layers(x)


class MultitaskGWASModel(nn.Module):
    """
    Multitask Neural Network for Genome-Wide Association Studies.
    
    This architecture simultaneously predicts multiple phenotypic traits from
    genomic data using a shared feature extraction backbone with task-specific
    attention mechanisms and prediction heads.
    
    Key Features:
    - Shared genomic feature extraction
    - Task-specific attention pooling
    - Individual prediction heads per trait
    - Batch normalization and dropout for regularization
    - Residual connections for improved gradient flow
    
    Args:
        input_dim (int): Number of genetic variants (SNPs) in input
        num_tasks (int): Number of phenotypic traits to predict
        hidden_dims (List[int]): Dimensions of shared hidden layers
        task_hidden_dim (int): Hidden dimension for task-specific heads
        output_dim (int): Output dimension per task (typically 1)
        dropout_rate (float): Dropout probability for regularization
        attention_temperature (float): Temperature for attention softmax
    
    Example:
        >>> model = MultitaskGWASModel(
        ...     input_dim=10000,  # 10K SNPs
        ...     num_tasks=3,      # 3 traits
        ...     hidden_dims=[512, 256, 128],
        ...     task_hidden_dim=64,
        ...     dropout_rate=0.3
        ... )
        >>> output = model(genomic_data)  # Shape: (batch_size, num_tasks)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        hidden_dims: Optional[List[int]] = None,
        task_hidden_dim: int = 64,
        output_dim: int = 1,
        dropout_rate: float = 0.3,
        attention_temperature: float = 1.0
    ):
        super(MultitaskGWASModel, self).__init__()
        
        # Default architecture if not specified
        if hidden_dims is None:
            hidden_dims = [min(512, input_dim // 2), 256, 128]
        
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        
        # Shared feature extraction backbone
        self.shared_extractor = SharedFeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
        
        # Attention mechanism for task-specific feature selection
        self.attention_pool = AttentionPooling(
            input_dim=self.shared_extractor.output_dim,
            num_tasks=num_tasks,
            temperature=attention_temperature
        )
        
        # Task-specific prediction heads
        self.task_heads = nn.ModuleList([
            TaskSpecificHead(
                input_dim=self.shared_extractor.output_dim,
                hidden_dim=task_hidden_dim,
                output_dim=output_dim,
                dropout_rate=dropout_rate
            ) for _ in range(num_tasks)
        ])
        
        # Model metadata
        self.model_info = {
            'input_dim': input_dim,
            'num_tasks': num_tasks,
            'hidden_dims': hidden_dims,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multitask GWAS model.
        
        Args:
            x: Input genomic data of shape (batch_size, input_dim)
            
        Returns:
            Predictions for all tasks of shape (batch_size, num_tasks * output_dim)
        """
        # Extract shared genomic features
        shared_features = self.shared_extractor(x)
        
        # Compute attention weights and normalize features
        attention_weights, normalized_features = self.attention_pool(shared_features)
        
        # Generate task-specific predictions
        task_outputs = []
        for task_idx in range(self.num_tasks):
            # Apply task-specific attention weighting
            task_attention = attention_weights[:, task_idx:task_idx+1]
            attended_features = normalized_features * task_attention
            
            # Generate task prediction
            task_output = self.task_heads[task_idx](attended_features)
            task_outputs.append(task_output)
        
        # Concatenate all task outputs
        return torch.cat(task_outputs, dim=-1)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for interpretability analysis.
        
        Args:
            x: Input genomic data
            
        Returns:
            Attention weights of shape (batch_size, num_tasks)
        """
        with torch.no_grad():
            shared_features = self.shared_extractor(x)
            attention_weights, _ = self.attention_pool(shared_features)
            return attention_weights
    
    def get_model_summary(self) -> dict:
        """Return model architecture summary for reporting."""
        return {
            **self.model_info,
            'shared_extractor_params': sum(p.numel() for p in self.shared_extractor.parameters()),
            'attention_params': sum(p.numel() for p in self.attention_pool.parameters()),
            'task_heads_params': sum(p.numel() for p in self.task_heads.parameters()),
            'architecture': 'Multitask GWAS with Attention Pooling'
        }


# Utility function for model instantiation
def create_gwas_model(
    num_snps: int,
    num_traits: int,
    model_complexity: str = 'medium'
) -> MultitaskGWASModel:
    """
    Factory function to create GWAS models with predefined architectures.
    
    Args:
        num_snps: Number of SNPs in the genomic data
        num_traits: Number of phenotypic traits to predict
        model_complexity: Model size ('small', 'medium', 'large')
        
    Returns:
        Configured MultitaskGWASModel instance
    """
    complexity_configs = {
        'small': {'hidden_dims': [128, 64], 'task_hidden_dim': 32, 'dropout_rate': 0.2},
        'medium': {'hidden_dims': [512, 256, 128], 'task_hidden_dim': 64, 'dropout_rate': 0.3},
        'large': {'hidden_dims': [1024, 512, 256, 128], 'task_hidden_dim': 128, 'dropout_rate': 0.4}
    }
    
    config = complexity_configs.get(model_complexity, complexity_configs['medium'])
    
    return MultitaskGWASModel(
        input_dim=num_snps,
        num_tasks=num_traits,
        **config
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Multitask GWAS Model - Academic Implementation")
    print("=" * 50)
    
    # Create model instance
    model = create_gwas_model(num_snps=1000, num_traits=3, model_complexity='medium')
    
    # Print model summary
    summary = model.get_model_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Test forward pass
    batch_size = 32
    test_input = torch.randn(batch_size, 1000)
    
    with torch.no_grad():
        output = model(test_input)
        attention = model.get_attention_weights(test_input)
    
    print(f"\nTest Results:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
