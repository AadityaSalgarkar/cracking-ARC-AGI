"""
Positional embedding utilities for the transformer layers.
"""

import math
import torch


@torch.no_grad()
def create_sinusoidal_embedding(
    position: int, 
    d_model: int, 
    base: int = 10000
) -> torch.Tensor:
    """
    Create sinusoidal positional embedding for a single position.
    
    Args:
        position: Integer position value
        d_model: Embedding dimension
        base: Base for frequency computation
        
    Returns:
        Tensor of shape (d_model,) with sinusoidal positional embedding
    """
    assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
    
    embedding = torch.zeros(d_model, dtype=torch.float32)
    
    # Create frequency divisors
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * 
        -(math.log(base) / d_model)
    )
    
    # Apply sinusoidal encoding
    embedding[0::2] = torch.sin(position * div_term)
    embedding[1::2] = torch.cos(position * div_term)
    
    return embedding