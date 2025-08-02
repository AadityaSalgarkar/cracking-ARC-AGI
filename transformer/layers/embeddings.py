"""
Embedding utilities for ARC spatial reasoning.

Implements sinusoidal embeddings for color values and Input class
with 4 positional coordinate transformations as specified in req.txt.
"""

import math

import torch


def create_sinusoidal_embedding(
    value: int, d_model: int, max_len: int = 10000
) -> torch.Tensor:
    """
    Create sinusoidal embedding for a single integer value between 0 to C-1.

    Args:
        value: Integer between 0 to C-1
        d_model: Embedding dimension (should be even)
        max_len: Maximum sequence length for frequency scaling

    Returns:
        embedding: Tensor of shape (d_model,) with sinusoidal embedding
    """
    assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
    assert value >= 0, f"value must be non-negative, got {value}"

    # Create sinusoidal embedding similar to standard positional encoding
    embedding = torch.zeros(d_model, dtype=torch.float32)

    # Create position vector
    position = torch.tensor(value, dtype=torch.float32)

    # Create frequency divisors
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * -(math.log(max_len) / d_model)
    )

    # Apply sin to even indices
    embedding[0::2] = torch.sin(position * div_term)

    # Apply cos to odd indices
    embedding[1::2] = torch.cos(position * div_term)

    return embedding


def create_batch_sinusoidal_embeddings(
    values: torch.Tensor, d_model: int, max_len: int = 10000
) -> torch.Tensor:
    """
    Create sinusoidal embeddings for a batch of integer values.

    Args:
        values: Tensor of integers between 0 to C-1, shape (...,)
        d_model: Embedding dimension (should be even)
        max_len: Maximum sequence length for frequency scaling

    Returns:
        embeddings: Tensor of shape (..., d_model) with sinusoidal embeddings
    """
    assert d_model % 2 == 0, f"d_model must be even, got {d_model}"

    # Get input shape for output
    input_shape = values.shape

    # Flatten input for processing
    values_flat = values.flatten().float()
    batch_size = values_flat.size(0)

    # Create batch embeddings
    embeddings = torch.zeros(batch_size, d_model, dtype=torch.float32)

    # Create frequency divisors
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * -(math.log(max_len) / d_model)
    )

    # Apply sinusoidal functions
    position_scaled = values_flat.unsqueeze(1) * div_term.unsqueeze(
        0
    )  # (batch_size, d_model//2)

    # Apply sin to even indices
    embeddings[:, 0::2] = torch.sin(position_scaled)

    # Apply cos to odd indices
    embeddings[:, 1::2] = torch.cos(position_scaled)

    # Reshape to match input dimensions
    output_shape = input_shape + (d_model,)
    embeddings = embeddings.view(output_shape)

    return embeddings
