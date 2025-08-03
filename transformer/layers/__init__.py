"""
Layers module for ARC spatial reasoning transformer.

Contains attention mechanisms, 2D positional embeddings, and other neural network layers.
"""

from .positional_embeddings import (
    Attention,
    RoPE2DAttention,
    apply_rotary_emb,
    compute_axial_cis,
    init_2d_freqs,
    init_t_xy,
    create_batch_sinusoidal_embeddings,
    create_sinusoidal_embedding,
    generate_sequence_with_2d_pos,
)

from .Encoding_Module import (
    EncodingModule,
    create_2d_positional_embedding,
)

from .classes import (
    Cell,
    PuzzleShape,
    Puzzle,
    TransformerIO,
)

__all__ = [
    # Attention layers
    "Attention", 
    "RoPE2DAttention",
    "init_2d_freqs",
    "init_t_xy",
    "compute_axial_cis",
    "apply_rotary_emb",
    # Embedding utilities
    "create_sinusoidal_embedding",
    "create_batch_sinusoidal_embeddings",
    # 2D positional sequence generation
    "generate_sequence_with_2d_pos",
    # Encoding Module
    "EncodingModule",
    "create_2d_positional_embedding",
    # Classes
    "Cell",
    "PuzzleShape", 
    "Puzzle",
    "TransformerIO",
]
