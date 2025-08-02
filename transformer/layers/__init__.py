"""
Layers module for ARC spatial reasoning transformer.

Contains attention mechanisms, embeddings, and other neural network layers.
"""

from .embeddings import (
    create_batch_sinusoidal_embeddings,
    create_sinusoidal_embedding,
)
from .rope_self_attn import (
    Attention,
    RoPEAttention,
    apply_rotary_emb,
    compute_axial_cis,
    compute_mixed_cis,
    init_2d_freqs,
    init_t_xy,
)

# Note: Cell, PuzzleShape, and Puzzle classes are available in classes.py
# Import directly: from classes import Cell, PuzzleShape, Puzzle
from .sequence_gen import (
    generate_sequence_with_rope,
)

__all__ = [
    # Attention layers
    "Attention",
    "RoPEAttention",
    "init_2d_freqs",
    "init_t_xy",
    "compute_mixed_cis",
    "compute_axial_cis",
    "apply_rotary_emb",
    # Embedding utilities
    "create_sinusoidal_embedding",
    "create_batch_sinusoidal_embeddings",
    # Note: Cell, PuzzleShape, Puzzle are in classes.py
    # Sequence generation
    "generate_sequence_with_rope",
]
