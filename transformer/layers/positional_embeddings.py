"""
2D Positional Embeddings for ARC spatial reasoning.

Implements 2D rotary position embeddings (RoPE) for grid-based data,
combining functionality from the removed embeddings.py, rope_self_attn.py,
and sequence_gen.py files.
"""

import math
import gin
import torch
import torch.nn as nn


def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    """Initialize 2D frequency components for RoPE."""
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for _ in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat(
            [mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1
        )
        fy = torch.cat(
            [mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1
        )
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs


def init_t_xy(end_x: int, end_y: int):
    """Initialize x,y coordinate tensors for a grid."""
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    """Compute axial rotary position embeddings for 2D grids."""
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape frequency tensor for broadcasting with input tensor."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply rotary embeddings to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


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

    embedding = torch.zeros(d_model, dtype=torch.float32)
    position = torch.tensor(value, dtype=torch.float32)

    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * -(math.log(max_len) / d_model)
    )

    embedding[0::2] = torch.sin(position * div_term)
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

    input_shape = values.shape
    values_flat = values.flatten().float()
    batch_size = values_flat.size(0)

    embeddings = torch.zeros(batch_size, d_model, dtype=torch.float32)

    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * -(math.log(max_len) / d_model)
    )

    position_scaled = values_flat.unsqueeze(1) * div_term.unsqueeze(0)
    embeddings[:, 0::2] = torch.sin(position_scaled)
    embeddings[:, 1::2] = torch.cos(position_scaled)

    output_shape = input_shape + (d_model,)
    embeddings = embeddings.view(output_shape)

    return embeddings


@gin.configurable
def generate_sequence_with_2d_pos(
    grid, value_embedding, n_seq=100, embed_dim=256, rope_theta=10000
):
    """
    Convert HxW grid to sequence with 2D positional embeddings.

    Args:
        grid: (batch_size, H, W) tensor with values 0 to C-1
        value_embedding: nn.Embedding layer for grid values
        n_seq: sequence length
        embed_dim: embedding dimension
        rope_theta: RoPE theta parameter

    Returns:
        sequence: (batch_size, n_seq, embed_dim) tensor
    """
    batch_size, H, W = grid.shape
    device = grid.device

    # Flatten grid to sequence and take first n_seq positions
    grid_flat = grid.view(batch_size, -1)  # (batch_size, H*W)
    if grid_flat.size(1) >= n_seq:
        grid_seq = grid_flat[:, :n_seq]
    else:
        # Pad if needed
        padding = torch.zeros(
            batch_size, n_seq - grid_flat.size(1), device=device, dtype=grid.dtype
        )
        grid_seq = torch.cat([grid_flat, padding], dim=1)

    # Get value embeddings
    value_emb = value_embedding(grid_seq)  # (batch_size, n_seq, embed_dim)

    # Create 2D positional embeddings based on grid coordinates
    pos_ids = torch.arange(n_seq, device=device, dtype=torch.long)
    
    # Convert flat indices back to 2D coordinates
    y_coords = pos_ids // W
    x_coords = pos_ids % W
    
    # Create separate embeddings for x and y coordinates
    x_div_term = torch.exp(
        torch.arange(0, embed_dim // 2, 2, device=device, dtype=torch.float) 
        * -(math.log(rope_theta) / (embed_dim // 2))
    )
    y_div_term = torch.exp(
        torch.arange(0, embed_dim // 2, 2, device=device, dtype=torch.float) 
        * -(math.log(rope_theta) / (embed_dim // 2))
    )
    
    # Create positional encoding
    pos_emb = torch.zeros(n_seq, embed_dim, device=device, dtype=torch.float)
    
    # X coordinate embeddings (first half of dimensions)
    pos_emb[:, 0:embed_dim//4:2] = torch.sin(x_coords.unsqueeze(-1).float() * x_div_term)
    pos_emb[:, 1:embed_dim//4:2] = torch.cos(x_coords.unsqueeze(-1).float() * x_div_term)
    
    # Y coordinate embeddings (second half of dimensions)  
    pos_emb[:, embed_dim//2:3*embed_dim//4:2] = torch.sin(y_coords.unsqueeze(-1).float() * y_div_term)
    pos_emb[:, embed_dim//2+1:3*embed_dim//4:2] = torch.cos(y_coords.unsqueeze(-1).float() * y_div_term)
    
    # Expand for batch
    pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)

    # Combine value and positional embeddings
    sequence = value_emb + pos_emb

    return sequence


class Attention(nn.Module):
    """Standard multi-head attention."""
    
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RoPE2DAttention(Attention):
    """Multi-head Attention block with 2D rotary position embeddings."""

    def __init__(self, dim, num_heads=8, H=10, W=10, rope_theta=10.0, **kwargs):
        super().__init__(dim, num_heads, **kwargs)
        self.H = H
        self.W = W
        self.rope_theta = rope_theta

    def forward(self, x, grid_shape=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Determine grid dimensions
        if grid_shape is not None:
            H, W = grid_shape
        else:
            H, W = self.H, self.W

        # Handle case where sequence length doesn't match H*W (for TransformerIO)
        if N != H * W:
            # Use simple positional encoding for arbitrary sequence lengths
            dim = C // self.num_heads
            freqs = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(N, dtype=torch.float32).to(x.device)
            freqs = torch.outer(t, freqs)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            
            # Pad to match the expected head dimension if needed
            if freqs_cis.shape[-1] < dim // 2:
                pad_size = dim // 2 - freqs_cis.shape[-1]
                padding = torch.ones(N, pad_size, dtype=torch.complex64).to(x.device)
                freqs_cis = torch.cat([freqs_cis, padding], dim=-1)
        else:
            # Use 2D axial approach for full grid sequences
            freqs_cis = compute_axial_cis(C // self.num_heads, H, W, self.rope_theta)
            freqs_cis = freqs_cis.to(x.device)

        # Apply rotary embeddings to q and k
        q, k = apply_rotary_emb(q, k, freqs_cis)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x