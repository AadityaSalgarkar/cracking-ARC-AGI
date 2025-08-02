import gin
import torch


@gin.configurable
def generate_sequence_with_rope(
    grid, value_embedding, pos_projection, n_seq=100, embed_dim=256, rope_theta=10000
):
    """
    Convert HxW grid to sequence with rotational positional embeddings.

    Args:
        grid: (batch_size, H, W) tensor with values 0 to C-1
        value_embedding: nn.Embedding layer for grid values
        pos_projection: nn.Linear layer to project positional embeddings
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

    # Simple positional encoding based on sequence position
    pos_ids = torch.arange(n_seq, device=device, dtype=torch.float)
    pos_emb = torch.sin(
        pos_ids.unsqueeze(-1)
        / (rope_theta ** (torch.arange(embed_dim, device=device) / embed_dim))
    )
    pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)

    # Combine value and positional embeddings
    sequence = value_emb + pos_emb

    return sequence
