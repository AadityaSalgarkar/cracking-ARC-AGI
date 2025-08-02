import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

from classes import PuzzleShape
from layers.rope_self_attn import RoPEAttention


class TiedEmbedding(nn.Module):
    """
    Tied embedding layer for color tokens.
    
    Maps color indices to embeddings and back to color distributions.
    Uses the same weight matrix for both input embedding and output projection.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Single embedding matrix used for both input and output
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, color_indices: torch.Tensor) -> torch.Tensor:
        """
        Convert color indices to embeddings.
        
        Args:
            color_indices: (batch_size, seq_len) or (seq_len,) color indices
            
        Returns:
            embeddings: (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
        """
        return self.embedding(color_indices)
    
    def decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Convert embeddings back to color distributions using tied weights.
        
        Args:
            embeddings: (..., embed_dim) embeddings
            
        Returns:
            logits: (..., vocab_size) color distribution logits
        """
        # Use the transpose of embedding weights for output projection
        return F.linear(embeddings, self.embedding.weight)  # (..., vocab_size)


@gin.configurable
class RoPETransformerLayer(nn.Module):
    """Transformer layer with RoPE attention for encoding/decoding modules."""

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
        rope_theta=10.0,
        H=30,
        W=30,
    ):
        super().__init__()
        self.attn = RoPEAttention(
            embed_dim,
            num_heads,
            H=H,
            W=W,
            attn_drop=dropout,
            proj_drop=dropout,
            rope_theta=rope_theta,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, grid_shape=None):
        x = x + self.attn(self.norm1(x), grid_shape=grid_shape)
        x = x + self.ffn(self.norm2(x))
        return x


@gin.configurable
class StandardTransformerLayer(nn.Module):
    """Standard transformer layer without positional embeddings for thinking module."""

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # Standard self-attention without positional embeddings
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + attn_out
        x = self.norm1(x)

        # Feed-forward
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


@gin.configurable
class EncodingModule(nn.Module):
    """
    Encoding module with RoPE-based transformer using input cell positional embeddings.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        ff_dim=1024,
        dropout=0.1,
        rope_theta=10.0,
        H_max=30,
        W_max=30,
        C=11,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.H_max = H_max
        self.W_max = W_max
        self.C = C

        # Tied color embedding
        self.color_embedding = TiedEmbedding(C, embed_dim)

        # Position embedding projection (from 4 position embeddings to embed_dim)
        self.pos_projection = nn.Linear(4 * embed_dim, embed_dim)

        # RoPE-based transformer layers
        self.layers = nn.ModuleList(
            [
                RoPETransformerLayer(
                    embed_dim, num_heads, ff_dim, dropout, rope_theta, H_max, W_max
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, puzzle_shape: PuzzleShape):
        """
        Encode input PuzzleShape using input cell positional embeddings.

        Args:
            puzzle_shape: PuzzleShape object with input cells

        Returns:
            encoded: (batch_size, H_max*W_max, embed_dim) encoded sequence
        """
        # Get embeddings from all cells
        embeddings = []
        for cell in puzzle_shape.cells:
            # Get color embedding
            color_emb = self.color_embedding(torch.tensor(cell.Color, dtype=torch.long))

            # Get position embeddings (use input positions)
            pos_embs = cell.get_position_embeddings(self.embed_dim)
            pos_concat = torch.cat(pos_embs, dim=0)  # Concatenate 4 position embeddings
            pos_emb = self.pos_projection(pos_concat)

            # Combine color and position embeddings
            combined_emb = color_emb + pos_emb
            embeddings.append(combined_emb)

        # Stack all embeddings
        x = torch.stack(embeddings, dim=0).unsqueeze(0)  # (1, seq_len, embed_dim)

        # Pass through RoPE transformer layers
        for layer in self.layers:
            x = layer(x, grid_shape=(self.H_max, self.W_max))

        return self.layer_norm(x)


@gin.configurable
class ThinkingModule(nn.Module):
    """
    Thinking module with standard transformer (no positional embeddings).
    Pure reasoning without spatial bias.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        ff_dim=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Standard transformer layers without positional embeddings
        self.layers = nn.ModuleList(
            [
                StandardTransformerLayer(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        """
        Process encoded sequence through pure reasoning transformer.

        Args:
            x: (batch_size, seq_len, embed_dim) encoded sequence
            mask: Optional attention mask

        Returns:
            thought: (batch_size, seq_len, embed_dim) processed sequence
        """
        for layer in self.layers:
            x = layer(x, mask=mask)

        return self.layer_norm(x)


@gin.configurable
class DecodingModule(nn.Module):
    """
    Decoding module with RoPE-based transformer using learned type 1 (i,j) positional embeddings.
    Since we don't have H and W info at test time, uses only basic (i,j) coordinates.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        ff_dim=1024,
        dropout=0.1,
        rope_theta=10.0,
        H_max=30,
        W_max=30,
        C=11,
        tied_embedding=None,  # Shared tied embedding from encoding module
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.H_max = H_max
        self.W_max = W_max
        self.C = C

        # Learned positional embeddings for type 1 (i,j) coordinates only
        # Create learnable position embeddings for each (i,j) position in 30x30 grid
        self.positional_embeddings = nn.Parameter(
            torch.randn(H_max * W_max, embed_dim) * 0.02
        )

        # RoPE-based transformer layers
        self.layers = nn.ModuleList(
            [
                RoPETransformerLayer(
                    embed_dim, num_heads, ff_dim, dropout, rope_theta, H_max, W_max
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Use tied embedding for color output or create new one
        self.color_embedding = tied_embedding if tied_embedding is not None else TiedEmbedding(C, embed_dim)
        self.location_head = nn.Linear(embed_dim, 2)  # Predict (i, j) coordinates

    def forward(self, x):
        """
        Decode sequence using learned type 1 (i,j) positional embeddings.

        Args:
            x: (batch_size, seq_len, embed_dim) thought sequence

        Returns:
            colors: (batch_size, seq_len, C) predicted colors
            locations: (batch_size, seq_len, 2) predicted locations
        """
        batch_size, seq_len, embed_dim = x.shape

        # Add learned positional embeddings to the thought sequence
        # positional_embeddings shape: (H_max*W_max, embed_dim)
        # Expand to match batch size
        pos_emb = self.positional_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Add positional bias to the thought sequence
        x = x + pos_emb[:, :seq_len, :]  # Handle variable sequence lengths

        # Pass through RoPE transformer layers
        for layer in self.layers:
            x = layer(x, grid_shape=(self.H_max, self.W_max))

        x = self.layer_norm(x)

        # Generate predictions using tied embeddings
        colors = self.color_embedding.decode(x)  # (batch_size, seq_len, C) - color distributions
        locations = self.location_head(x)  # (batch_size, seq_len, 2)

        return colors, locations


@gin.configurable
class SpatialReasoningModel(nn.Module):
    """
    Complete three-module spatial reasoning model:
    1. EncodingModule: RoPE transformer with input positional embeddings
    2. ThinkingModule: Standard transformer without positional embeddings
    3. DecodingModule: RoPE transformer with output positional embeddings
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        encoder_layers=4,
        thinking_layers=6,
        decoder_layers=4,
        ff_dim=1024,
        dropout=0.1,
        rope_theta=10.0,
        H_max=30,
        W_max=30,
        C=11,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.H_max = H_max
        self.W_max = W_max
        self.C = C

        # Create shared tied embedding for color vocabulary
        self.tied_embedding = TiedEmbedding(C, embed_dim)
        
        # Input projection for handling TransformerIO embeddings (4*embed_dim -> embed_dim)
        self.input_projection = nn.Linear(4 * embed_dim, embed_dim)

        # Three modules
        self.encoding_module = EncodingModule(
            embed_dim,
            num_heads,
            encoder_layers,
            ff_dim,
            dropout,
            rope_theta,
            H_max,
            W_max,
            C,
        )
        # Replace encoding module's color embedding with shared tied embedding
        self.encoding_module.color_embedding = self.tied_embedding

        self.thinking_module = ThinkingModule(
            embed_dim, num_heads, thinking_layers, ff_dim, dropout
        )

        self.decoding_module = DecodingModule(
            embed_dim,
            num_heads,
            decoder_layers,
            ff_dim,
            dropout,
            rope_theta,
            H_max,
            W_max,
            C,
            tied_embedding=self.tied_embedding,  # Pass shared tied embedding
        )

    def forward(self, input_puzzle_shape: PuzzleShape):
        """
        Forward pass through the three-module architecture.

        Args:
            input_puzzle_shape: PuzzleShape with input cells and their positions

        Returns:
            colors: (batch_size, seq_len, C) predicted output colors
            locations: (batch_size, seq_len, 2) predicted output locations
        """
        # 1. Encoding: Process input with input positional embeddings
        encoded = self.encoding_module(input_puzzle_shape)

        # 2. Thinking: Pure reasoning without positional bias
        thought = self.thinking_module(encoded)

        # 3. Decoding: Generate output with learned positional embeddings
        colors, locations = self.decoding_module(thought)

        return colors, locations

    def forward_from_embeddings(self, input_embeddings: torch.Tensor):
        """
        Forward pass using pre-computed input embeddings from TransformerIO batch.
        
        This method bypasses the encoding module and works directly with embeddings,
        making it suitable for training with TransformerIO batched data.

        Args:
            input_embeddings: (batch_size, n_seq, 4*embed_dim) pre-computed input embeddings

        Returns:
            colors: (batch_size, n_seq, C) predicted output color distributions
            locations: (batch_size, n_seq, 2) predicted output locations
        """
        # Project from 4*embed_dim to embed_dim for thinking module
        projected = self.input_projection(input_embeddings)  # (batch_size, n_seq, embed_dim)

        # 2. Thinking: Pure reasoning without positional bias
        thought = self.thinking_module(projected)

        # 3. Decoding: Generate output with learned positional embeddings
        colors, locations = self.decoding_module(thought)

        return colors, locations

    def encode_only(self, input_puzzle_shape: PuzzleShape):
        """Get encoded representation of input."""
        return self.encoding_module(input_puzzle_shape)

    def think_only(self, encoded_sequence):
        """Process encoded sequence through thinking module."""
        return self.thinking_module(encoded_sequence)

    def decode_only(self, thought_sequence):
        """Decode thought sequence to output predictions."""
        return self.decoding_module(thought_sequence)


@gin.configurable
def create_spatial_reasoning_model(
    embed_dim=256,
    num_heads=8,
    encoder_layers=4,
    thinking_layers=6,
    decoder_layers=4,
    ff_dim=1024,
    dropout=0.1,
    rope_theta=10.0,
    H_max=30,
    W_max=30,
    C=11,
):
    """Factory function for creating three-module SpatialReasoningModel."""
    return SpatialReasoningModel(
        embed_dim=embed_dim,
        num_heads=num_heads,
        encoder_layers=encoder_layers,
        thinking_layers=thinking_layers,
        decoder_layers=decoder_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        rope_theta=rope_theta,
        H_max=H_max,
        W_max=W_max,
        C=C,
    )
