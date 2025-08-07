"""
Encoding Module for ARC spatial reasoning.

Implements the transformer specified in req.txt with:
- Input: [batch_size, ctx_len] tuples (color, location=(i,j))
- Tied embeddings for color of size d_model
- 2D positional embeddings with 4 coordinate transformations
- Configurable via gin
"""

import math
import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from .classes import Coordinates

@torch.no_grad()
def create_2d_positional_embedding(i: int, j: int, d_positional_input: int, max_len: int = 10000) -> torch.Tensor:
    """
    Create 2D positional embedding for coordinates (i,j).
    
    Args:
        i: Row coordinate
        j: Column coordinate  
        d_positional_input: Positional embedding dimension
        max_len: Maximum sequence length for frequency scaling
        
    Returns:
        Tensor of shape (d_positional_input,) with 2D positional embedding
    """
    assert d_positional_input % 2 == 0, f"d_positional_input must be even, got {d_positional_input}"
    
    embedding = torch.zeros(d_positional_input, dtype=torch.float32)
    
    # Use i for first half of dimensions, j for second half
    half_dim = d_positional_input // 2
    
    # Create frequency divisors
    div_term = torch.exp(
        torch.arange(0, half_dim, 2, dtype=torch.float32) * -(math.log(max_len) / half_dim)
    )
    
    # i coordinate embeddings (first half)
    embedding[0:half_dim:2] = torch.sin(i * div_term)
    embedding[1:half_dim:2] = torch.cos(i * div_term)
    
    # j coordinate embeddings (second half)
    embedding[half_dim:half_dim + len(div_term) * 2:2] = torch.sin(j * div_term)
    embedding[half_dim + 1:half_dim + len(div_term) * 2:2] = torch.cos(j * div_term)
    
    return embedding


class CustomTransformerEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer using standard nn.MultiheadAttention.
    
    Similar to nn.TransformerEncoderLayer but allows adding positional embeddings
    before the attention computation.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5
    ):
        """
        Initialize Custom Transformer Encoder Layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function ("gelu" or "relu")
            layer_norm_eps: Layer norm epsilon
        """
        super().__init__()
        
        # Multi-head attention (supports flash attention internally in PyTorch 2.0+)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(
        self,
        src: torch.Tensor,
        positional_embedding: torch.Tensor = None,
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through the encoder layer.
        
        Args:
            src: Input tensor [batch_size, seq_len, d_model]
            positional_embedding: Optional positional embedding to add before attention
            src_mask: Optional source mask
            src_key_padding_mask: Optional source key padding mask
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm (pre-norm)
        src_norm = self.norm1(src)
        
        # Add positional embedding before attention if provided
        if positional_embedding is not None:
            src_with_pos = src_norm + positional_embedding
        else:
            src_with_pos = src_norm
            
        attn_output, _ = self.self_attn(
            query=src_with_pos,
            key=src_with_pos,
            value=src_with_pos,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(attn_output)
        
        # Feed-forward with residual connection and layer norm (pre-norm)
        src_norm = self.norm2(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(ff_output)
        
        return src


@gin.configurable
class CustomEncoderModule(nn.Module):
    """
    Transformer with input ctx_len list of (i,j) positions as specified in req.txt.
    
    Input: [batch_size, ctx_len] tuples (color, location=(i,j))
    - Tied embeddings for color of size d_model
    - 2D positional embeddings with 4 coordinate transformations
    - Standard encoder transformer for n_layers
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        C: int = 11,
        H_max: int = 30,
        W_max: int = 30,
        max_len: int = 10000
    ):
        """
        Initialize Encoding Module.
        
        Args:
            d_model: Model dimension (must be divisible by 4)
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            C: Number of colors (0 to C-1)
            H_max: Maximum height for coordinate transformations
            W_max: Maximum width for coordinate transformations
            max_len: Maximum sequence length for positional embeddings
        """
        super().__init__()
        
        assert d_model % 4 == 0, f"d_model must be divisible by 4, got {d_model}"
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.H_max = H_max
        self.W_max = W_max
        self.d_positional_input = d_model // 4
        self.max_len = max_len
        
        # Tied embeddings for color
        self.color_embedding = nn.Embedding(C, d_model)
        
        # Custom transformer encoder layers using CustomMHA
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
    def create_positional_embeddings_input(
        self, 
        positions: torch.Tensor, 
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Create 4 positional embeddings for each (i,j) position.
        
        Args:
            positions: Tensor of shape [batch_size, ctx_len, 2] with (i,j) coordinates
            H: Height of current grid
            W: Width of current grid
            
        Returns:
            Tensor of shape [batch_size, ctx_len, d_model] with combined positional embeddings
        """
        batch_size, ctx_len, _ = positions.shape
        device = positions.device
        
        # Extract i,j coordinates
        i_coords = positions[:, :, 0]  # [batch_size, ctx_len]
        j_coords = positions[:, :, 1]  # [batch_size, ctx_len]
        
        # Prepare output tensor
        pos_embeddings = torch.zeros(batch_size, ctx_len, self.d_model, device=device)
        
        # Create the 4 coordinate transformations
        for batch_idx in range(batch_size):
            for seq_idx in range(ctx_len):
                i = i_coords[batch_idx, seq_idx].item()
                j = j_coords[batch_idx, seq_idx].item()
                
                # Transform 1: (i,j)
                pos1 = create_2d_positional_embedding(
                    int(i), int(j), self.d_positional_input, self.max_len
                ).to(device)
                
                # Transform 2: (H_max+H-1-i, j)
                i2 = self.H_max + H - 1 - i
                pos2 = create_2d_positional_embedding(
                    int(i2), int(j), self.d_positional_input, self.max_len
                ).to(device)
                
                # Transform 3: (H_max+H-1-i, W_max+W-1-j)
                j3 = self.W_max + W - 1 - j
                pos3 = create_2d_positional_embedding(
                    int(i2), int(j3), self.d_positional_input, self.max_len
                ).to(device)
                
                # Transform 4: (i, W_max+W-1-j)
                pos4 = create_2d_positional_embedding(
                    int(i), int(j3), self.d_positional_input, self.max_len
                ).to(device)
                
                # Concatenate 4 positional embeddings to create d_model tensor
                combined_pos = torch.cat([pos1, pos2, pos3, pos4], dim=0)
                pos_embeddings[batch_idx, seq_idx] = combined_pos
                
        return pos_embeddings
    
    def create_positional_embeddings_output(
        self, 
        positions: torch.Tensor, 
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Create positional embeddings for output positions using only (i,j) coordinates.
        
        This creates embeddings of size d_model (4*d_positional_input) but uses only 
        the single (i,j) coordinate without transformations, repeated 4 times.
        
        Args:
            positions: Tensor of shape [batch_size, ctx_len, 2] with (i,j) coordinates
            H: Height of current grid (included for API consistency, not used)
            W: Width of current grid (included for API consistency, not used)
            
        Returns:
            Tensor of shape [batch_size, ctx_len, d_model] with positional embeddings
        """
        batch_size, ctx_len, _ = positions.shape
        device = positions.device
        
        # Extract i,j coordinates
        i_coords = positions[:, :, 0]  # [batch_size, ctx_len]
        j_coords = positions[:, :, 1]  # [batch_size, ctx_len]
        
        # Prepare output tensor
        pos_embeddings = torch.zeros(batch_size, ctx_len, self.d_model, device=device)
        
        # Create embeddings using only (i,j) without transformations
        for batch_idx in range(batch_size):
            for seq_idx in range(ctx_len):
                i = i_coords[batch_idx, seq_idx].item()
                j = j_coords[batch_idx, seq_idx].item()
                
                # Create single positional embedding for (i,j)
                single_pos = create_2d_positional_embedding(
                    int(i), int(j), self.d_positional_input, self.max_len
                ).to(device)
                
                # Repeat the same embedding 4 times to match d_model size
                # This maintains API compatibility while using only (i,j)
                combined_pos = torch.cat([single_pos, single_pos, single_pos, single_pos], dim=0)
                pos_embeddings[batch_idx, seq_idx] = combined_pos
                
        return pos_embeddings
    
    def create_positional_embeddings_from_coordinates(
        self,
        coordinates_list: list[Coordinates],
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Create positional embeddings from a list of Coordinates objects.
        
        Args:
            coordinates_list: List of Coordinates objects
            H: Height of current grid
            W: Width of current grid
            
        Returns:
            Tensor of shape [1, len(coordinates_list), d_model] with combined positional embeddings
        """
        ctx_len = len(coordinates_list)
        device = next(self.parameters()).device
        
        # Prepare output tensor
        pos_embeddings = torch.zeros(1, ctx_len, self.d_model, device=device)
        
        # Create the 4 coordinate transformations
        for seq_idx, coord in enumerate(coordinates_list):
            i, j = coord.i, coord.j
            
            # Transform 1: (i,j)
            pos1 = create_2d_positional_embedding(
                int(i), int(j), self.d_positional_input, self.max_len
            ).to(device)
            
            # Transform 2: (H_max+H-1-i, j)
            i2 = self.H_max + H - 1 - i
            pos2 = create_2d_positional_embedding(
                int(i2), int(j), self.d_positional_input, self.max_len
            ).to(device)
            
            # Transform 3: (H_max+H-1-i, W_max+W-1-j)
            j3 = self.W_max + W - 1 - j
            pos3 = create_2d_positional_embedding(
                int(i2), int(j3), self.d_positional_input, self.max_len
            ).to(device)
            
            # Transform 4: (i, W_max+W-1-j)
            pos4 = create_2d_positional_embedding(
                int(i), int(j3), self.d_positional_input, self.max_len
            ).to(device)
            
            # Concatenate 4 positional embeddings to create d_model tensor
            combined_pos = torch.cat([pos1, pos2, pos3, pos4], dim=0)
            pos_embeddings[0, seq_idx] = combined_pos
            
        return pos_embeddings
    
    def forward(
        self, 
        colors: torch.Tensor, 
        positions: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Forward pass through encoding module.
        
        Args:
            colors: Tensor of shape [batch_size, ctx_len] with color values (0 to C-1)
            positions: Tensor of shape [batch_size, ctx_len, 2] with (i,j) coordinates
            H: Height of current grid
            W: Width of current grid
            
        Returns:
            Encoded tensor of shape [batch_size, ctx_len, d_model]
        """
        # Get tied color embeddings
        color_emb = self.color_embedding(colors)  # [batch_size, ctx_len, d_model]
        
        # Create 2D positional embeddings with 4 transformations
        pos_emb = self.create_positional_embeddings_input(positions, H, W)  # [batch_size, ctx_len, d_model]
        
        # Add color and positional embeddings
        embeddings = color_emb + pos_emb  # [batch_size, ctx_len, d_model]
        
        # Pass through custom transformer encoder layers
        x = embeddings
        for layer in self.layers:
            x = layer(x, positional_embedding=pos_emb)  # [batch_size, ctx_len, d_model]
        
        return x
    
    def forward_with_coordinates(
        self,
        colors: torch.Tensor,
        coordinates_list: list[Coordinates],
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Forward pass through encoding module using Coordinates objects.
        
        Args:
            colors: Tensor of shape [ctx_len] with color values (0 to C-1)
            coordinates_list: List of Coordinates objects with (i,j) positions
            H: Height of current grid
            W: Width of current grid
            
        Returns:
            Encoded tensor of shape [1, ctx_len, d_model]
        """
        # Ensure colors tensor has batch dimension
        if colors.dim() == 1:
            colors = colors.unsqueeze(0)  # [1, ctx_len]
        
        # Get tied color embeddings
        color_emb = self.color_embedding(colors)  # [1, ctx_len, d_model]
        
        # Create 2D positional embeddings with 4 transformations from Coordinates
        pos_emb = self.create_positional_embeddings_from_coordinates(
            coordinates_list, H, W
        )  # [1, ctx_len, d_model]
        
        # Add color and positional embeddings
        embeddings = color_emb + pos_emb  # [1, ctx_len, d_model]
        
        # Pass through custom transformer encoder layers
        x = embeddings
        for layer in self.layers:
            x = layer(x, positional_embedding=pos_emb)  # [1, ctx_len, d_model]
        
        return x