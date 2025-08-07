"""
Prediction Module for ARC spatial reasoning.

Implements the transformer specified in req.txt with:
- Tied embeddings for colors
- Three stages of EncoderTransformer layers with positional embedding manipulation
- Cross entropy loss function
"""

from typing import Optional, Tuple

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

from .classes import Coordinates
from .Encoding_Module import create_2d_positional_embedding


@gin.configurable
class PredictionModule(nn.Module):
    """
    Prediction Module implementing the architecture from req.txt:
    1. Tied color embeddings
    2. Add 4*d dim positional_embeddings_input
    3. N_1 layers of EncoderTransformer
    4. Subtract position_embeddings_input
    5. N_2 layers of EncoderTransformer
    6. Add 4*d dim positional_embeddings_output
    7. N_3 layers of EncoderTransformer
    8. Unembed using tied embeddings
    """

    @gin.configurable
    def __init__(
        self,
        d_model: int = gin.REQUIRED,
        n_layers_1: int = gin.REQUIRED,
        n_layers_2: int = gin.REQUIRED,
        n_layers_3: int = gin.REQUIRED,
        n_heads: int = gin.REQUIRED,
        d_ff: int = gin.REQUIRED,
        dropout: float = gin.REQUIRED,
        C: int = gin.REQUIRED,
        H_max: int = gin.REQUIRED,
        W_max: int = gin.REQUIRED,
        max_len: int = 10000,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        batch_first: bool = True
    ):
        """
        Initialize Prediction Module.
        
        Args:
            d_model: Model dimension (must be divisible by 4)
            n_layers_1: Number of transformer layers in first stage
            n_layers_2: Number of transformer layers in second stage
            n_layers_3: Number of transformer layers in third stage
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            C: Number of colors (0 to C-1)
            H_max: Maximum height for coordinate transformations
            W_max: Maximum width for coordinate transformations
            max_len: Maximum sequence length for positional embeddings
            activation: Activation function for feedforward network
            layer_norm_eps: Layer norm epsilon
            norm_first: Whether to apply layer norm before other operations
            batch_first: Whether batch dimension is first
        """
        super().__init__()

        assert d_model % 4 == 0, f"d_model must be divisible by 4, got {d_model}"

        self.d_model = d_model
        self.n_layers_1 = n_layers_1
        self.n_layers_2 = n_layers_2
        self.n_layers_3 = n_layers_3
        self.H_max = H_max
        self.W_max = W_max
        self.d_positional = d_model // 4
        self.max_len = max_len
        self.batch_first = batch_first

        # Tied embeddings for color
        self.color_embedding = nn.Embedding(C, d_model)

        # Create encoder layer template
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first
        )

        # Three stages of transformer encoder layers
        # Stage 1: With input positional embeddings
        self.encoder_stage_1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first
            ),
            num_layers=n_layers_1
        )

        # Stage 2: Without positional embeddings
        self.encoder_stage_2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first
            ),
            num_layers=n_layers_2
        )

        # Stage 3: With output positional embeddings
        self.encoder_stage_3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first
            ),
            num_layers=n_layers_3
        )

    @torch.no_grad()
    def create_positional_embeddings_input(
        self,
        positions: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Create 4 positional embeddings for input positions.
        
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
                    int(i), int(j), self.d_positional, self.max_len
                ).to(device)

                # Transform 2: (H_max+H-1-i, j)
                i2 = self.H_max + H - 1 - i
                pos2 = create_2d_positional_embedding(
                    int(i2), int(j), self.d_positional, self.max_len
                ).to(device)

                # Transform 3: (H_max+H-1-i, W_max+W-1-j)
                j3 = self.W_max + W - 1 - j
                pos3 = create_2d_positional_embedding(
                    int(i2), int(j3), self.d_positional, self.max_len
                ).to(device)

                # Transform 4: (i, W_max+W-1-j)
                pos4 = create_2d_positional_embedding(
                    int(i), int(j3), self.d_positional, self.max_len
                ).to(device)

                # Concatenate 4 positional embeddings to create d_model tensor
                combined_pos = torch.cat([pos1, pos2, pos3, pos4], dim=0)
                pos_embeddings[batch_idx, seq_idx] = combined_pos

        return pos_embeddings

    @torch.no_grad()
    def create_positional_embeddings_output(
        self,
        positions: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Create 4 positional embeddings for output positions.
        Same as input but conceptually for output coordinates.
        
        Args:
            positions: Tensor of shape [batch_size, ctx_len, 2] with (i,j) coordinates
            H: Height of current grid
            W: Width of current grid
            
        Returns:
            Tensor of shape [batch_size, ctx_len, d_model] with combined positional embeddings
        """
        # For now, using the same transformation as input
        # This could be modified if different output transformations are needed
        return self.create_positional_embeddings_input(positions, H, W)

    def unembed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert embeddings back to color logits using tied weights.
        
        Args:
            x: Tensor of shape [batch_size, ctx_len, d_model]
            
        Returns:
            Tensor of shape [batch_size, ctx_len, C] with color logits
        """
        # Use the transpose of embedding weights for output projection
        return F.linear(x, self.color_embedding.weight)

    def forward(
        self,
        input_colors: torch.Tensor,
        input_positions: torch.Tensor,
        output_positions: torch.Tensor,
        H: int,
        W: int,
        target_colors: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through prediction module.
        
        Args:
            input_colors: Tensor of shape [batch_size, ctx_len] with color values (0 to C-1)
            input_positions: Tensor of shape [batch_size, ctx_len, 2] with input (i,j) coordinates
            output_positions: Tensor of shape [batch_size, ctx_len, 2] with output (i,j) coordinates
            H: Height of current grid
            W: Width of current grid
            target_colors: Optional tensor of shape [batch_size, ctx_len] with target colors for loss
            
        Returns:
            Tuple of:
                - logits: Tensor of shape [batch_size, ctx_len, C] with color predictions
                - loss: Optional scalar tensor with cross entropy loss if target_colors provided
        """
        # Step 1: Get tied color embeddings
        color_emb = self.color_embedding(input_colors)  # [batch_size, ctx_len, d_model]

        # Step 2: Add 4*d dim positional_embeddings_input
        pos_emb_input = self.create_positional_embeddings_input(input_positions, H, W)
        x = color_emb + pos_emb_input

        # Step 3: N_1 layers of EncoderTransformer
        x = self.encoder_stage_1(x)

        # Step 4: Subtract position_embeddings_input
        x = x - pos_emb_input

        # Step 5: N_2 layers of EncoderTransformer
        x = self.encoder_stage_2(x)

        # Step 6: Add 4*d dim positional_embeddings_output
        pos_emb_output = self.create_positional_embeddings_output(output_positions, H, W)
        x = x + pos_emb_output

        # Step 7: N_3 layers of EncoderTransformer
        x = self.encoder_stage_3(x)

        # Step 8: Unembed using tied embeddings
        logits = self.unembed(x)  # [batch_size, ctx_len, C]

        # Calculate loss if targets provided
        loss = None
        if target_colors is not None:
            # Reshape for cross entropy loss
            batch_size, ctx_len, C = logits.shape
            logits_flat = logits.view(-1, C)
            targets_flat = target_colors.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def forward_with_coordinates(
        self,
        input_colors: torch.Tensor,
        input_coordinates: list[Coordinates],
        output_coordinates: list[Coordinates],
        H: int,
        W: int,
        target_colors: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using Coordinates objects.
        
        Args:
            input_colors: Tensor of shape [ctx_len] with color values (0 to C-1)
            input_coordinates: List of Coordinates objects for input positions
            output_coordinates: List of Coordinates objects for output positions
            H: Height of current grid
            W: Width of current grid
            target_colors: Optional tensor of shape [ctx_len] with target colors for loss
            
        Returns:
            Tuple of:
                - logits: Tensor of shape [1, ctx_len, C] with color predictions
                - loss: Optional scalar tensor with cross entropy loss if target_colors provided
        """
        device = next(self.parameters()).device

        # Ensure colors tensor has batch dimension
        if input_colors.dim() == 1:
            input_colors = input_colors.unsqueeze(0)  # [1, ctx_len]

        # Convert Coordinates to tensors
        ctx_len = len(input_coordinates)
        input_positions = torch.zeros(1, ctx_len, 2, device=device)
        output_positions = torch.zeros(1, ctx_len, 2, device=device)

        for idx, (in_coord, out_coord) in enumerate(zip(input_coordinates, output_coordinates, strict=False)):
            input_positions[0, idx, 0] = in_coord.i
            input_positions[0, idx, 1] = in_coord.j
            output_positions[0, idx, 0] = out_coord.i
            output_positions[0, idx, 1] = out_coord.j

        # Add batch dimension to target if needed
        if target_colors is not None and target_colors.dim() == 1:
            target_colors = target_colors.unsqueeze(0)  # [1, ctx_len]

        return self.forward(
            input_colors,
            input_positions,
            output_positions,
            H,
            W,
            target_colors
        )

