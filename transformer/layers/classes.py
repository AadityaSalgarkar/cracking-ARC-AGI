from dataclasses import dataclass
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import torch

from .positional_embeddings import create_sinusoidal_embedding


@dataclass
class Cell:
    """
    Input class containing color and 4 position coordinate transformations.

    As specified in req.txt:
    - Color: int, lies in 0 to C-1
    - Position_1: (i,j)
    - Position_2: (H_max+H-1-i, j)
    - Position_3: (H_max+H-1-i, W_max+W-1-j)
    - Position_4: (i, W_max+W-1-j)
    """

    Color: int
    Position_1: Tuple[int, int]  # (i,j)
    Position_2: Tuple[int, int]  # (H_max+H-1-i, j)
    Position_3: Tuple[int, int]  # (H_max+H-1-i, W_max+W-1-j)
    Position_4: Tuple[int, int]  # (i, W_max+W-1-j)

    # Cache for position embeddings to avoid recomputation in attention
    _cached_position_embedding: torch.Tensor = None
    _cache_params: Tuple[int, int, int] = (
        None  # (d_model, rope_base, sinusoidal_max_len)
    )

    def __post_init__(self):
        """Validate the input values."""
        # Validate color is non-negative integer
        if not isinstance(self.Color, int) or self.Color < 0:
            raise ValueError(f"Color must be non-negative integer, got {self.Color}")

        # Validate positions are tuples of two integers
        for i, pos in enumerate(
            [self.Position_1, self.Position_2, self.Position_3, self.Position_4], 1
        ):
            if not isinstance(pos, tuple) or len(pos) != 2:
                raise ValueError(f"Position_{i} must be tuple of length 2, got {pos}")
            if not all(isinstance(x, int) for x in pos):
                raise ValueError(f"Position_{i} must contain integers, got {pos}")

    @classmethod
    def from_grid_position(
        cls,
        color: int,
        i: int,
        j: int,
        H: int,
        W: int,
        H_max: int = 30,
        W_max: int = 30,
    ) -> "Cell":
        """
        Create Input instance from grid position and transformations.

        Args:
            color: Color value (0 to C-1)
            i: Row coordinate in original grid
            j: Column coordinate in original grid
            H: Height of original grid
            W: Width of original grid
            H_max: Maximum height for coordinate transformations
            W_max: Maximum width for coordinate transformations

        Returns:
            Input instance with computed coordinate transformations
        """
        # Validate inputs
        if not (0 <= i < H):
            raise ValueError(f"i={i} must be in range [0, {H})")
        if not (0 <= j < W):
            raise ValueError(f"j={j} must be in range [0, {W})")
        if H > H_max:
            raise ValueError(f"H={H} must be <= H_max={H_max}")
        if W > W_max:
            raise ValueError(f"W={W} must be <= W_max={W_max}")

        # Compute the 4 coordinate transformations as specified in req.txt
        position_1 = (i, j)
        position_2 = (H_max + (H - 1 - i), j)
        position_3 = (H_max + (H - 1 - i), W_max + (W - 1 - j))
        position_4 = (i, W_max + (W - 1 - j))

        return cls(
            Color=color,
            Position_1=position_1,
            Position_2=position_2,
            Position_3=position_3,
            Position_4=position_4,
        )

    def get_color_embedding(self, d_model: int, max_len: int = 10000) -> torch.Tensor:
        """Get sinusoidal embedding for the color value."""
        return create_sinusoidal_embedding(self.Color, d_model, max_len)

    def get_position_embeddings(
        self, d_model: int, base: int = 10000
    ) -> Tuple[torch.Tensor, ...]:
        """
        Get simple positional embeddings for all 4 positions.

        Args:
            d_model: Embedding dimension
            base: Base for frequency computation

        Returns:
            Tuple of 4 tensors, each of shape (d_model,) for the 4 positions
        """

        def create_pos_emb(pos_tuple, d_model, base):
            """Create simple positional embedding from coordinate tuple."""
            i, j = pos_tuple
            pos_val = i * 1000 + j  # Simple encoding of 2D position
            return create_sinusoidal_embedding(pos_val, d_model, base)

        pos1_emb = create_pos_emb(self.Position_1, d_model, base)
        pos2_emb = create_pos_emb(self.Position_2, d_model, base)
        pos3_emb = create_pos_emb(self.Position_3, d_model, base)
        pos4_emb = create_pos_emb(self.Position_4, d_model, base)

        return pos1_emb, pos2_emb, pos3_emb, pos4_emb

    def get_combined_embedding(
        self, d_model: int, rope_base: int = 10000, sinusoidal_max_len: int = 10000
    ) -> torch.Tensor:
        """
        Get combined embedding with color and all 4 position embeddings.

        Based on updated req.txt: color_embedding + position_i_embedding for i in range(1,5)
        and stack them vertically to create 4*d_model tensor.

        Args:
            d_model: Embedding dimension for each component
            rope_base: Base for RoPE frequency computation
            sinusoidal_max_len: Max length for sinusoidal color embedding

        Returns:
            Combined embedding tensor of shape (4*d_model,) containing:
            [color_emb + pos1_emb, color_emb + pos2_emb, color_emb + pos3_emb, color_emb + pos4_emb]
            stacked vertically into a single flattened vector.
        """
        # Get color embedding
        color_emb = self.get_color_embedding(d_model, sinusoidal_max_len)

        # Get position embeddings
        pos_embs = self.get_position_embeddings(d_model, rope_base)

        # Create combined embeddings: color_embedding + position_i_embedding for each position
        combined_embeddings = []
        for pos_emb in pos_embs:
            combined_emb = color_emb + pos_emb  # Element-wise addition
            combined_embeddings.append(combined_emb)

        # Stack vertically to get shape (4, d_model), then flatten to (4*d_model,)
        combined_stacked = torch.stack(combined_embeddings, dim=0)  # (4, d_model)
        combined_flattened = combined_stacked.flatten()  # (4*d_model,)

        return combined_flattened

    def get_position_embedding_for_attention(
        self, d_model: int, rope_base: int = 10000, sinusoidal_max_len: int = 10000
    ) -> torch.Tensor:
        """
        Get cached position embedding tensor of shape (4*d_model,) for attention module reuse.

        This method caches the flattened position embedding tensor to avoid recomputation
        when the same parameters are used multiple times (e.g., in attention mechanisms).

        Args:
            d_model: Embedding dimension for each component
            rope_base: Base for RoPE frequency computation
            sinusoidal_max_len: Max length for sinusoidal embedding

        Returns:
            Cached position embedding tensor of shape (4*d_model,) containing the 4 position
            embeddings concatenated together: [pos1_emb, pos2_emb, pos3_emb, pos4_emb]
        """
        current_params = (d_model, rope_base, sinusoidal_max_len)

        # Check if we have a valid cache
        if (
            self._cached_position_embedding is not None
            and self._cache_params == current_params
        ):
            return self._cached_position_embedding

        # Compute position embeddings
        pos_embs = self.get_position_embeddings(d_model, rope_base)

        # Concatenate the 4 position embeddings into a single tensor
        position_embedding = torch.cat(pos_embs, dim=0)  # Shape: (4*d_model,)

        # Cache the result
        self._cached_position_embedding = position_embedding
        self._cache_params = current_params

        return position_embedding

    def get_stacked_embedding(
        self, d_model: int, rope_base: int = 10000, sinusoidal_max_len: int = 10000
    ) -> torch.Tensor:
        """
        Get stacked embedding (non-flattened version for compatibility).

        Args:
            d_model: Embedding dimension for each component
            rope_base: Base for RoPE frequency computation
            sinusoidal_max_len: Max length for sinusoidal color embedding

        Returns:
            Stacked embedding tensor of shape (4, d_model) containing:
            [color_emb + pos1_emb, color_emb + pos2_emb, color_emb + pos3_emb, color_emb + pos4_emb]
            Each row is element-wise addition of color embedding with position embedding.
        """
        # Get color embedding
        color_emb = self.get_color_embedding(d_model, sinusoidal_max_len)

        # Get position embeddings
        pos_embs = self.get_position_embeddings(d_model, rope_base)

        # Create combined embeddings: color_embedding + position_i_embedding for each position
        combined_embeddings = []
        for pos_emb in pos_embs:
            combined_emb = color_emb + pos_emb  # Element-wise addition
            combined_embeddings.append(combined_emb)

        # Stack to get final shape (4, d_model)
        combined = torch.stack(combined_embeddings, dim=0)

        return combined


@dataclass
class PuzzleShape:
    """
    Puzzle shape containing H_max * W_max Cell objects representing a full grid.

    Creates a grid of Cell objects with proper coordinate transformations
    and provides visualization functionality.
    """

    cells: List[Cell]
    H_max: int = 30
    W_max: int = 30

    def __post_init__(self):
        """Validate that we have the correct number of cells."""
        expected_size = self.H_max * self.W_max
        if len(self.cells) != expected_size:
            raise ValueError(f"Expected {expected_size} cells, got {len(self.cells)}")

    @classmethod
    def from_grid(
        cls, grid: torch.Tensor, H_max: int = 30, W_max: int = 30, C: int = 11
    ) -> "PuzzleShape":
        """
        Create PuzzleShape from a grid tensor.

        Args:
            grid: (H, W) tensor with color values 0 to C-1
            H_max: Maximum height for coordinate transformations
            W_max: Maximum width for coordinate transformations
            C: Number of colors (0 to C-1), cells outside grid get C-1

        Returns:
            PuzzleShape instance with Cell objects for the full H_max x W_max grid
        """
        H, W = grid.shape
        cells = []

        # Create H_max * W_max cells
        for i in range(H_max):
            for j in range(W_max):
                if i < H and j < W:
                    # Cell within the actual grid
                    color = grid[i, j].item()
                else:
                    # Cell outside the grid - use color C-1
                    color = C - 1

                # Create Cell with coordinate transformations
                cell = Cell.from_grid_position(
                    color=color,
                    i=i,
                    j=j,
                    H=H_max,  # Use H_max as the grid size for transformations
                    W=W_max,  # Use W_max as the grid size for transformations
                    H_max=H_max,
                    W_max=W_max,
                )
                cells.append(cell)

        return cls(cells=cells, H_max=H_max, W_max=W_max)

    @classmethod
    def from_random(
        cls,
        H: int = 13,
        W: int = 9,
        C: int = 7,
        H_max: int = 30,
        W_max: int = 30,
        seed: int = None,
    ) -> "PuzzleShape":
        """
        Create PuzzleShape with random colors in H x W region, rest filled with C-1.

        Args:
            H: Height of the random region
            W: Width of the random region
            C: Number of colors (0 to C-1)
            H_max: Maximum height for coordinate transformations
            W_max: Maximum width for coordinate transformations
            seed: Random seed for reproducibility

        Returns:
            PuzzleShape instance with random colors in H x W region
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Create random grid for H x W region
        random_grid = torch.randint(0, C, (H, W), dtype=torch.long)

        # Create full H_max x W_max grid filled with C-1 (outside grid color)
        full_grid = torch.full((H_max, W_max), C - 1, dtype=torch.long)
        full_grid[:H, :W] = random_grid

        return cls.from_grid(full_grid, H_max, W_max, C)

    def get_cell(self, i: int, j: int) -> Cell:
        """Get cell at position (i, j)."""
        if not (0 <= i < self.H_max and 0 <= j < self.W_max):
            raise ValueError(
                f"Position ({i}, {j}) out of bounds for {self.H_max}x{self.W_max} grid"
            )

        index = i * self.W_max + j
        return self.cells[index]

    def get_grid(self) -> torch.Tensor:
        """Get the color grid as H_max x W_max tensor."""
        grid = torch.zeros((self.H_max, self.W_max), dtype=torch.long)

        for i in range(self.H_max):
            for j in range(self.W_max):
                cell = self.get_cell(i, j)
                grid[i, j] = cell.Color

        return grid

    def visualize(self, title: str = "InputPuzzle", cell_size: float = 0.5):
        """
        Create and display H_max x W_max sized grid with perfect square cells.
        Each cell is exactly square, creating a proper H_max × W_max grid.

        Args:
            title: Title for the plot (not displayed, kept for API compatibility)
            cell_size: Size of each cell in inches (default 0.5)
        """
        # Note: title parameter kept for API compatibility but not used in clean visualization
        grid = self.get_grid().numpy()

        # Calculate figure size to make each cell exactly square
        # Width = W_max * cell_size, Height = H_max * cell_size
        figsize = (self.W_max * cell_size, self.H_max * cell_size)

        # Create color map - standard ARC colors
        arc_colors = [
            "#000000",  # 0: black
            "#0074D9",  # 1: blue
            "#FF4136",  # 2: red
            "#2ECC40",  # 3: green
            "#FFDC00",  # 4: yellow
            "#AAAAAA",  # 5: gray
            "#F012BE",  # 6: magenta
            "#FF851B",  # 7: orange
            "#7FDBFF",  # 8: aqua
            "#870C25",  # 9: brown
            "#FFFFFF",  # 10: white
        ]

        # Extend colors if needed
        max_color = grid.max()
        while len(arc_colors) <= max_color:
            arc_colors.append("#808080")  # Default gray for extra colors

        # Create figure with exact dimensions
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=100)

        # Remove all margins and padding to use full figure area
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Create custom colormap from ARC colors
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(arc_colors[: max_color + 1])

        # Display grid with perfect cell mapping
        # Use extent to map grid to exact pixel coordinates
        ax.imshow(
            grid,
            cmap=cmap,
            vmin=0,
            vmax=max_color,
            interpolation="nearest",
            aspect="equal",
            extent=[-0.5, self.W_max - 0.5, self.H_max - 0.5, -0.5],
        )

        # Set exact limits to show only the grid
        ax.set_xlim(-0.5, self.W_max - 0.5)
        ax.set_ylim(self.H_max - 0.5, -0.5)

        # Remove all axes, ticks, labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis("off")

        # Remove any borders or spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add 30x30 grid lines to aid visualization
        # Vertical lines
        for i in range(self.W_max + 1):
            ax.axvline(x=i - 0.5, color="gray", linewidth=0.5, alpha=0.3)

        # Horizontal lines
        for i in range(self.H_max + 1):
            ax.axhline(y=i - 0.5, color="gray", linewidth=0.5, alpha=0.3)

        plt.show()

    def __repr__(self) -> str:
        """String representation of the puzzle."""
        grid = self.get_grid()
        unique_colors = torch.unique(grid).tolist()
        return f"PuzzleShape({self.H_max}x{self.W_max}, colors={unique_colors})"


@dataclass
class Puzzle:
    """
    Complete puzzle with input and output PuzzleShape objects.

    Contains both the input grid and expected output grid for ARC-style puzzles.
    """

    input: PuzzleShape
    output: PuzzleShape

    def __post_init__(self):
        """Validate that input and output have same dimensions."""
        if (
            self.input.H_max != self.output.H_max
            or self.input.W_max != self.output.W_max
        ):
            raise ValueError(
                f"Input ({self.input.H_max}x{self.input.W_max}) and output "
                f"({self.output.H_max}x{self.output.W_max}) must have same dimensions"
            )

    @classmethod
    def from_grids(
        cls,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor,
        H_max: int = 30,
        W_max: int = 30,
        C: int = 11,
    ) -> "Puzzle":
        """
        Create Puzzle from input and output grid tensors.

        Args:
            input_grid: Input grid tensor
            output_grid: Output grid tensor
            H_max: Maximum height for both grids
            W_max: Maximum width for both grids
            C: Number of colors (0 to C-1)

        Returns:
            Puzzle instance with input and output PuzzleShape objects
        """
        input_shape = PuzzleShape.from_grid(input_grid, H_max, W_max, C)
        output_shape = PuzzleShape.from_grid(output_grid, H_max, W_max, C)

        return cls(input=input_shape, output=output_shape)

    def visualize(self, cell_size: float = 0.25, side_by_side: bool = True):
        """
        Visualize both input and output puzzles.

        Args:
            cell_size: Size of each cell in inches
            side_by_side: If True, show input and output side by side
        """
        if side_by_side:
            # Create side-by-side visualization
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(
                1,
                2,
                figsize=(
                    2 * self.input.W_max * cell_size,
                    self.input.H_max * cell_size,
                ),
            )

            # Get grids
            input_grid = self.input.get_grid().numpy()
            output_grid = self.output.get_grid().numpy()

            # Create color map
            arc_colors = [
                "#000000",
                "#0074D9",
                "#FF4136",
                "#2ECC40",
                "#FFDC00",
                "#AAAAAA",
                "#F012BE",
                "#FF851B",
                "#7FDBFF",
                "#870C25",
                "#FFFFFF",
            ]
            max_color = max(input_grid.max(), output_grid.max())
            while len(arc_colors) <= max_color:
                arc_colors.append("#808080")

            from matplotlib.colors import ListedColormap

            cmap = ListedColormap(arc_colors[: max_color + 1])

            # Plot input
            ax1.imshow(
                input_grid,
                cmap=cmap,
                vmin=0,
                vmax=max_color,
                interpolation="nearest",
                aspect="equal",
                extent=[-0.5, self.input.W_max - 0.5, self.input.H_max - 0.5, -0.5],
            )
            ax1.set_title("Input")

            # Add grid lines for input
            for i in range(self.input.W_max + 1):
                ax1.axvline(x=i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
            for i in range(self.input.H_max + 1):
                ax1.axhline(y=i - 0.5, color="gray", linewidth=0.5, alpha=0.3)

            ax1.set_xlim(-0.5, self.input.W_max - 0.5)
            ax1.set_ylim(self.input.H_max - 0.5, -0.5)
            ax1.set_xticks([])
            ax1.set_yticks([])

            # Plot output
            ax2.imshow(
                output_grid,
                cmap=cmap,
                vmin=0,
                vmax=max_color,
                interpolation="nearest",
                aspect="equal",
                extent=[-0.5, self.output.W_max - 0.5, self.output.H_max - 0.5, -0.5],
            )
            ax2.set_title("Output")

            # Add grid lines for output
            for i in range(self.output.W_max + 1):
                ax2.axvline(x=i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
            for i in range(self.output.H_max + 1):
                ax2.axhline(y=i - 0.5, color="gray", linewidth=0.5, alpha=0.3)

            ax2.set_xlim(-0.5, self.output.W_max - 0.5)
            ax2.set_ylim(self.output.H_max - 0.5, -0.5)
            ax2.set_xticks([])
            ax2.set_yticks([])

            plt.tight_layout()
            plt.show()
        else:
            # Show sequentially
            print("Input:")
            self.input.visualize(cell_size=cell_size)
            print("Output:")
            self.output.visualize(cell_size=cell_size)

    def __repr__(self) -> str:
        """String representation of the puzzle."""
        return f"Puzzle(input={self.input}, output={self.output})"


@dataclass
class TransformerIO:
    """
    TransformerIO class for creating input/output sequences from puzzle data.

    Creates sequences of Cells with:
    - Input: n_seq cells chosen by heuristic, using 4 different RoPE position types
    - Output: n_seq cells chosen by heuristic, using (i,j) type repeated 4 times
    - Ground truth: Optional sequence for training/validation loss computation
    """

    input_sequence: List[Cell]
    output_sequence: List[Cell]
    n_seq: int
    ground_truth: Optional[List[Cell]] = None

    def __post_init__(self):
        """Validate sequence lengths."""
        if len(self.input_sequence) != self.n_seq:
            raise ValueError(
                f"Input sequence length {len(self.input_sequence)} != n_seq {self.n_seq}"
            )
        if len(self.output_sequence) != self.n_seq:
            raise ValueError(
                f"Output sequence length {len(self.output_sequence)} != n_seq {self.n_seq}"
            )
        if self.ground_truth is not None and len(self.ground_truth) != self.n_seq:
            raise ValueError(
                f"Ground truth sequence length {len(self.ground_truth)} != n_seq {self.n_seq}"
            )

    @classmethod
    def from_puzzle(
        cls,
        puzzle: Puzzle,
        n_seq: int,
        input_heuristic: str = "random",
        output_heuristic: str = "random",
        include_ground_truth: bool = False,
        seed: int = None,
    ) -> "TransformerIO":
        """
        Create TransformerIO from a Puzzle using specified heuristics.

        Args:
            puzzle: Puzzle object with input and output PuzzleShapes
            n_seq: Length of both input and output sequences
            input_heuristic: Heuristic for selecting input cells ("random")
            output_heuristic: Heuristic for selecting output cells ("random")
            include_ground_truth: Whether to include ground truth sequence (for train/val)
            seed: Random seed for reproducibility

        Returns:
            TransformerIO instance with selected sequences and optional ground truth
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Get all available cells from input and output
        input_cells = puzzle.input.cells
        output_cells = puzzle.output.cells

        # Apply heuristics to select cells
        if input_heuristic == "random":
            input_indices = torch.randperm(len(input_cells))[:n_seq]
            selected_input_cells = [input_cells[i] for i in input_indices]
        else:
            raise NotImplementedError(
                f"Input heuristic '{input_heuristic}' not implemented"
            )

        if output_heuristic == "random":
            output_indices = torch.randperm(len(output_cells))[:n_seq]
            selected_output_cells = [output_cells[i] for i in output_indices]
        else:
            raise NotImplementedError(
                f"Output heuristic '{output_heuristic}' not implemented"
            )

        # Modify output cells to have (i,j) type repeated 4 times
        modified_output_cells = []
        for cell in selected_output_cells:
            # Create new cell with Position_1 (i,j) repeated for all 4 positions
            modified_cell = Cell(
                Color=cell.Color,
                Position_1=cell.Position_1,  # (i,j)
                Position_2=cell.Position_1,  # (i,j) repeated
                Position_3=cell.Position_1,  # (i,j) repeated
                Position_4=cell.Position_1,  # (i,j) repeated
            )
            modified_output_cells.append(modified_cell)

        # Create ground truth sequence if requested
        ground_truth = None
        if include_ground_truth:
            # Ground truth is the expected output colors/positions for the input sequence
            # Use the same cells as output but maintain original positions for ground truth
            ground_truth = (
                selected_output_cells  # Keep original output cells without modification
            )

        return cls(
            input_sequence=selected_input_cells,
            output_sequence=modified_output_cells,
            n_seq=n_seq,
            ground_truth=ground_truth,
        )

    def get_input_embeddings(self, embed_dim: int) -> torch.Tensor:
        """
        Get embeddings for input sequence using 4 different RoPE types.

        Args:
            embed_dim: Embedding dimension

        Returns:
            Tensor of shape (n_seq, 4*embed_dim) with input embeddings
        """
        embeddings = []
        for cell in self.input_sequence:
            # Get color embedding
            color_emb = cell.get_color_embedding(embed_dim)

            # Get all 4 position embeddings (different RoPE types)
            pos_embs = cell.get_position_embeddings(embed_dim)

            # Combine color with each position embedding
            combined_embs = []
            for pos_emb in pos_embs:
                combined_emb = color_emb + pos_emb
                combined_embs.append(combined_emb)

            # Concatenate all 4 combined embeddings to get 4*embed_dim
            full_emb = torch.cat(combined_embs, dim=0)  # Shape: (4*embed_dim,)
            embeddings.append(full_emb)

        return torch.stack(embeddings, dim=0)  # Shape: (n_seq, 4*embed_dim)

    def get_output_embeddings(self, embed_dim: int) -> torch.Tensor:
        """
        Get embeddings for output sequence using (i,j) type repeated 4 times.

        Args:
            embed_dim: Embedding dimension

        Returns:
            Tensor of shape (n_seq, embed_dim) with output embeddings
        """
        embeddings = []
        for cell in self.output_sequence:
            # Get color embedding
            color_emb = cell.get_color_embedding(embed_dim)

            # Get position embeddings (all should be the same (i,j) type)
            pos_embs = cell.get_position_embeddings(embed_dim)

            # Since all 4 positions are the same (i,j), just use the first one
            pos_emb = pos_embs[0]
            combined_emb = color_emb + pos_emb
            embeddings.append(combined_emb)

        return torch.stack(embeddings, dim=0)

    def get_input_colors(self) -> torch.Tensor:
        """Get input sequence colors as tensor."""
        colors = [cell.Color for cell in self.input_sequence]
        return torch.tensor(colors, dtype=torch.long)

    def get_output_colors(self) -> torch.Tensor:
        """Get output sequence colors as tensor."""
        colors = [cell.Color for cell in self.output_sequence]
        return torch.tensor(colors, dtype=torch.long)

    def get_output_positions(self) -> torch.Tensor:
        """Get output sequence positions as tensor of (i,j) coordinates."""
        positions = [
            cell.Position_1 for cell in self.output_sequence
        ]  # All positions are the same (i,j)
        return torch.tensor(positions, dtype=torch.float)

    def get_ground_truth_colors(self) -> Optional[torch.Tensor]:
        """Get ground truth sequence colors as tensor."""
        if self.ground_truth is None:
            return None
        colors = [cell.Color for cell in self.ground_truth]
        return torch.tensor(colors, dtype=torch.long)

    def get_ground_truth_positions(self) -> Optional[torch.Tensor]:
        """Get ground truth sequence positions as tensor of (i,j) coordinates."""
        if self.ground_truth is None:
            return None
        positions = [cell.Position_1 for cell in self.ground_truth]
        return torch.tensor(positions, dtype=torch.float)

    def has_ground_truth(self) -> bool:
        """Check if ground truth is available."""
        return self.ground_truth is not None

    def __repr__(self) -> str:
        """String representation."""
        gt_info = (
            f", has_gt={self.has_ground_truth()}"
            if self.ground_truth is not None
            else ""
        )
        return f"TransformerIO(n_seq={self.n_seq}, input_colors={len(set(c.Color for c in self.input_sequence))}, output_colors={len(set(c.Color for c in self.output_sequence))}{gt_info})"
