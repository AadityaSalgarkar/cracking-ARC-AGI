"""
Create puzzle for ARC spatial reasoning task.

Creates a 13x9 grid with random colors from 0 to 6, then converts it to 14x10 grid
by shifting all cells 1 position right and 1 position down, filling new cells with color 0.
"""

import numpy as np
import torch


def create_shift_puzzle(seed=None):
    """
    Create a spatial shift puzzle.

    Creates a 13x9 input grid with random colors 0-6, then produces a 14x10 output grid
    where all original cells are shifted 1 right and 1 down, with new boundary cells
    filled with color 0.

    Args:
        seed: Optional random seed for reproducibility

    Returns:
        tuple: (input_grid, output_grid) both as torch tensors
            - input_grid: (13, 9) tensor with values 0-6
            - output_grid: (14, 10) tensor with shifted values and 0-filled borders
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Create 13x9 input grid with random colors 0-6
    input_grid = torch.randint(0, 7, (13, 9), dtype=torch.long)

    # Create 14x10 output grid initialized with zeros
    output_grid = torch.zeros((14, 10), dtype=torch.long)

    # Shift all cells 1 right and 1 down (i.e., place input at [1:14, 1:10])
    output_grid[1:14, 1:10] = input_grid

    # The first row and first column remain 0 (already initialized)

    return input_grid, output_grid


def visualize_puzzle(input_grid, output_grid):
    """
    Print a visual representation of the puzzle.

    Args:
        input_grid: Input grid tensor
        output_grid: Output grid tensor
    """
    print("Input Grid (13x9):")
    print(input_grid.numpy())
    print("\nOutput Grid (14x10):")
    print(output_grid.numpy())
    print(f"\nInput grid shape: {input_grid.shape}")
    print(f"Output grid shape: {output_grid.shape}")
    print(f"Colors in input: {torch.unique(input_grid).tolist()}")
    print(f"Colors in output: {torch.unique(output_grid).tolist()}")


def save_puzzle(input_grid, output_grid, filename="puzzle.pt"):
    """
    Save puzzle grids to file.

    Args:
        input_grid: Input grid tensor
        output_grid: Output grid tensor
        filename: Output filename
    """
    puzzle_data = {
        "input": input_grid,
        "output": output_grid,
        "transformation": "shift_right_1_down_1_fill_0",
    }
    torch.save(puzzle_data, filename)
    print(f"Puzzle saved to {filename}")


def main():
    """Create and display a sample puzzle."""
    print("Creating spatial shift puzzle...")

    # Create puzzle with fixed seed for reproducibility
    input_grid, output_grid = create_shift_puzzle(seed=42)

    # Visualize the puzzle
    visualize_puzzle(input_grid, output_grid)

    # Save the puzzle
    save_puzzle(input_grid, output_grid, "puzzles/shift_puzzle.pt")

    # Verify the transformation
    print("\nVerification:")
    print(
        "Original cell [0,0] ->",
        input_grid[0, 0].item(),
        "moved to output [1,1] ->",
        output_grid[1, 1].item(),
    )
    print(
        "Original cell [12,8] ->",
        input_grid[12, 8].item(),
        "moved to output [13,9] ->",
        output_grid[13, 9].item(),
    )
    print("New border cells filled with 0:")
    print(f"Top row: {output_grid[0, :].tolist()}")
    print(f"Left column: {output_grid[:, 0].tolist()}")


if __name__ == "__main__":
    main()
