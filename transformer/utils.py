import gin
import matplotlib.pyplot as plt
import torch


@gin.configurable
def create_coordinate_tensor(H, W, H_max=30, W_max=30):
    """
    Create 8 x H_max x W_max tensor with coordinate transformations for given (H,W).

    Args:
        H: Height of the input grid
        W: Width of the input grid
        H_max: Maximum height dimension
        W_max: Maximum width dimension

    Returns:
        coords: (8, H_max, W_max) tensor with coordinate transformations:
            - Channel 0,1: (i,j)
            - Channel 2,3: (H_max + (H-1-i), j)
            - Channel 4,5: (H_max + (H-1-i), W_max + (W-1-j))
            - Channel 6,7: (i, W_max + (W-1-j))
    """
    assert H <= H_max, f"Grid height {H} exceeds H_max {H_max}"
    assert W <= W_max, f"Grid width {W} exceeds W_max {W_max}"

    # Initialize output tensor with zeros
    coords = torch.zeros((8, H_max, W_max), dtype=torch.long)

    # Create coordinate grids for the full H_max x W_max space
    i_full = torch.arange(H_max).unsqueeze(1).expand(H_max, W_max)
    j_full = torch.arange(W_max).unsqueeze(0).expand(H_max, W_max)

    # The 4 RoPE rotational coordinate transformations across full space:

    # Channels 0,1: (i,j)
    coords[0] = i_full
    coords[1] = j_full

    # Channels 2,3: (H_max + (H-1-i), j) - but only for active region
    coords[2] = i_full  # Default to identity
    coords[3] = j_full
    coords[2, :H, :W] = H_max + (H - 1 - i_full[:H, :W])  # Transform active region

    # Channels 4,5: (H_max + (H-1-i), W_max + (W-1-j)) - but only for active region
    coords[4] = i_full  # Default to identity
    coords[5] = j_full  # Default to identity
    coords[4, :H, :W] = H_max + (H - 1 - i_full[:H, :W])  # Transform active region
    coords[5, :H, :W] = W_max + (W - 1 - j_full[:H, :W])  # Transform active region

    # Channels 6,7: (i, W_max + (W-1-j)) - but only for active region
    coords[6] = i_full  # Default to identity
    coords[7] = j_full  # Default to identity
    coords[7, :H, :W] = W_max + (W - 1 - j_full[:H, :W])  # Transform active region

    return coords


@gin.configurable
def transform_grid_with_coordinates(grid, H_max=30, W_max=30, C=11):
    """
    Transform HxW grid to H_max x W_max x 9 tensor with coordinate information.

    Args:
        grid: (H, W) tensor with values 0 to C-1
        H_max: Maximum height dimension
        W_max: Maximum width dimension
        C: Number of classes (values 0 to C-1)

    Returns:
        output: (H_max, W_max, 9) tensor where:
            - Channel 0: Original grid values in top-left, rest filled with C-1
            - Channel 1: i coordinates (row indices)
            - Channel 2: j coordinates (column indices)
            - Channel 3: H_max + (H-1-i) coordinates
            - Channel 4: j coordinates (same as channel 2)
            - Channel 5: H_max + (H-1-i) coordinates (same as channel 3)
            - Channel 6: W_max + (W-1-j) coordinates
            - Channel 7: i coordinates (same as channel 1)
            - Channel 8: W_max + (W-1-j) coordinates (same as channel 6)

    The 4 RoPE rotational coordinate transformations are:
        - Channels 1,2: (i,j)
        - Channels 3,4: (H_max + (H-1-i), j)
        - Channels 5,6: (H_max + (H-1-i), W_max + (W-1-j))
        - Channels 7,8: (i, W_max + (W-1-j))
    """
    H, W = grid.shape

    # Ensure grid fits in H_max x W_max
    assert H <= H_max, f"Grid height {H} exceeds H_max {H_max}"
    assert W <= W_max, f"Grid width {W} exceeds W_max {W_max}"

    # Initialize output tensor
    output = torch.full((H_max, W_max, 9), C - 1, dtype=grid.dtype, device=grid.device)

    # Channel 0: Original grid values in top-left corner
    output[:H, :W, 0] = grid

    # Create coordinate grids for the original H x W region
    i_coords = (
        torch.arange(H, dtype=grid.dtype, device=grid.device).unsqueeze(1).expand(H, W)
    )
    j_coords = (
        torch.arange(W, dtype=grid.dtype, device=grid.device).unsqueeze(0).expand(H, W)
    )

    # The 4 RoPE rotational coordinate transformations:
    # (i,j), (H_max + (H-1-i), j), (H_max + (H-1-i), W_max + (W-1-j)), (i, W_max + (W-1-j))

    # Channels 1,2: (i,j)
    output[:H, :W, 1] = i_coords
    output[:H, :W, 2] = j_coords

    # Channels 3,4: (H_max + (H-1-i), j)
    output[:H, :W, 3] = H_max + (H - 1 - i_coords)
    output[:H, :W, 4] = j_coords

    # Channels 5,6: (H_max + (H-1-i), W_max + (W-1-j))
    output[:H, :W, 5] = H_max + (H - 1 - i_coords)
    output[:H, :W, 6] = W_max + (W - 1 - j_coords)

    # Channels 7,8: (i, W_max + (W-1-j))
    output[:H, :W, 7] = i_coords
    output[:H, :W, 8] = W_max + (W - 1 - j_coords)

    return output


@gin.configurable
def create_augmented_dataset(grids, H_max=30, W_max=30, C=11):
    """
    Create augmented dataset with coordinate information for a batch of grids.

    Args:
        grids: List of (H_i, W_i) tensors or (batch_size, H, W) tensor
        H_max, W_max, C: Same as transform_grid_with_coordinates

    Returns:
        augmented: (len(grids), H_max, W_max, 9) tensor
    """
    if isinstance(grids, torch.Tensor) and grids.dim() == 3:
        # Handle batch tensor
        batch_size = grids.size(0)
        results = []
        for i in range(batch_size):
            result = transform_grid_with_coordinates(grids[i], H_max, W_max, C)
            results.append(result)
        return torch.stack(results)
    else:
        # Handle list of tensors
        if len(grids) == 0:
            # Return empty tensor with correct shape
            return torch.empty((0, H_max, W_max, 9))

        results = []
        for grid in grids:
            result = transform_grid_with_coordinates(grid, H_max, W_max, C)
            results.append(result)
        return torch.stack(results)


def visualize_coordinate_tensor(H, W, H_max=30, W_max=30, save_path=None):
    """
    Visualize the 8 coordinate transformation channels as 2D plots.

    Args:
        H: Height of the input grid
        W: Width of the input grid
        H_max: Maximum height dimension
        W_max: Maximum width dimension
        save_path: Path to save the figure (optional)
    """
    # Create the coordinate tensor
    coords = create_coordinate_tensor(H, W, H_max, W_max)

    # Channel names for the 4 coordinate transformations (i,j pairs)
    channel_names = [
        "Channel 0: i (identity)",
        "Channel 1: j (identity)",
        "Channel 2: H_max + (H-1-i)",
        "Channel 3: j (for transform 2)",
        "Channel 4: H_max + (H-1-i)",
        "Channel 5: W_max + (W-1-j)",
        "Channel 6: i (for transform 4)",
        "Channel 7: W_max + (W-1-j)",
    ]

    # Create subplot layout: 2 rows, 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"Coordinate Transformations for H={H}, W={W} (H_max={H_max}, W_max={W_max})",
        fontsize=16,
    )

    # Plot each channel
    for ch in range(8):
        row = ch // 4
        col = ch % 4
        ax = axes[row, col]

        # Convert to numpy for plotting
        data = coords[ch].numpy()

        # Create the plot with color mapping
        im = ax.imshow(data, cmap="viridis", aspect="equal")
        ax.set_title(channel_names[ch], fontsize=10)
        ax.set_xlabel("W dimension")
        ax.set_ylabel("H dimension")

        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Highlight the active region (H x W) with a rectangle
        if H < H_max or W < W_max:
            from matplotlib.patches import Rectangle

            rect = Rectangle(
                (0 - 0.5, 0 - 0.5), W, H, linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)

        # Set ticks to show grid structure
        ax.set_xticks(range(0, W_max, max(1, W_max // 10)))
        ax.set_yticks(range(0, H_max, max(1, H_max // 10)))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    # Print some sample values for verification
    print(f"\nSample coordinate values for active region (H={H}, W={W}):")
    print("Position (0,0):")
    for ch in range(8):
        print(f"  Channel {ch}: {coords[ch, 0, 0].item()}")

    if H > 1 and W > 1:
        print(f"Position ({H - 1},{W - 1}):")
        for ch in range(8):
            print(f"  Channel {ch}: {coords[ch, H - 1, W - 1].item()}")

    # Show the 4 coordinate transformations grouped
    print("\nThe 4 RoPE coordinate transformations:")
    print("1. (i,j): Channels 0,1")
    print("2. (H_max + (H-1-i), j): Channels 2,3")
    print("3. (H_max + (H-1-i), W_max + (W-1-j)): Channels 4,5")
    print("4. (i, W_max + (W-1-j)): Channels 6,7")
