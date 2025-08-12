#!/usr/bin/env python3
"""
Script to inspect and visualize ARC puzzle structure.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def load_puzzle(file_path):
    """Load a puzzle from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_puzzle(puzzle_data):
    """Analyze puzzle structure and patterns."""
    print("Puzzle Analysis")
    print("=" * 50)
    
    # Analyze training examples
    train_examples = puzzle_data.get('train', [])
    test_examples = puzzle_data.get('test', [])
    
    print(f"Number of training examples: {len(train_examples)}")
    print(f"Number of test examples: {len(test_examples)}")
    print()
    
    # Analyze each training example
    for idx, example in enumerate(train_examples):
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        print(f"Training Example {idx + 1}:")
        print(f"  Input shape: {input_grid.shape}")
        print(f"  Output shape: {output_grid.shape}")
        print(f"  Unique colors in input: {sorted(np.unique(input_grid).tolist())}")
        print(f"  Unique colors in output: {sorted(np.unique(output_grid).tolist())}")
        
        # Check for patterns
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape
        
        print(f"  Size transformation: {h_in}x{w_in} -> {h_out}x{w_out}")
        
        # Check if it's a multiplication pattern
        if h_out % h_in == 0 and w_out % w_in == 0:
            h_mult = h_out // h_in
            w_mult = w_out // w_in
            print(f"  Possible multiplication: {h_mult}x{w_mult}")
            
            # Check if it's a tiling pattern
            is_tiling = True
            for i in range(h_mult):
                for j in range(w_mult):
                    tile = output_grid[i*h_in:(i+1)*h_in, j*w_in:(j+1)*w_in]
                    if not np.array_equal(tile, input_grid):
                        is_tiling = False
                        break
                if not is_tiling:
                    break
            
            if is_tiling:
                print(f"  Pattern: Simple {h_mult}x{w_mult} tiling")
            else:
                print(f"  Pattern: Complex transformation (not simple tiling)")
        
        print()
    
    # Analyze test example
    for idx, example in enumerate(test_examples):
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        print(f"Test Example {idx + 1}:")
        print(f"  Input shape: {input_grid.shape}")
        print(f"  Output shape: {output_grid.shape}")
        print(f"  Unique colors in input: {sorted(np.unique(input_grid).tolist())}")
        print(f"  Unique colors in output: {sorted(np.unique(output_grid).tolist())}")
        print()


def visualize_puzzle(puzzle_data, example_idx=0, split='train'):
    """Visualize a specific example from the puzzle."""
    examples = puzzle_data.get(split, [])
    if example_idx >= len(examples):
        print(f"Example {example_idx} not found in {split} split")
        return
    
    example = examples[example_idx]
    input_grid = np.array(example['input'])
    output_grid = np.array(example['output'])
    
    # Create color map (ARC uses colors 0-9)
    colors = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', 
              '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot input
    ax1.imshow(input_grid, cmap='tab10', vmin=0, vmax=9)
    ax1.set_title(f'Input ({input_grid.shape[0]}x{input_grid.shape[1]})')
    ax1.grid(True, which='both', color='gray', linewidth=0.5)
    ax1.set_xticks(np.arange(-0.5, input_grid.shape[1], 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, input_grid.shape[0], 1), minor=True)
    ax1.tick_params(which='minor', size=0)
    
    # Add value labels
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):
            text = ax1.text(j, i, str(input_grid[i, j]),
                          ha="center", va="center", color="white", fontsize=12)
    
    # Plot output
    ax2.imshow(output_grid, cmap='tab10', vmin=0, vmax=9)
    ax2.set_title(f'Output ({output_grid.shape[0]}x{output_grid.shape[1]})')
    ax2.grid(True, which='both', color='gray', linewidth=0.5)
    ax2.set_xticks(np.arange(-0.5, output_grid.shape[1], 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, output_grid.shape[0], 1), minor=True)
    ax2.tick_params(which='minor', size=0)
    
    # Add value labels
    for i in range(output_grid.shape[0]):
        for j in range(output_grid.shape[1]):
            text = ax2.text(j, i, str(output_grid[i, j]),
                          ha="center", va="center", color="white", fontsize=8)
    
    plt.suptitle(f'{split.capitalize()} Example {example_idx + 1}')
    plt.tight_layout()
    plt.show()


def check_transformation_pattern(puzzle_data):
    """Check what kind of transformation pattern this puzzle follows."""
    train_examples = puzzle_data.get('train', [])
    
    print("\nTransformation Pattern Analysis")
    print("=" * 50)
    
    # Check if all examples follow the same size transformation
    size_transforms = []
    for example in train_examples:
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape
        size_transforms.append((h_in, w_in, h_out, w_out))
    
    # Check if it's a consistent multiplication
    if all(h_out == 9 and w_out == 9 for _, _, h_out, w_out in size_transforms):
        print("All outputs are 9x9 grids")
        
        # Check if inputs are all 3x3
        if all(h_in == 3 and w_in == 3 for h_in, w_in, _, _ in size_transforms):
            print("All inputs are 3x3 grids")
            print("Pattern: 3x3 -> 9x9 transformation (3x scaling)")
            
            # Analyze the transformation pattern
            print("\nDetailed pattern analysis:")
            for idx, example in enumerate(train_examples):
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                
                print(f"\nExample {idx + 1}:")
                # Check different regions of the output
                regions = {
                    "Top-left (0:3, 0:3)": output_grid[0:3, 0:3],
                    "Top-center (0:3, 3:6)": output_grid[0:3, 3:6],
                    "Top-right (0:3, 6:9)": output_grid[0:3, 6:9],
                    "Middle-left (3:6, 0:3)": output_grid[3:6, 0:3],
                    "Middle-center (3:6, 3:6)": output_grid[3:6, 3:6],
                    "Middle-right (3:6, 6:9)": output_grid[3:6, 6:9],
                    "Bottom-left (6:9, 0:3)": output_grid[6:9, 0:3],
                    "Bottom-center (6:9, 3:6)": output_grid[6:9, 3:6],
                    "Bottom-right (6:9, 6:9)": output_grid[6:9, 6:9],
                }
                
                matches = []
                for region_name, region in regions.items():
                    if np.array_equal(region, input_grid):
                        matches.append(region_name)
                
                if matches:
                    print(f"  Input pattern found in: {', '.join(matches)}")
                else:
                    print(f"  No exact matches found - checking for transformations...")
                    
                    # Check for rotations/flips
                    rotated_90 = np.rot90(input_grid)
                    rotated_180 = np.rot90(input_grid, 2)
                    rotated_270 = np.rot90(input_grid, 3)
                    flipped_h = np.fliplr(input_grid)
                    flipped_v = np.flipud(input_grid)
                    
                    for region_name, region in regions.items():
                        transforms = []
                        if np.array_equal(region, rotated_90):
                            transforms.append("90° rotation")
                        if np.array_equal(region, rotated_180):
                            transforms.append("180° rotation")
                        if np.array_equal(region, rotated_270):
                            transforms.append("270° rotation")
                        if np.array_equal(region, flipped_h):
                            transforms.append("horizontal flip")
                        if np.array_equal(region, flipped_v):
                            transforms.append("vertical flip")
                        
                        if transforms:
                            print(f"    {region_name}: {', '.join(transforms)}")


def main():
    # Load and analyze a sample puzzle
    puzzle_path = Path("../dataset/ARC-1/data/training/007bbfb7.json")
    
    print(f"Loading puzzle from: {puzzle_path}")
    puzzle_data = load_puzzle(puzzle_path)
    
    # Analyze structure
    analyze_puzzle(puzzle_data)
    
    # Check transformation pattern
    check_transformation_pattern(puzzle_data)
    
    # Visualize examples
    print("\nVisualizing training examples...")
    for i in range(min(3, len(puzzle_data.get('train', [])))):
        visualize_puzzle(puzzle_data, example_idx=i, split='train')
    
    # Visualize test example
    print("\nVisualizing test example...")
    visualize_puzzle(puzzle_data, example_idx=0, split='test')


if __name__ == "__main__":
    main()