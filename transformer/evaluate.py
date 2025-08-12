#!/usr/bin/env python3
"""
Evaluation script for ARC Puzzle Transformer using PredictionModule.

This script evaluates a trained PredictionModule on ARC-AGI test sets and generates
detailed metrics and visualizations.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
import argparse

import gin
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from layers import PredictionModule
from train import ARCDataset, ARCDataLoader, setup_gin_config


# ARC color palette
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue  
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Gray
    '#F012BE',  # 6: Magenta
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Light Blue
    '#870C25',  # 9: Brown
    '#FFFFFF',  # 10: White (background)
]


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[PredictionModule, Dict]:
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Setup gin configuration
    setup_gin_config()
    
    # Create model
    model = PredictionModule().to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def grid_to_image(grid: np.ndarray) -> np.ndarray:
    """Convert ARC grid to RGB image."""
    cmap = ListedColormap(ARC_COLORS)
    # Normalize to 0-1 range for colormap
    normalized = grid / 10.0
    # Apply colormap
    image = cmap(normalized)[:, :, :3]  # Drop alpha channel
    return image


def visualize_prediction(
    input_grid: np.ndarray,
    target_grid: np.ndarray,
    pred_grid: np.ndarray,
    task_id: str,
    save_path: Optional[str] = None
):
    """Visualize input, target, and predicted grids."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input
    axes[0].imshow(grid_to_image(input_grid))
    axes[0].set_title('Input', fontsize=14)
    axes[0].axis('off')
    
    # Target
    axes[1].imshow(grid_to_image(target_grid))
    axes[1].set_title('Target', fontsize=14)
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(grid_to_image(pred_grid))
    axes[2].set_title('Prediction', fontsize=14)
    axes[2].axis('off')
    
    # Add grid lines
    for ax, grid in zip(axes, [input_grid, target_grid, pred_grid]):
        h, w = grid.shape
        for i in range(h + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(w + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)
    
    plt.suptitle(f'Task: {task_id}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def reconstruct_grid_from_predictions(
    color_logits: torch.Tensor,
    positions: torch.Tensor,
    original_shape: Tuple[int, int],
    C: int = 11
) -> np.ndarray:
    """
    Reconstruct a grid from model predictions.
    
    Args:
        color_logits: (seq_len, C) - color predictions
        positions: (seq_len, 2) - position coordinates (already discrete)
        original_shape: (H, W) - shape of the output grid
        C: Number of colors
        
    Returns:
        Reconstructed grid of shape (H, W)
    """
    H, W = original_shape
    grid = np.zeros((H, W), dtype=np.int32)
    
    # Get predicted colors
    predicted_colors = color_logits.argmax(dim=-1).cpu().numpy()
    positions_np = positions.cpu().numpy()
    
    # Place predictions on grid
    seq_len = min(len(predicted_colors), H * W)
    for idx in range(seq_len):
        color = predicted_colors[idx]
        # Positions are already discrete grid coordinates
        y = int(positions_np[idx, 0])
        x = int(positions_np[idx, 1])
        
        # Ensure within bounds
        if 0 <= y < H and 0 <= x < W:
            grid[y, x] = color
    
    return grid


def evaluate_single_task(
    model: PredictionModule,
    task_data: Dict,
    device: torch.device
) -> Dict:
    """Evaluate model on a single task."""
    model.eval()
    
    with torch.no_grad():
        # Get input and output grids
        input_grid = np.array(task_data['input'], dtype=np.int32)
        output_grid = np.array(task_data['output'], dtype=np.int32)
        
        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape
        
        # Use output dimensions
        H, W = H_out, W_out
        
        # Prepare data for PredictionModule
        input_colors = []
        input_positions = []
        output_positions = []
        target_colors = []
        
        # Process grids row by row
        for i in range(H):
            for j in range(W):
                # For input positions, use input grid if within bounds
                if i < H_in and j < W_in:
                    input_colors.append(input_grid[i, j])
                else:
                    input_colors.append(0)  # Background color
                input_positions.append([i, j])
                
                # Output positions and colors
                output_positions.append([i, j])
                target_colors.append(output_grid[i, j])
        
        # Convert to tensors and add batch dimension
        input_colors_t = torch.tensor(input_colors, dtype=torch.long).unsqueeze(0).to(device)
        input_positions_t = torch.tensor(input_positions, dtype=torch.float32).unsqueeze(0).to(device)
        output_positions_t = torch.tensor(output_positions, dtype=torch.float32).unsqueeze(0).to(device)
        target_colors_t = torch.tensor(target_colors, dtype=torch.long).unsqueeze(0).to(device)
        
        # Forward pass
        logits, loss = model(
            input_colors_t,
            input_positions_t,
            output_positions_t,
            H, W,
            target_colors_t
        )
        
        # Compute accuracy
        pred_colors = logits.argmax(dim=-1)
        color_accuracy = (pred_colors == target_colors_t).float().mean().item()
        
        # Reconstruct predicted grid
        pred_grid = reconstruct_grid_from_predictions(
            logits[0],  # First item in batch
            output_positions_t[0],
            (H, W),
            C=11
        )
        
        # Compute pixel accuracy
        pixel_accuracy = (pred_grid == output_grid).mean()
        
        return {
            'loss': loss.item() if loss is not None else 0.0,
            'color_accuracy': color_accuracy,
            'pixel_accuracy': pixel_accuracy,
            'predicted_grid': pred_grid,
            'target_grid': output_grid,
            'input_grid': input_grid
        }


def evaluate_dataset(
    model: PredictionModule,
    dataset_path: str,
    split: str,
    device: torch.device,
    max_tasks: Optional[int] = None,
    visualize: bool = False,
    output_dir: Optional[str] = None
) -> Dict:
    """Evaluate model on entire dataset split."""
    print(f"\nEvaluating on {split} split...")
    
    # Load dataset
    dataset = ARCDataset(dataset_path, split=split, max_tasks=max_tasks)
    
    # Group by task_id
    tasks_dict = {}
    for item in dataset.tasks:
        task_id = item['task_id']
        if task_id not in tasks_dict:
            tasks_dict[task_id] = []
        tasks_dict[task_id].append(item)
    
    print(f"Evaluating {len(tasks_dict)} unique tasks")
    
    # Metrics
    all_metrics = {
        'loss': [],
        'color_accuracy': [],
        'pixel_accuracy': [],
        'perfect_predictions': 0,
        'total_examples': 0
    }
    
    task_results = {}
    
    # Evaluate each task
    for task_idx, (task_id, examples) in enumerate(tasks_dict.items()):
        if task_idx % 10 == 0:
            print(f"  Processing task {task_idx + 1}/{len(tasks_dict)}")
        
        task_metrics = {
            'loss': [],
            'color_accuracy': [],
            'pixel_accuracy': []
        }
        
        for example in examples:
            # Evaluate
            results = evaluate_single_task(model, example, device)
            
            # Update metrics
            task_metrics['loss'].append(results['loss'])
            task_metrics['color_accuracy'].append(results['color_accuracy'])
            task_metrics['pixel_accuracy'].append(results['pixel_accuracy'])
            
            all_metrics['loss'].append(results['loss'])
            all_metrics['color_accuracy'].append(results['color_accuracy'])
            all_metrics['pixel_accuracy'].append(results['pixel_accuracy'])
            all_metrics['total_examples'] += 1
            
            if results['pixel_accuracy'] == 1.0:
                all_metrics['perfect_predictions'] += 1
            
            # Visualize first example of each task
            if visualize and example == examples[0] and output_dir:
                viz_path = Path(output_dir) / 'visualizations' / f'{task_id}.png'
                viz_path.parent.mkdir(parents=True, exist_ok=True)
                visualize_prediction(
                    results['input_grid'],
                    results['target_grid'],
                    results['predicted_grid'],
                    task_id,
                    save_path=str(viz_path)
                )
        
        # Store task results
        task_results[task_id] = {
            'num_examples': len(examples),
            'avg_loss': np.mean(task_metrics['loss']),
            'avg_color_accuracy': np.mean(task_metrics['color_accuracy']),
            'avg_pixel_accuracy': np.mean(task_metrics['pixel_accuracy'])
        }
    
    # Compute overall metrics
    overall_metrics = {
        'num_tasks': len(tasks_dict),
        'num_examples': all_metrics['total_examples'],
        'avg_loss': np.mean(all_metrics['loss']),
        'avg_color_accuracy': np.mean(all_metrics['color_accuracy']),
        'avg_pixel_accuracy': np.mean(all_metrics['pixel_accuracy']),
        'perfect_accuracy_rate': all_metrics['perfect_predictions'] / max(all_metrics['total_examples'], 1)
    }
    
    return overall_metrics, task_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate ARC PredictionModule')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='../dataset/ARC-1',
                        help='Path to ARC dataset')
    parser.add_argument('--split', type=str, default='evaluation',
                        choices=['training', 'evaluation', 'rearc-training'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Maximum number of tasks to evaluate')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/mps/cpu)')
    
    args = parser.parse_args()
    
    # Set device - check for CUDA, then MPS, then CPU
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint_info = load_checkpoint(args.checkpoint, device)
    print(f"Loaded model from epoch {checkpoint_info.get('epoch', 'unknown')}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    print(f"\nEvaluating on {args.dataset} - {args.split} split")
    start_time = time.time()
    
    overall_metrics, task_results = evaluate_dataset(
        model,
        args.dataset,
        args.split,
        device,
        max_tasks=args.max_tasks,
        visualize=args.visualize,
        output_dir=str(output_dir)
    )
    
    eval_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Number of tasks: {overall_metrics['num_tasks']}")
    print(f"Number of examples: {overall_metrics['num_examples']}")
    print(f"Evaluation time: {eval_time:.1f}s")
    print("\nMetrics:")
    print(f"  Average Loss: {overall_metrics['avg_loss']:.4f}")
    print(f"  Average Color Accuracy: {overall_metrics['avg_color_accuracy']:.3f}")
    print(f"  Average Pixel Accuracy: {overall_metrics['avg_pixel_accuracy']:.3f}")
    print(f"  Perfect Prediction Rate: {overall_metrics['perfect_accuracy_rate']:.3f}")
    
    # Save detailed results
    results_file = output_dir / f'results_{args.split}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'overall_metrics': overall_metrics,
            'task_results': task_results,
            'eval_time': eval_time
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Find best and worst performing tasks
    sorted_tasks = sorted(task_results.items(), 
                         key=lambda x: x[1]['avg_pixel_accuracy'], 
                         reverse=True)
    
    if len(sorted_tasks) > 0:
        print("\nBest performing tasks:")
        for task_id, metrics in sorted_tasks[:5]:
            print(f"  {task_id}: {metrics['avg_pixel_accuracy']:.3f} pixel accuracy")
        
        if len(sorted_tasks) > 5:
            print("\nWorst performing tasks:")
            for task_id, metrics in sorted_tasks[-5:]:
                print(f"  {task_id}: {metrics['avg_pixel_accuracy']:.3f} pixel accuracy")
    
    if args.visualize:
        print(f"\nVisualizations saved to: {output_dir}/visualizations/")


if __name__ == "__main__":
    main()