#!/usr/bin/env python3
"""
Training script for ARC Puzzle Transformer on actual ARC-AGI datasets.

This script loads tasks from the ARC-1/ARC-2 datasets and trains the
spatial reasoning transformer model with RoPE attention.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from model import SpatialReasoningModel
from classes import Puzzle
from dataloader import TransformerIODataModule


class ARCDataset(Dataset):
    """Dataset for loading ARC-AGI tasks."""
    
    def __init__(self, data_dir: str, split: str = "training", max_tasks: Optional[int] = None):
        """
        Initialize ARC dataset.
        
        Args:
            data_dir: Path to ARC dataset directory (e.g., dataset/ARC-1)
            split: One of "training", "evaluation", "rearc-training"
            max_tasks: Maximum number of tasks to load (None for all)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.tasks = []
        
        # Load all task files
        task_dir = self.data_dir / "data" / split
        if not task_dir.exists():
            raise ValueError(f"Task directory {task_dir} does not exist")
            
        task_files = sorted(task_dir.glob("*.json"))
        if max_tasks:
            task_files = task_files[:max_tasks]
            
        print(f"Loading {len(task_files)} tasks from {task_dir}")
        
        for task_file in task_files:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
                
            # Each task has 'train' examples and 'test' examples
            task_id = task_file.stem
            
            # Process training examples
            for idx, example in enumerate(task_data.get('train', [])):
                self.tasks.append({
                    'task_id': task_id,
                    'example_type': 'train',
                    'example_idx': idx,
                    'input': example['input'],
                    'output': example['output']
                })
                
            # Process test examples
            for idx, example in enumerate(task_data.get('test', [])):
                self.tasks.append({
                    'task_id': task_id,
                    'example_type': 'test',
                    'example_idx': idx,
                    'input': example['input'],
                    'output': example['output']
                })
                
        print(f"Loaded {len(self.tasks)} examples from {len(task_files)} tasks")
        
    def __len__(self):
        return len(self.tasks)
        
    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        # Convert to numpy arrays
        input_grid = np.array(task['input'], dtype=np.int32)
        output_grid = np.array(task['output'], dtype=np.int32)
        
        # Create Puzzle object
        puzzle = Puzzle.from_grids(
            input_grid, 
            output_grid, 
            H_max=30,  # ARC grids are max 30x30
            W_max=30,
            C=11  # ARC uses 11 colors (0-10)
        )
        
        return puzzle


def create_data_loaders(
    dataset_path: str,
    batch_size: int = 16,
    n_seq: int = 128,
    embed_dim: int = 128,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    max_tasks: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for ARC dataset.
    
    Args:
        dataset_path: Path to ARC dataset (e.g., "dataset/ARC-1")
        batch_size: Batch size for training
        n_seq: Sequence length for TransformerIO format
        embed_dim: Embedding dimension
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        num_workers: Number of data loading workers
        max_tasks: Maximum number of tasks to load
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load dataset
    dataset = ARCDataset(dataset_path, split="training", max_tasks=max_tasks)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset splits: {train_size} train, {val_size} val, {test_size} test")
    
    # Create TransformerIO data module
    data_module = TransformerIODataModule(
        train_puzzles=list(train_dataset),
        val_puzzles=list(val_dataset),
        test_puzzles=list(test_dataset),
        n_seq=n_seq,
        batch_size=batch_size,
        embed_dim=embed_dim,
        input_heuristic="random",
        output_heuristic="random",
        num_workers=num_workers
    )
    
    return (
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader()
    )


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy for color predictions."""
    # predictions: (batch_size, n_seq, C) - logits
    # targets: (batch_size, n_seq) - indices
    pred_indices = predictions.argmax(dim=-1)
    correct = (pred_indices == targets).float()
    return correct.mean().item()


def train_epoch(
    model: SpatialReasoningModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[CosineAnnealingLR],
    device: torch.device,
    epoch: int,
    log_interval: int = 10
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_color_loss = 0.0
    total_position_loss = 0.0
    total_color_acc = 0.0
    num_batches = len(train_loader)
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        input_emb = batch["input_embeddings"].to(device)
        gt_colors = batch["ground_truth_colors"].to(device)
        gt_positions = batch["ground_truth_positions"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        color_logits, positions = model.forward_from_embeddings(input_emb)
        
        # Compute losses
        color_loss = nn.CrossEntropyLoss()(
            color_logits.view(-1, color_logits.size(-1)),
            gt_colors.view(-1)
        )
        position_loss = nn.MSELoss()(positions, gt_positions)
        total = color_loss + position_loss
        
        # Backward pass
        total.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Update metrics
        total_loss += total.item()
        total_color_loss += color_loss.item()
        total_position_loss += position_loss.item()
        total_color_acc += compute_accuracy(color_logits, gt_colors)
        
        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed
            
            print(
                f"Epoch {epoch} [{batch_idx + 1}/{num_batches}] "
                f"Loss: {total.item():.4f} "
                f"(Color: {color_loss.item():.4f}, Pos: {position_loss.item():.4f}) "
                f"Color Acc: {compute_accuracy(color_logits, gt_colors):.3f} "
                f"Speed: {batches_per_sec:.1f} batch/s"
            )
    
    # Update scheduler
    if scheduler:
        scheduler.step()
        
    # Average metrics
    metrics = {
        'train_loss': total_loss / num_batches,
        'train_color_loss': total_color_loss / num_batches,
        'train_position_loss': total_position_loss / num_batches,
        'train_color_acc': total_color_acc / num_batches,
        'epoch_time': time.time() - start_time
    }
    
    return metrics


def validate(
    model: SpatialReasoningModel,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Validate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_color_loss = 0.0
    total_position_loss = 0.0
    total_color_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            input_emb = batch["input_embeddings"].to(device)
            gt_colors = batch["ground_truth_colors"].to(device)
            gt_positions = batch["ground_truth_positions"].to(device)
            
            # Forward pass
            color_logits, positions = model.forward_from_embeddings(input_emb)
            
            # Compute losses
            color_loss = nn.CrossEntropyLoss()(
                color_logits.view(-1, color_logits.size(-1)),
                gt_colors.view(-1)
            )
            position_loss = nn.MSELoss()(positions, gt_positions)
            total = color_loss + position_loss
            
            # Update metrics
            total_loss += total.item()
            total_color_loss += color_loss.item()
            total_position_loss += position_loss.item()
            total_color_acc += compute_accuracy(color_logits, gt_colors)
            num_batches += 1
    
    metrics = {
        'val_loss': total_loss / max(num_batches, 1),
        'val_color_loss': total_color_loss / max(num_batches, 1),
        'val_position_loss': total_position_loss / max(num_batches, 1),
        'val_color_acc': total_color_acc / max(num_batches, 1)
    }
    
    return metrics


def save_checkpoint(
    model: SpatialReasoningModel,
    optimizer: optim.Optimizer,
    scheduler: Optional[CosineAnnealingLR],
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str
):
    """Save model checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    
    filename = checkpoint_path / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, filename)
    
    # Also save as latest
    latest_path = checkpoint_path / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_path)
    
    print(f"Saved checkpoint to {filename}")


def main():
    """Main training function."""
    # Training configuration
    config = {
        'dataset_path': 'dataset/ARC-1',
        'batch_size': 32,
        'n_seq': 128,
        'embed_dim': 128,
        'num_heads': 8,
        'encoder_layers': 4,
        'thinking_layers': 4,
        'decoder_layers': 4,
        'ff_dim': 512,
        'dropout': 0.1,
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'warmup_epochs': 5,
        'weight_decay': 0.01,
        'max_tasks': None,  # Use all tasks
        'checkpoint_dir': f'checkpoints/arc_transformer_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'log_interval': 10,
        'save_interval': 5,
        'num_workers': 4
    }
    
    print("ARC Puzzle Transformer Training")
    print("=" * 50)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_path=config['dataset_path'],
        batch_size=config['batch_size'],
        n_seq=config['n_seq'],
        embed_dim=config['embed_dim'],
        train_split=0.8,
        val_split=0.1,
        num_workers=config['num_workers'],
        max_tasks=config['max_tasks']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = SpatialReasoningModel(
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        encoder_layers=config['encoder_layers'],
        thinking_layers=config['thinking_layers'],
        decoder_layers=config['decoder_layers'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout'],
        H_max=30,
        W_max=30,
        C=11
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'] - config['warmup_epochs'],
        eta_min=1e-5
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            log_interval=config['log_interval']
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Train Color Loss: {train_metrics['train_color_loss']:.4f}")
        print(f"  Train Position Loss: {train_metrics['train_position_loss']:.4f}")
        print(f"  Train Color Acc: {train_metrics['train_color_acc']:.3f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val Color Loss: {val_metrics['val_color_loss']:.4f}")
        print(f"  Val Position Loss: {val_metrics['val_position_loss']:.4f}")
        print(f"  Val Color Acc: {val_metrics['val_color_acc']:.3f}")
        print(f"  Epoch Time: {train_metrics['epoch_time']:.1f}s")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if epoch % config['save_interval'] == 0:
            all_metrics = {**train_metrics, **val_metrics}
            save_checkpoint(
                model, optimizer, scheduler, epoch, all_metrics,
                config['checkpoint_dir']
            )
            
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            checkpoint_path = Path(config['checkpoint_dir'])
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            best_path = checkpoint_path / 'checkpoint_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss
            }, best_path)
            print(f"  New best model saved! Val Loss: {best_val_loss:.4f}")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")


if __name__ == "__main__":
    main()