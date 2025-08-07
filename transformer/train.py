#!/usr/bin/env python3
"""
Training script for ARC Puzzle Transformer using PredictionModule.

This script loads tasks from the ARC-1/ARC-2 datasets and trains the
PredictionModule with its three-stage architecture.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime

import gin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from layers import PredictionModule
from classes import Puzzle


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
        return self.tasks[idx]


class ARCDataLoader:
    """Custom data loader for ARC tasks that creates batches for PredictionModule."""
    
    def __init__(self, dataset: ARCDataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __len__(self):
        return len(self.dataset) // self.batch_size
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
            
        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            
            # Process batch to create input/output tensors
            batch_tensors = self._process_batch(batch_data)
            yield batch_tensors
            
    def _process_batch(self, batch_data: List[Dict]) -> Dict[str, torch.Tensor]:
        """Process batch data into tensors for PredictionModule."""
        max_seq_len = 900  # 30x30 max grid size
        batch_size = len(batch_data)
        
        # Initialize lists to collect data
        all_input_colors = []
        all_input_positions = []
        all_output_positions = []
        all_target_colors = []
        all_H = []
        all_W = []
        all_seq_lens = []
        
        for data in batch_data:
            input_grid = np.array(data['input'], dtype=np.int32)
            output_grid = np.array(data['output'], dtype=np.int32)
            
            H_in, W_in = input_grid.shape
            H_out, W_out = output_grid.shape
            
            # Use output dimensions for both (simplified approach)
            H, W = H_out, W_out
            
            # Flatten grids and create position tensors
            input_colors = []
            input_positions = []
            output_colors = []
            output_positions = []
            
            # Process input grid
            for i in range(H_in):
                for j in range(W_in):
                    if i < H and j < W:  # Only if within output dimensions
                        input_colors.append(input_grid[i, j])
                        input_positions.append([i, j])
                        
            # Process output grid
            for i in range(H_out):
                for j in range(W_out):
                    output_colors.append(output_grid[i, j])
                    output_positions.append([i, j])
                    
            # Ensure we have matching lengths
            seq_len = min(len(input_colors), len(output_colors))
            
            all_input_colors.append(torch.tensor(input_colors[:seq_len], dtype=torch.long))
            all_input_positions.append(torch.tensor(input_positions[:seq_len], dtype=torch.float32))
            all_output_positions.append(torch.tensor(output_positions[:seq_len], dtype=torch.float32))
            all_target_colors.append(torch.tensor(output_colors[:seq_len], dtype=torch.long))
            all_H.append(H)
            all_W.append(W)
            all_seq_lens.append(seq_len)
            
        # Pad sequences to max length in batch
        max_len = max(all_seq_lens)
        
        # Create padded tensors
        input_colors = torch.zeros(batch_size, max_len, dtype=torch.long)
        input_positions = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
        output_positions = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
        target_colors = torch.zeros(batch_size, max_len, dtype=torch.long)
        
        for idx in range(batch_size):
            seq_len = all_seq_lens[idx]
            input_colors[idx, :seq_len] = all_input_colors[idx]
            input_positions[idx, :seq_len] = all_input_positions[idx]
            output_positions[idx, :seq_len] = all_output_positions[idx]
            target_colors[idx, :seq_len] = all_target_colors[idx]
            
        # Use the most common H and W in the batch (simplified)
        H = max(all_H)
        W = max(all_W)
        
        return {
            'input_colors': input_colors,
            'input_positions': input_positions,
            'output_positions': output_positions,
            'target_colors': target_colors,
            'H': H,
            'W': W,
            'seq_lens': torch.tensor(all_seq_lens, dtype=torch.long)
        }


def create_data_loaders(
    dataset_path: str,
    batch_size: int = 16,
    train_split: float = 0.8,
    val_split: float = 0.1,
    max_tasks: Optional[int] = None
) -> Tuple[ARCDataLoader, ARCDataLoader, ARCDataLoader]:
    """
    Create data loaders for ARC dataset.
    
    Args:
        dataset_path: Path to ARC dataset (e.g., "dataset/ARC-1")
        batch_size: Batch size for training
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
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
    
    # Create subset indices
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    train_dataset = ARCDataset.__new__(ARCDataset)
    train_dataset.tasks = [dataset.tasks[i] for i in train_indices]
    
    val_dataset = ARCDataset.__new__(ARCDataset)
    val_dataset.tasks = [dataset.tasks[i] for i in val_indices]
    
    test_dataset = ARCDataset.__new__(ARCDataset)
    test_dataset.tasks = [dataset.tasks[i] for i in test_indices]
    
    print(f"Dataset splits: {len(train_dataset.tasks)} train, {len(val_dataset.tasks)} val, {len(test_dataset.tasks)} test")
    
    # Create data loaders
    train_loader = ARCDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = ARCDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = ARCDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, seq_lens: torch.Tensor) -> float:
    """Compute accuracy for color predictions, accounting for sequence lengths."""
    batch_size = predictions.shape[0]
    total_correct = 0
    total_count = 0
    
    pred_indices = predictions.argmax(dim=-1)
    
    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        correct = (pred_indices[b, :seq_len] == targets[b, :seq_len]).float().sum()
        total_correct += correct.item()
        total_count += seq_len
        
    return total_correct / max(total_count, 1)


def train_epoch(
    model: PredictionModule,
    train_loader: ARCDataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[CosineAnnealingLR],
    device: torch.device,
    epoch: int,
    log_interval: int = 10
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        input_colors = batch['input_colors'].to(device)
        input_positions = batch['input_positions'].to(device)
        output_positions = batch['output_positions'].to(device)
        target_colors = batch['target_colors'].to(device)
        seq_lens = batch['seq_lens'].to(device)
        H = batch['H']
        W = batch['W']
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, loss = model(
            input_colors,
            input_positions,
            output_positions,
            H, W,
            target_colors
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_acc += compute_accuracy(logits, target_colors, seq_lens)
        num_batches += 1
        
        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed
            
            print(
                f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {compute_accuracy(logits, target_colors, seq_lens):.3f} "
                f"Speed: {batches_per_sec:.1f} batch/s"
            )
    
    # Update scheduler
    if scheduler:
        scheduler.step()
        
    # Average metrics
    metrics = {
        'train_loss': total_loss / max(num_batches, 1),
        'train_acc': total_acc / max(num_batches, 1),
        'epoch_time': time.time() - start_time
    }
    
    return metrics


def validate(
    model: PredictionModule,
    val_loader: ARCDataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Validate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            input_colors = batch['input_colors'].to(device)
            input_positions = batch['input_positions'].to(device)
            output_positions = batch['output_positions'].to(device)
            target_colors = batch['target_colors'].to(device)
            seq_lens = batch['seq_lens'].to(device)
            H = batch['H']
            W = batch['W']
            
            # Forward pass
            logits, loss = model(
                input_colors,
                input_positions,
                output_positions,
                H, W,
                target_colors
            )
            
            # Update metrics
            total_loss += loss.item()
            total_acc += compute_accuracy(logits, target_colors, seq_lens)
            num_batches += 1
    
    metrics = {
        'val_loss': total_loss / max(num_batches, 1),
        'val_acc': total_acc / max(num_batches, 1)
    }
    
    return metrics


def save_checkpoint(
    model: PredictionModule,
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


def setup_gin_config():
    """Setup gin configuration for PredictionModule."""
    gin_config = """
    # PredictionModule configuration
    PredictionModule.d_model = 256
    PredictionModule.n_layers_1 = 2
    PredictionModule.n_layers_2 = 3
    PredictionModule.n_layers_3 = 2
    PredictionModule.n_heads = 8
    PredictionModule.d_ff = 1024
    PredictionModule.dropout = 0.1
    PredictionModule.C = 11
    PredictionModule.H_max = 30
    PredictionModule.W_max = 30
    """
    gin.parse_config(gin_config)


def main():
    """Main training function."""
    # Setup gin configuration
    setup_gin_config()
    
    # Training configuration
    config = {
        'dataset_path': 'dataset/ARC-1',
        'batch_size': 32,
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'warmup_epochs': 5,
        'weight_decay': 0.01,
        'max_tasks': None,  # Use all tasks
        'checkpoint_dir': f'checkpoints/prediction_module_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'log_interval': 10,
        'save_interval': 5
    }
    
    print("ARC PredictionModule Training")
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
        train_split=0.8,
        val_split=0.1,
        max_tasks=config['max_tasks']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating PredictionModule...")
    model = PredictionModule().to(device)
    
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
        print(f"  Train Accuracy: {train_metrics['train_acc']:.3f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['val_acc']:.3f}")
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