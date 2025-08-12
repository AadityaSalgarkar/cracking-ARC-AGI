#!/usr/bin/env python3
"""
Training script for ARC Puzzle Transformer with 9-channel encoding.

This script loads tasks from the ARC-1/ARC-2 datasets and trains the model
using the 9-channel coordinate encoding with random sequence sampling.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
import argparse

import gin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from layers import PredictionModule
from utils import transform_grid_with_coordinates


class ARCDatasetV2(Dataset):
    """Dataset for loading ARC-AGI tasks with 9-channel encoding."""
    
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


def process_puzzle_to_9channel(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    H_max: int = 30,
    W_max: int = 30,
    C: int = 11
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process a puzzle into 9-channel input and 3-channel output format.
    
    Args:
        input_grid: Input puzzle grid (H_in, W_in)
        output_grid: Output puzzle grid (H_out, W_out)
        H_max: Maximum height
        W_max: Maximum width
        C: Number of colors
        
    Returns:
        input_tensor: (H_max, W_max, 9) tensor with 9-channel encoding
        output_tensor: (H_max, W_max, 3) tensor with output colors and positions
    """
    # Convert to torch tensors
    input_grid = torch.tensor(input_grid, dtype=torch.long)
    output_grid = torch.tensor(output_grid, dtype=torch.long)
    
    # Create 9-channel input using existing function
    input_tensor = transform_grid_with_coordinates(input_grid, H_max, W_max, C)
    
    # Create 3-channel output: [color, i, j]
    H_out, W_out = output_grid.shape
    output_tensor = torch.full((H_max, W_max, 3), C - 1, dtype=torch.long)
    
    # Fill in the output grid in top-left corner
    output_tensor[:H_out, :W_out, 0] = output_grid
    
    # Add i,j coordinates for entire grid
    for i in range(H_max):
        for j in range(W_max):
            output_tensor[i, j, 1] = i
            output_tensor[i, j, 2] = j
    
    return input_tensor, output_tensor


def sample_random_sequences(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    seq_len: int = 256
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample independent random sequences from input and output tensors.
    
    Args:
        input_tensor: (H_max, W_max, 9) input tensor
        output_tensor: (H_max, W_max, 3) output tensor
        seq_len: Length of sequence to sample
        
    Returns:
        input_seq: (seq_len, 9) sampled input sequence
        output_seq: (seq_len, 3) sampled output sequence
        input_indices: (seq_len,) indices of sampled input positions
        output_indices: (seq_len,) indices of sampled output positions
    """
    H_max, W_max, _ = input_tensor.shape
    total_positions = H_max * W_max
    
    # Flatten the tensors
    input_flat = input_tensor.view(-1, 9)  # (H_max*W_max, 9)
    output_flat = output_tensor.view(-1, 3)  # (H_max*W_max, 3)
    
    # Sample random indices for input
    if total_positions <= seq_len:
        # If total positions less than seq_len, use all positions with padding
        input_indices = torch.arange(total_positions)
        output_indices = torch.arange(total_positions)
        # Pad with repeated random indices if needed
        if total_positions < seq_len:
            pad_size = seq_len - total_positions
            pad_input_indices = torch.randint(0, total_positions, (pad_size,))
            pad_output_indices = torch.randint(0, total_positions, (pad_size,))
            input_indices = torch.cat([input_indices, pad_input_indices])
            output_indices = torch.cat([output_indices, pad_output_indices])
    else:
        # Sample without replacement for both input and output independently
        input_indices = torch.randperm(total_positions)[:seq_len]
        output_indices = torch.randperm(total_positions)[:seq_len]
    
    # Extract sequences
    input_seq = input_flat[input_indices]
    output_seq = output_flat[output_indices]
    
    return input_seq, output_seq, input_indices, output_indices


class ARCDataLoaderV2:
    """Custom data loader for ARC tasks with 9-channel encoding and random sampling."""
    
    def __init__(
        self,
        dataset: ARCDatasetV2,
        batch_size: int = 32,
        seq_len: int = 256,
        shuffle: bool = True,
        H_max: int = 30,
        W_max: int = 30
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.H_max = H_max
        self.W_max = W_max
        
    def __len__(self):
        return len(self.dataset) // self.batch_size
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
            
        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            
            # Process batch
            batch_tensors = self._process_batch(batch_data)
            yield batch_tensors
            
    def _process_batch(self, batch_data: List[Dict]) -> Dict[str, torch.Tensor]:
        """Process batch data into tensors with 9-channel encoding."""
        batch_size = len(batch_data)
        
        # Initialize batch tensors
        input_batch = torch.zeros(batch_size, self.seq_len, 9, dtype=torch.float32)
        output_batch = torch.zeros(batch_size, self.seq_len, 3, dtype=torch.long)
        input_positions_batch = torch.zeros(batch_size, self.seq_len, 2, dtype=torch.long)
        output_positions_batch = torch.zeros(batch_size, self.seq_len, 2, dtype=torch.long)
        
        for idx, data in enumerate(batch_data):
            # Convert grids to numpy arrays
            input_grid = np.array(data['input'], dtype=np.int32)
            output_grid = np.array(data['output'], dtype=np.int32)
            
            # Process to 9-channel and 3-channel format
            input_tensor, output_tensor = process_puzzle_to_9channel(
                input_grid, output_grid, self.H_max, self.W_max
            )
            
            # Sample random sequences independently from input and output
            input_seq, output_seq, input_indices, output_indices = sample_random_sequences(
                input_tensor, output_tensor, self.seq_len
            )
            
            # Add to batch
            input_batch[idx] = input_seq.float()
            output_batch[idx] = output_seq
            
            # Store the actual positions that were sampled
            # Convert flat indices back to 2D positions for reference
            for i, (in_idx, out_idx) in enumerate(zip(input_indices, output_indices)):
                input_positions_batch[idx, i, 0] = in_idx // self.W_max  # row
                input_positions_batch[idx, i, 1] = in_idx % self.W_max   # col
                output_positions_batch[idx, i, 0] = out_idx // self.W_max
                output_positions_batch[idx, i, 1] = out_idx % self.W_max
        
        return {
            'input': input_batch,  # (batch_size, seq_len, 9)
            'output': output_batch,  # (batch_size, seq_len, 3)
            'target_colors': output_batch[:, :, 0],  # (batch_size, seq_len)
            'input_positions': input_positions_batch,  # (batch_size, seq_len, 2)
            'output_positions': output_positions_batch  # (batch_size, seq_len, 2)
        }


def create_model_for_9channel(
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 1024,
    dropout: float = 0.1,
    C: int = 11
) -> nn.Module:
    """
    Create a transformer model that maps from input sequences to output sequences.
    
    Since input and output sequences are sampled independently, the model needs
    to learn the general transformation from any input position to any output position.
    """
    
    class InputOutputTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Project 9-channel input to d_model
            self.input_projection = nn.Linear(9, d_model)
            
            # Project 3-channel output positions to d_model
            # We'll use the output positions (channels 1,2) to condition the prediction
            self.output_pos_projection = nn.Linear(2, d_model)
            
            # Transformer encoder for processing input
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            
            # Cross-attention layer to attend from output positions to input features
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Final transformer for output prediction
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)
            
            # Output projection to predict colors
            self.output_projection = nn.Linear(d_model, C)
            
        def forward(self, input_seq, output_positions, target_colors=None):
            """
            Args:
                input_seq: (batch_size, seq_len, 9) input tensor with 9-channel encoding
                output_positions: (batch_size, seq_len, 2) output position coordinates
                target_colors: (batch_size, seq_len) target color labels
                
            Returns:
                logits: (batch_size, seq_len, C) color predictions
                loss: scalar loss if target_colors provided
            """
            batch_size, seq_len, _ = input_seq.shape
            
            # Encode input sequence
            input_features = self.input_projection(input_seq)  # (batch_size, seq_len, d_model)
            input_encoded = self.encoder(input_features)  # (batch_size, seq_len, d_model)
            
            # Encode output positions
            output_queries = self.output_pos_projection(output_positions)  # (batch_size, seq_len, d_model)
            
            # Cross-attention: output positions attend to input features
            attended_features, _ = self.cross_attention(
                query=output_queries,
                key=input_encoded,
                value=input_encoded
            )  # (batch_size, seq_len, d_model)
            
            # Combine with output position information
            combined = attended_features + output_queries
            
            # Final decoding
            decoded = self.decoder(combined)  # (batch_size, seq_len, d_model)
            
            # Project to output colors
            logits = self.output_projection(decoded)  # (batch_size, seq_len, C)
            
            # Calculate loss if targets provided
            loss = None
            if target_colors is not None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, C),
                    target_colors.view(-1)
                )
            
            return logits, loss
    
    return InputOutputTransformer()


def train_epoch(
    model: nn.Module,
    train_loader: ARCDataLoaderV2,
    optimizer: optim.Optimizer,
    scheduler: Optional[CosineAnnealingLR],
    device: torch.device,
    epoch: int,
    log_interval: int = 10
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        input_data = batch['input'].to(device)
        output_positions = batch['output_positions'].to(device).float()
        target_colors = batch['target_colors'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, loss = model(input_data, output_positions, target_colors)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        correct = (predictions == target_colors).sum().item()
        batch_samples = target_colors.numel()
        
        # Update metrics
        total_loss += loss.item()
        total_correct += correct
        total_samples += batch_samples
        num_batches += 1
        
        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed
            batch_acc = correct / batch_samples
            
            print(
                f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {batch_acc:.3f} "
                f"Speed: {batches_per_sec:.1f} batch/s"
            )
    
    # Update scheduler
    if scheduler:
        scheduler.step()
        
    # Average metrics
    metrics = {
        'train_loss': total_loss / max(num_batches, 1),
        'train_acc': total_correct / max(total_samples, 1),
        'epoch_time': time.time() - start_time
    }
    
    return metrics


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ARC Transformer with 9-channel encoding')
    parser.add_argument('--max-tasks', type=int, default=1,
                        help='Maximum number of tasks to use for training (default: 1)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--seq-len', type=int, default=256,
                        help='Sequence length for random sampling (default: 256)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--dataset', type=str, default='../dataset/ARC-1',
                        help='Path to dataset (default: ../dataset/ARC-1)')
    parser.add_argument('--d-model', type=int, default=256,
                        help='Model dimension (default: 256)')
    parser.add_argument('--n-layers', type=int, default=6,
                        help='Number of transformer layers (default: 6)')
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'dataset_path': args.dataset,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'learning_rate': args.lr,
        'num_epochs': args.epochs,
        'max_tasks': args.max_tasks,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'checkpoint_dir': f'checkpoints/9channel_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'log_interval': 10,
    }
    
    print("ARC 9-Channel Transformer Training")
    print("=" * 50)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Set device - check for CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create dataset and dataloader
    print("\nCreating dataset...")
    dataset = ARCDatasetV2(
        config['dataset_path'],
        split="training",
        max_tasks=config['max_tasks']
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices = list(range(train_size))
    
    # Create subset
    train_dataset = ARCDatasetV2.__new__(ARCDatasetV2)
    train_dataset.tasks = [dataset.tasks[i] for i in train_indices]
    
    print(f"Training with {len(train_dataset.tasks)} examples")
    
    # Create data loader
    train_loader = ARCDataLoaderV2(
        train_dataset,
        batch_size=config['batch_size'],
        seq_len=config['seq_len'],
        shuffle=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model_for_9channel(
        d_model=config['d_model'],
        n_layers=config['n_layers']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    # Training loop
    print("\nStarting training...")
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, None, device, epoch,
            log_interval=config['log_interval']
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Train Accuracy: {train_metrics['train_acc']:.3f}")
        print(f"  Epoch Time: {train_metrics['epoch_time']:.1f}s")
        
        # Save checkpoint
        if epoch % 5 == 0:
            checkpoint_path = Path(config['checkpoint_dir'])
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            checkpoint_file = checkpoint_path / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['train_loss'],
                'train_acc': train_metrics['train_acc']
            }, checkpoint_file)
            print(f"  Saved checkpoint to {checkpoint_file}")
    
    print("\nTraining completed!")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")


if __name__ == "__main__":
    main()