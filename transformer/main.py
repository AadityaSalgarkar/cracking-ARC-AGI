#!/usr/bin/env python3
"""
Training loop for ARC puzzle transformer using TransformerIO sequences.

Trains the three-module transformer architecture on shift puzzles from
puzzles/create_puzzle.py with proper ground truth loss computation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import gin

from model import SpatialReasoningModel
from classes import Puzzle
from dataloader import TransformerIODataModule, create_transformer_io_dataloader
from puzzles.create_puzzle import create_shift_puzzle


def create_puzzle_dataset(num_puzzles: int = 1000, seed: int = 42) -> list[Puzzle]:
    """Create a dataset of shift puzzles."""
    puzzles = []
    torch.manual_seed(seed)

    for i in range(num_puzzles):
        input_grid, output_grid = create_shift_puzzle(seed=seed + i)
        puzzle = Puzzle.from_grids(input_grid, output_grid, H_max=30, W_max=30, C=11)
        puzzles.append(puzzle)

    return puzzles


def compute_loss(
    model_outputs: tuple[torch.Tensor, torch.Tensor],
    ground_truth_colors: torch.Tensor,
    ground_truth_positions: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Compute loss for color and position predictions using tied embeddings.

    Args:
        model_outputs: (color_logits, positions) from model
            - color_logits: (batch_size, n_seq, C) distributions over color vocabulary
            - positions: (batch_size, n_seq, 2) predicted positions
        ground_truth_colors: (batch_size, n_seq) target color indices
        ground_truth_positions: (batch_size, n_seq, 2) target positions

    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary with individual losses
    """
    color_logits, positions = model_outputs

    # Color loss using cross-entropy on vocabulary distributions
    # color_logits: (batch_size, n_seq, C) - distributions over color vocabulary
    # ground_truth_colors: (batch_size, n_seq) - color indices
    color_loss = nn.CrossEntropyLoss()(
        color_logits.view(-1, color_logits.size(-1)),  # (batch_size * n_seq, C)
        ground_truth_colors.view(-1),  # (batch_size * n_seq)
    )

    # Position loss (MSE) 
    position_loss = nn.MSELoss()(positions, ground_truth_positions)

    # Combined loss
    total_loss = color_loss + position_loss

    loss_dict = {
        "total_loss": total_loss.item(),
        "color_loss": color_loss.item(),
        "position_loss": position_loss.item(),
    }

    return total_loss, loss_dict


def validate_model(
    model: SpatialReasoningModel, val_loader, device: torch.device
) -> dict:
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_color_loss = 0.0
    total_position_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            input_emb = batch["input_embeddings"].to(device)
            gt_colors = batch["ground_truth_colors"].to(device)
            gt_positions = batch["ground_truth_positions"].to(device)

            # Forward pass through the model
            model_outputs = model.forward_from_embeddings(input_emb)
            
            # Compute validation loss
            loss, loss_dict = compute_loss(model_outputs, gt_colors, gt_positions)
            
            # Accumulate losses
            total_loss += loss_dict["total_loss"]
            total_color_loss += loss_dict["color_loss"] 
            total_position_loss += loss_dict["position_loss"]
            num_batches += 1

    return {
        "val_loss": total_loss / max(num_batches, 1),
        "val_color_loss": total_color_loss / max(num_batches, 1),
        "val_position_loss": total_position_loss / max(num_batches, 1),
    }


def train_epoch(
    model: SpatialReasoningModel,
    train_loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    total_color_loss = 0.0
    total_position_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        input_emb = batch["input_embeddings"].to(device)
        output_emb = batch["output_embeddings"].to(device)
        gt_colors = batch["ground_truth_colors"].to(device)
        gt_positions = batch["ground_truth_positions"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass through the model using pre-computed input embeddings
        # This uses the forward_from_embeddings method which handles TransformerIO batch format
        model_outputs = model.forward_from_embeddings(input_emb)
        
        # Compute loss using the actual model outputs
        loss, loss_dict = compute_loss(model_outputs, gt_colors, gt_positions)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update running totals
        total_loss += loss_dict["total_loss"]
        total_color_loss += loss_dict["color_loss"]
        total_position_loss += loss_dict["position_loss"]

        # Print progress
        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                f"Loss: {loss_dict['total_loss']:.4f} "
                f"(Color: {loss_dict['color_loss']:.4f}, "
                f"Position: {loss_dict['position_loss']:.4f})"
            )

    return {
        "train_loss": total_loss / num_batches,
        "train_color_loss": total_color_loss / num_batches,
        "train_position_loss": total_position_loss / num_batches,
    }


def main():
    """Main training loop."""
    print("Starting ARC Puzzle Transformer Training")
    print("=" * 50)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset
    print("Creating puzzle dataset...")
    puzzles = create_puzzle_dataset(num_puzzles=1000, seed=42)
    print(f"Created {len(puzzles)} puzzles")

    # Split dataset
    train_size = int(0.8 * len(puzzles))
    val_size = int(0.1 * len(puzzles))
    test_size = len(puzzles) - train_size - val_size

    train_puzzles, val_puzzles, test_puzzles = random_split(
        puzzles,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(
        f"Dataset split: {len(train_puzzles)} train, {len(val_puzzles)} val, {len(test_puzzles)} test"
    )

    # Create data module
    data_module = TransformerIODataModule(
        train_puzzles=train_puzzles,
        val_puzzles=val_puzzles,
        test_puzzles=test_puzzles,
        n_seq=128,  # Sequence length
        batch_size=16,
        embed_dim=64,  # Smaller for faster training
        input_heuristic="random",
        output_heuristic="random",
        num_workers=0,
    )

    print(f"Data module: {data_module}")

    # Create model
    print("Creating model...")
    model = SpatialReasoningModel(
        embed_dim=64,
        num_heads=8,
        encoder_layers=2,
        thinking_layers=2,
        decoder_layers=2,
        ff_dim=256,
        dropout=0.1,
        H_max=30,
        W_max=30,
        C=11,
    )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Using Adam optimizer with lr=1e-3")

    # Get data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test one batch to verify data loading
    print("\nTesting data loading...")
    sample_batch = next(iter(train_loader))
    print(f"Sample batch keys: {list(sample_batch.keys())}")
    print(f"Input embeddings shape: {sample_batch['input_embeddings'].shape}")
    print(f"Output embeddings shape: {sample_batch['output_embeddings'].shape}")
    print(f"Ground truth colors shape: {sample_batch['ground_truth_colors'].shape}")
    print(
        f"Ground truth positions shape: {sample_batch['ground_truth_positions'].shape}"
    )

    # Training loop
    print("\nStarting training...")
    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch + 1)
        
        # Validate
        val_metrics = validate_model(model, val_loader, device)

        # Print epoch summary
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Train Color Loss: {train_metrics['train_color_loss']:.4f}")
        print(f"  Train Position Loss: {train_metrics['train_position_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val Color Loss: {val_metrics['val_color_loss']:.4f}")
        print(f"  Val Position Loss: {val_metrics['val_position_loss']:.4f}")

    print("\nTraining completed!")
    print("The model has been successfully integrated with TransformerIO format.")
    print("Training uses actual model forward pass with tied embeddings.")


if __name__ == "__main__":
    main()
