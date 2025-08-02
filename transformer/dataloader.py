#!/usr/bin/env python3
"""
DataLoader for TransformerIO with appropriate collate function.

Provides PyTorch DataLoader functionality for training transformer models
on ARC-style puzzle sequences.
"""

from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from classes import Puzzle, TransformerIO


class TransformerIODataset(Dataset):
    """
    Dataset for TransformerIO instances.

    Args:
        puzzles: List of Puzzle objects
        n_seq: Sequence length for both input and output
        input_heuristic: Heuristic for input cell selection
        output_heuristic: Heuristic for output cell selection
        transform: Optional transform function to apply to TransformerIO instances
    """

    def __init__(
        self,
        puzzles: List[Puzzle],
        n_seq: int,
        input_heuristic: str = "random",
        output_heuristic: str = "random",
        include_ground_truth: bool = False,
        transform: Optional[Callable] = None,
        seed_offset: int = 0,
    ):
        self.puzzles = puzzles
        self.n_seq = n_seq
        self.input_heuristic = input_heuristic
        self.output_heuristic = output_heuristic
        self.include_ground_truth = include_ground_truth
        self.transform = transform
        self.seed_offset = seed_offset

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, idx: int) -> TransformerIO:
        """Get a TransformerIO instance for the given index."""
        puzzle = self.puzzles[idx]

        # Create TransformerIO with deterministic but varied seed
        seed = self.seed_offset + idx
        transformer_io = TransformerIO.from_puzzle(
            puzzle=puzzle,
            n_seq=self.n_seq,
            input_heuristic=self.input_heuristic,
            output_heuristic=self.output_heuristic,
            include_ground_truth=self.include_ground_truth,
            seed=seed,
        )

        if self.transform:
            transformer_io = self.transform(transformer_io)

        return transformer_io


def collate_transformer_io(
    batch: List[TransformerIO], embed_dim: int = 256
) -> Dict[str, torch.Tensor]:
    """
    Collate function for TransformerIO instances.

    Args:
        batch: List of TransformerIO instances
        embed_dim: Embedding dimension for computing embeddings

    Returns:
        Dictionary containing batched tensors:
        - input_embeddings: (batch_size, n_seq, 4*embed_dim)
        - output_embeddings: (batch_size, n_seq, embed_dim)
        - input_colors: (batch_size, n_seq)
        - output_colors: (batch_size, n_seq)
        - output_positions: (batch_size, n_seq, 2)
        - sequence_lengths: (batch_size,) - all equal to n_seq
        - ground_truth_colors: (batch_size, n_seq) - if available
        - ground_truth_positions: (batch_size, n_seq, 2) - if available
    """
    batch_size = len(batch)
    if batch_size == 0:
        raise ValueError("Empty batch")

    n_seq = batch[0].n_seq

    # Validate all items have same sequence length
    for i, item in enumerate(batch):
        if item.n_seq != n_seq:
            raise ValueError(
                f"Inconsistent sequence lengths: batch[0].n_seq={n_seq}, batch[{i}].n_seq={item.n_seq}"
            )

    # Collect embeddings and other data
    input_embeddings = []
    output_embeddings = []
    input_colors = []
    output_colors = []
    output_positions = []
    ground_truth_colors = []
    ground_truth_positions = []

    for transformer_io in batch:
        # Get embeddings
        input_emb = transformer_io.get_input_embeddings(embed_dim)
        output_emb = transformer_io.get_output_embeddings(embed_dim)

        input_embeddings.append(input_emb)
        output_embeddings.append(output_emb)

        # Get colors and positions
        input_colors.append(transformer_io.get_input_colors())
        output_colors.append(transformer_io.get_output_colors())
        output_positions.append(transformer_io.get_output_positions())

        # Get ground truth if available
        gt_colors = transformer_io.get_ground_truth_colors()
        gt_positions = transformer_io.get_ground_truth_positions()

        if gt_colors is not None:
            ground_truth_colors.append(gt_colors)
        if gt_positions is not None:
            ground_truth_positions.append(gt_positions)

    # Stack into batch tensors
    batched_data = {
        "input_embeddings": torch.stack(
            input_embeddings, dim=0
        ),  # (batch_size, n_seq, 4*embed_dim)
        "output_embeddings": torch.stack(
            output_embeddings, dim=0
        ),  # (batch_size, n_seq, embed_dim)
        "input_colors": torch.stack(input_colors, dim=0),  # (batch_size, n_seq)
        "output_colors": torch.stack(output_colors, dim=0),  # (batch_size, n_seq)
        "output_positions": torch.stack(
            output_positions, dim=0
        ),  # (batch_size, n_seq, 2)
        "sequence_lengths": torch.full(
            (batch_size,), n_seq, dtype=torch.long
        ),  # (batch_size,)
        "n_seq": n_seq,  # Convenience field
    }

    # Add ground truth if available
    if ground_truth_colors:
        batched_data["ground_truth_colors"] = torch.stack(
            ground_truth_colors, dim=0
        )  # (batch_size, n_seq)
    if ground_truth_positions:
        batched_data["ground_truth_positions"] = torch.stack(
            ground_truth_positions, dim=0
        )  # (batch_size, n_seq, 2)

    return batched_data


def create_transformer_io_dataloader(
    puzzles: List[Puzzle],
    n_seq: int,
    batch_size: int = 32,
    embed_dim: int = 256,
    input_heuristic: str = "random",
    output_heuristic: str = "random",
    include_ground_truth: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    seed_offset: int = 0,
    **dataloader_kwargs,
) -> DataLoader:
    """
    Create a DataLoader for TransformerIO instances.

    Args:
        puzzles: List of Puzzle objects
        n_seq: Sequence length for input/output
        batch_size: Batch size for DataLoader
        embed_dim: Embedding dimension for collate function
        input_heuristic: Heuristic for input cell selection
        output_heuristic: Heuristic for output cell selection
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes
        seed_offset: Offset for deterministic seeding
        **dataloader_kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader instance with appropriate collate function
    """
    dataset = TransformerIODataset(
        puzzles=puzzles,
        n_seq=n_seq,
        input_heuristic=input_heuristic,
        output_heuristic=output_heuristic,
        include_ground_truth=include_ground_truth,
        seed_offset=seed_offset,
    )

    # Create collate function with fixed embed_dim
    def collate_fn(batch):
        return collate_transformer_io(batch, embed_dim=embed_dim)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )

    return dataloader


class TransformerIODataModule:
    """
    Data module for managing train/val/test DataLoaders.

    Provides a convenient interface for training transformer models with
    TransformerIO data.
    """

    def __init__(
        self,
        train_puzzles: List[Puzzle],
        val_puzzles: Optional[List[Puzzle]] = None,
        test_puzzles: Optional[List[Puzzle]] = None,
        n_seq: int = 256,
        batch_size: int = 32,
        embed_dim: int = 256,
        input_heuristic: str = "random",
        output_heuristic: str = "random",
        num_workers: int = 0,
    ):
        self.train_puzzles = train_puzzles
        self.val_puzzles = val_puzzles
        self.test_puzzles = test_puzzles
        self.n_seq = n_seq
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.input_heuristic = input_heuristic
        self.output_heuristic = output_heuristic
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return create_transformer_io_dataloader(
            puzzles=self.train_puzzles,
            n_seq=self.n_seq,
            batch_size=self.batch_size,
            embed_dim=self.embed_dim,
            input_heuristic=self.input_heuristic,
            output_heuristic=self.output_heuristic,
            include_ground_truth=True,  # Include ground truth for training
            shuffle=True,
            num_workers=self.num_workers,
            seed_offset=0,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation DataLoader."""
        if self.val_puzzles is None:
            return None

        return create_transformer_io_dataloader(
            puzzles=self.val_puzzles,
            n_seq=self.n_seq,
            batch_size=self.batch_size,
            embed_dim=self.embed_dim,
            input_heuristic=self.input_heuristic,
            output_heuristic=self.output_heuristic,
            include_ground_truth=True,  # Include ground truth for validation
            shuffle=False,
            num_workers=self.num_workers,
            seed_offset=10000,  # Different seed offset for val
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test DataLoader."""
        if self.test_puzzles is None:
            return None

        return create_transformer_io_dataloader(
            puzzles=self.test_puzzles,
            n_seq=self.n_seq,
            batch_size=self.batch_size,
            embed_dim=self.embed_dim,
            input_heuristic=self.input_heuristic,
            output_heuristic=self.output_heuristic,
            include_ground_truth=False,  # No ground truth for test
            shuffle=False,
            num_workers=self.num_workers,
            seed_offset=20000,  # Different seed offset for test
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TransformerIODataModule(train={len(self.train_puzzles)}, "
            f"val={len(self.val_puzzles) if self.val_puzzles else 0}, "
            f"test={len(self.test_puzzles) if self.test_puzzles else 0}, "
            f"n_seq={self.n_seq}, batch_size={self.batch_size})"
        )
