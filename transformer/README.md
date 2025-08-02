# ARC Spatial Reasoning with RoPE Transformers

A configurable transformer encoder with RoPE (Rotary Position Embedding) self-attention for spatial reasoning tasks, designed for the ARC (Abstraction and Reasoning Corpus) challenge.

## Overview

- **Architecture**: Gin-configurable transformer encoder with RoPE self-attention
- **Input/Output**: H × W × C tensors with values 0 to C-1 (C=11 by default)
- **Spatial Embeddings**: 4 2D positional embeddings based on RoPE for coordinate transformations:
  - (i,j) - original coordinates
  - (H_max + (H-1-i), j) - vertical flip with offset
  - (H_max + (H-1-i), W_max + (W-1-j)) - both flips with offsets
  - (i, W_max + (W-1-j)) - horizontal flip with offset
- **Objective**: Predict output location (i',j') given input location (i,j)
- **Training**: Reinforcement learning based objectives

## Quick Start

```bash
# Setup environment
uv sync
source .venv/bin/activate

# Run the model
python main.py

# Format code (automatic on commit)
ruff format .
```

## Key Features

- **RoPE Self-Attention**: 2D spatial awareness with rotational invariance
- **Dynamic Grid Sizes**: Handles variable H×W dimensions without retraining  
- **Gin Configuration**: All hyperparameters configurable via gin-config
- **Pre-commit Hooks**: Automatic code formatting with ruff
- **Modular Design**: Separate embedding, attention, and prediction components

## Coordinate Transformation System

The system uses a 9-channel coordinate encoding:
- **Channel 0**: Original grid values (0 to C-1)
- **Channels 1,2**: (i,j) - original coordinates
- **Channels 3,4**: (H_max + (H-1-i), j) - vertical flip with offset
- **Channels 5,6**: (H_max + (H-1-i), W_max + (W-1-j)) - both flips with offsets
- **Channels 7,8**: (i, W_max + (W-1-j)) - horizontal flip with offset

This encoding enables the transformer to learn spatial reasoning with rotational symmetries while maintaining unique coordinate representations for each transformation.
