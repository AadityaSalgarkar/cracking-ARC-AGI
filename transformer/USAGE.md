# Training and Evaluation Scripts Usage

## Training Script (`train.py`)

The training script loads ARC-AGI datasets and trains the spatial reasoning transformer model.

### Basic Usage

```bash
cd transformer/
uv run python train.py
```

### Key Features

- Loads tasks from ARC-1/ARC-2 datasets
- Supports both training and evaluation splits
- Uses TransformerIO data format for efficient sequence processing
- Implements color and position prediction losses
- Saves checkpoints periodically
- Tracks comprehensive metrics including accuracy

### Configuration

The script uses a configuration dictionary that can be modified in the `main()` function:

```python
config = {
    'dataset_path': 'dataset/ARC-1',     # Path to ARC dataset
    'batch_size': 32,                    # Batch size
    'n_seq': 128,                        # Sequence length
    'embed_dim': 128,                    # Embedding dimension
    'num_heads': 8,                      # Number of attention heads
    'encoder_layers': 4,                 # Number of encoder layers
    'thinking_layers': 4,                # Number of thinking layers
    'decoder_layers': 4,                 # Number of decoder layers
    'ff_dim': 512,                       # Feed-forward dimension
    'dropout': 0.1,                      # Dropout rate
    'learning_rate': 1e-3,               # Learning rate
    'num_epochs': 50,                    # Number of epochs
    'max_tasks': None,                   # Max tasks to load (None = all)
    'checkpoint_dir': 'checkpoints/...'  # Checkpoint directory
}
```

### Output

- Checkpoints saved to `checkpoints/arc_transformer_[timestamp]/`
- Best model saved as `checkpoint_best.pt`
- Latest model saved as `checkpoint_latest.pt`
- Periodic checkpoints saved as `checkpoint_epoch_N.pt`

## Evaluation Script (`evaluate.py`)

The evaluation script loads a trained model and evaluates it on ARC test sets.

### Basic Usage

```bash
cd transformer/
uv run python evaluate.py --checkpoint checkpoints/arc_transformer_*/checkpoint_best.pt
```

### Command Line Arguments

```bash
uv run python evaluate.py \
    --checkpoint path/to/checkpoint.pt \
    --dataset dataset/ARC-1 \
    --split evaluation \
    --max-tasks 100 \
    --visualize \
    --output-dir evaluation_results
```

### Arguments

- `--checkpoint`: Path to model checkpoint (required)
- `--dataset`: Path to ARC dataset (default: dataset/ARC-1)
- `--split`: Dataset split to evaluate (training/evaluation/rearc-training)
- `--max-tasks`: Maximum number of tasks to evaluate (default: all)
- `--visualize`: Generate visualization images
- `--output-dir`: Output directory for results (default: evaluation_results)
- `--device`: Device to use (cuda/cpu)

### Output

- Detailed metrics printed to console
- Results saved to JSON file: `results_[split]_[timestamp].json`
- Visualizations saved to `[output_dir]/visualizations/` (if --visualize)

### Metrics

- **Color Loss**: Cross-entropy loss for color predictions
- **Position Loss**: MSE loss for position predictions  
- **Color Accuracy**: Accuracy of color predictions
- **Pixel Accuracy**: Accuracy of reconstructed grid
- **Perfect Prediction Rate**: Percentage of perfectly reconstructed grids

## Example Workflow

1. **Train a model**:
   ```bash
   cd transformer/
   uv run python train.py
   ```

2. **Monitor training**:
   - Watch console output for loss and accuracy metrics
   - Check checkpoint directory for saved models

3. **Evaluate on test set**:
   ```bash
   uv run python evaluate.py \
       --checkpoint checkpoints/arc_transformer_*/checkpoint_best.pt \
       --split evaluation \
       --visualize
   ```

4. **Analyze results**:
   - Check console output for overall metrics
   - Review JSON file for detailed per-task results
   - Examine visualizations to understand model behavior

## Notes

- The scripts assume the dataset is located at `../dataset/` relative to the transformer directory
- GPU is used automatically if available
- The TransformerIO format sequences grid cells for efficient processing
- Position predictions use continuous coordinates that are discretized for grid reconstruction