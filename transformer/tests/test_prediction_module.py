"""
Tests for the PredictionModule class.
"""

import pytest
import torch
import torch.nn as nn
from layers import PredictionModule, Coordinates


class TestPredictionModule:
    """Test suite for PredictionModule."""
    
    def test_module_initialization(self):
        """Test that PredictionModule initializes correctly."""
        module = PredictionModule(
            d_model=256,
            n_layers_1=2,
            n_layers_2=2,
            n_layers_3=2,
            n_heads=8,
            d_ff=1024,
            dropout=0.1,
            C=11,
            H_max=30,
            W_max=30
        )
        
        assert module.d_model == 256
        assert module.n_layers_1 == 2
        assert module.n_layers_2 == 2
        assert module.n_layers_3 == 2
        assert module.d_positional == 64  # 256 / 4
        assert module.encoder_stage_1.num_layers == 2
        assert module.encoder_stage_2.num_layers == 2
        assert module.encoder_stage_3.num_layers == 2
        
    def test_d_model_divisibility(self):
        """Test that d_model must be divisible by 4."""
        with pytest.raises(AssertionError):
            PredictionModule(
                d_model=255,  # Not divisible by 4
                n_layers_1=1,
                n_layers_2=1,
                n_layers_3=1,
                n_heads=8,
                d_ff=1024,
                dropout=0.1,
                C=11,
                H_max=30,
                W_max=30
            )
            
    def test_tied_embeddings(self):
        """Test that embeddings are properly tied."""
        module = PredictionModule(
            d_model=256,
            n_layers_1=1,
            n_layers_2=1,
            n_layers_3=1,
            n_heads=8,
            d_ff=1024,
            dropout=0.1,
            C=11,
            H_max=30,
            W_max=30
        )
        
        # Check that color_embedding exists and has correct shape
        assert isinstance(module.color_embedding, nn.Embedding)
        assert module.color_embedding.weight.shape == (11, 256)
        
        # Test embedding and unembedding with tied weights
        colors = torch.tensor([[0, 1, 2, 3]])  # [batch=1, ctx_len=4]
        embedded = module.color_embedding(colors)  # [1, 4, 256]
        unembedded = module.unembed(embedded)  # [1, 4, 11]
        
        assert embedded.shape == (1, 4, 256)
        assert unembedded.shape == (1, 4, 11)
        
    def test_positional_embeddings_input(self):
        """Test creation of input positional embeddings."""
        module = PredictionModule(
            d_model=256,
            n_layers_1=1,
            n_layers_2=1,
            n_layers_3=1,
            n_heads=8,
            d_ff=1024,
            dropout=0.1,
            C=11,
            H_max=30,
            W_max=30
        )
        
        positions = torch.tensor([[[0, 0], [1, 1], [2, 2]]])  # [batch=1, ctx_len=3, 2]
        H, W = 10, 10
        
        pos_emb = module.create_positional_embeddings_input(positions, H, W)
        
        assert pos_emb.shape == (1, 3, 256)
        assert not torch.allclose(pos_emb[0, 0], pos_emb[0, 1])  # Different positions have different embeddings
        
    def test_positional_embeddings_output(self):
        """Test creation of output positional embeddings."""
        module = PredictionModule(
            d_model=256,
            n_layers_1=1,
            n_layers_2=1,
            n_layers_3=1,
            n_heads=8,
            d_ff=1024,
            dropout=0.1,
            C=11,
            H_max=30,
            W_max=30
        )
        
        positions = torch.tensor([[[0, 0], [1, 1], [2, 2]]])  # [batch=1, ctx_len=3, 2]
        H, W = 10, 10
        
        pos_emb = module.create_positional_embeddings_output(positions, H, W)
        
        assert pos_emb.shape == (1, 3, 256)
        
    def test_forward_pass_without_targets(self):
        """Test forward pass without target colors."""
        module = PredictionModule(
            d_model=128,  # Smaller for testing
            n_layers_1=1,
            n_layers_2=1,
            n_layers_3=1,
            n_heads=4,
            d_ff=512,
            dropout=0.1,
            C=11,
            H_max=30,
            W_max=30
        )
        
        batch_size, ctx_len = 2, 5
        input_colors = torch.randint(0, 11, (batch_size, ctx_len))
        input_positions = torch.randint(0, 10, (batch_size, ctx_len, 2))
        output_positions = torch.randint(0, 10, (batch_size, ctx_len, 2))
        H, W = 10, 10
        
        logits, loss = module(
            input_colors,
            input_positions,
            output_positions,
            H, W
        )
        
        assert logits.shape == (batch_size, ctx_len, 11)
        assert loss is None
        
    def test_forward_pass_with_targets(self):
        """Test forward pass with target colors and loss calculation."""
        module = PredictionModule(
            d_model=128,
            n_layers_1=1,
            n_layers_2=1,
            n_layers_3=1,
            n_heads=4,
            d_ff=512,
            dropout=0.1,
            C=11,
            H_max=30,
            W_max=30
        )
        
        batch_size, ctx_len = 2, 5
        input_colors = torch.randint(0, 11, (batch_size, ctx_len))
        target_colors = torch.randint(0, 11, (batch_size, ctx_len))
        input_positions = torch.randint(0, 10, (batch_size, ctx_len, 2))
        output_positions = torch.randint(0, 10, (batch_size, ctx_len, 2))
        H, W = 10, 10
        
        logits, loss = module(
            input_colors,
            input_positions,
            output_positions,
            H, W,
            target_colors
        )
        
        assert logits.shape == (batch_size, ctx_len, 11)
        assert loss is not None
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() > 0  # Loss should be positive
        
    def test_forward_with_coordinates(self):
        """Test forward pass using Coordinates objects."""
        module = PredictionModule(
            d_model=128,
            n_layers_1=1,
            n_layers_2=1,
            n_layers_3=1,
            n_heads=4,
            d_ff=512,
            dropout=0.1,
            C=11,
            H_max=30,
            W_max=30
        )
        
        # Create coordinates
        input_coordinates = [
            Coordinates(0, 0),
            Coordinates(1, 1),
            Coordinates(2, 2)
        ]
        output_coordinates = [
            Coordinates(3, 3),
            Coordinates(4, 4),
            Coordinates(5, 5)
        ]
        
        input_colors = torch.tensor([0, 1, 2])
        target_colors = torch.tensor([3, 4, 5])
        H, W = 10, 10
        
        logits, loss = module.forward_with_coordinates(
            input_colors,
            input_coordinates,
            output_coordinates,
            H, W,
            target_colors
        )
        
        assert logits.shape == (1, 3, 11)
        assert loss is not None
        assert loss.dim() == 0
        
    def test_gradient_flow(self):
        """Test that gradients flow through the module correctly."""
        module = PredictionModule(
            d_model=64,
            n_layers_1=1,
            n_layers_2=1,
            n_layers_3=1,
            n_heads=2,
            d_ff=256,
            dropout=0.1,
            C=11,
            H_max=30,
            W_max=30
        )
        
        # Create simple input
        input_colors = torch.tensor([[0, 1, 2]])
        target_colors = torch.tensor([[1, 2, 3]])
        positions = torch.tensor([[[0, 0], [1, 1], [2, 2]]])
        H, W = 5, 5
        
        # Forward pass
        logits, loss = module(
            input_colors,
            positions,
            positions,  # Same positions for simplicity
            H, W,
            target_colors
        )
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert module.color_embedding.weight.grad is not None
        # Check that encoder has gradients (check first layer)
        for param in module.encoder_stage_1.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break
        
    def test_stage_separation(self):
        """Test that the three stages process data correctly."""
        module = PredictionModule(
            d_model=128,
            n_layers_1=2,
            n_layers_2=3,
            n_layers_3=1,
            n_heads=4,
            d_ff=512,
            dropout=0.1,
            C=11,
            H_max=30,
            W_max=30
        )
        
        # Check that stages have correct number of layers
        assert module.encoder_stage_1.num_layers == 2
        assert module.encoder_stage_2.num_layers == 3
        assert module.encoder_stage_3.num_layers == 1
        
        # Run forward pass to ensure all stages work
        input_colors = torch.tensor([[0, 1]])
        positions = torch.tensor([[[0, 0], [1, 1]]])
        H, W = 5, 5
        
        logits, _ = module(input_colors, positions, positions, H, W)
        assert logits.shape == (1, 2, 11)
        
    def test_batch_processing(self):
        """Test that module handles batches correctly."""
        module = PredictionModule(
            d_model=128,
            n_layers_1=1,
            n_layers_2=1,
            n_layers_3=1,
            n_heads=4,
            d_ff=512,
            dropout=0.1,
            C=11,
            H_max=30,
            W_max=30
        )
        
        batch_size = 4
        ctx_len = 6
        input_colors = torch.randint(0, 11, (batch_size, ctx_len))
        positions = torch.randint(0, 10, (batch_size, ctx_len, 2))
        H, W = 10, 10
        
        logits, _ = module(input_colors, positions, positions, H, W)
        
        assert logits.shape == (batch_size, ctx_len, 11)
        
        # Check that different batches produce different outputs
        assert not torch.allclose(logits[0], logits[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])