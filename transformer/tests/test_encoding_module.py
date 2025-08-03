"""
Tests for EncodingModule implementation.

Tests the transformer specified in req.txt with:
- Input: [batch_size, ctx_len] tuples (color, location=(i,j))
- Tied embeddings for color of size d_model
- 2D positional embeddings with 4 coordinate transformations
- Gin configurability
"""

import pytest
import torch
import gin
import math

from layers.Encoding_Module import EncodingModule, create_2d_positional_embedding


class TestCreate2DPositionalEmbedding:
    """Test the create_2d_positional_embedding function."""
    
    def test_embedding_shape(self):
        """Test that embedding has correct shape."""
        d_positional_input = 64
        embedding = create_2d_positional_embedding(5, 10, d_positional_input)
        assert embedding.shape == (d_positional_input,)
    
    def test_embedding_dtype(self):
        """Test that embedding has correct dtype."""
        embedding = create_2d_positional_embedding(0, 0, 32)
        assert embedding.dtype == torch.float32
    
    def test_even_dimension_requirement(self):
        """Test that d_positional_input must be even."""
        with pytest.raises(AssertionError, match="d_positional_input must be even"):
            create_2d_positional_embedding(0, 0, 33)  # Odd dimension
    
    def test_different_positions_different_embeddings(self):
        """Test that different positions produce different embeddings."""
        d_pos = 32
        emb1 = create_2d_positional_embedding(0, 0, d_pos)
        emb2 = create_2d_positional_embedding(1, 0, d_pos)
        emb3 = create_2d_positional_embedding(0, 1, d_pos)
        
        # Should not be equal
        assert not torch.allclose(emb1, emb2)
        assert not torch.allclose(emb1, emb3)
        assert not torch.allclose(emb2, emb3)
    
    def test_same_position_same_embedding(self):
        """Test that same position produces same embedding."""
        d_pos = 32
        emb1 = create_2d_positional_embedding(5, 7, d_pos)
        emb2 = create_2d_positional_embedding(5, 7, d_pos)
        
        assert torch.allclose(emb1, emb2)
    
    def test_sinusoidal_structure(self):
        """Test that embedding follows sinusoidal pattern."""
        d_pos = 4
        i, j = 2, 3
        embedding = create_2d_positional_embedding(i, j, d_pos)
        
        # Check that values are bounded (sin/cos range)
        assert torch.all(embedding >= -1.0)
        assert torch.all(embedding <= 1.0)


class TestEncodingModule:
    """Test the EncodingModule class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.d_model = 256
        self.n_layers = 4
        self.n_heads = 8
        self.batch_size = 2
        self.ctx_len = 10
        self.C = 11
        self.H_max = 30
        self.W_max = 30
        
        self.model = EncodingModule(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            C=self.C,
            H_max=self.H_max,
            W_max=self.W_max
        )
    
    def test_initialization(self):
        """Test model initialization."""
        assert self.model.d_model == self.d_model
        assert self.model.n_layers == self.n_layers
        assert self.model.d_positional_input == self.d_model // 4
        assert self.model.H_max == self.H_max
        assert self.model.W_max == self.W_max
    
    def test_d_model_divisible_by_4(self):
        """Test that d_model must be divisible by 4."""
        with pytest.raises(AssertionError, match="d_model must be divisible by 4"):
            EncodingModule(d_model=255)  # Not divisible by 4
    
    def test_color_embedding_initialization(self):
        """Test color embedding layer initialization."""
        assert isinstance(self.model.color_embedding, torch.nn.Embedding)
        assert self.model.color_embedding.num_embeddings == self.C
        assert self.model.color_embedding.embedding_dim == self.d_model
    
    def test_transformer_encoder_initialization(self):
        """Test transformer encoder initialization."""
        assert isinstance(self.model.transformer_encoder, torch.nn.TransformerEncoder)
        assert len(self.model.transformer_encoder.layers) == self.n_layers
    
    def test_forward_pass_shapes(self):
        """Test forward pass produces correct output shapes."""
        colors = torch.randint(0, self.C, (self.batch_size, self.ctx_len))
        positions = torch.randint(0, 30, (self.batch_size, self.ctx_len, 2)).float()
        
        output = self.model(colors, positions, H=15, W=15)
        
        assert output.shape == (self.batch_size, self.ctx_len, self.d_model)
    
    def test_forward_pass_dtypes(self):
        """Test forward pass produces correct output dtypes."""
        colors = torch.randint(0, self.C, (self.batch_size, self.ctx_len))
        positions = torch.randint(0, 30, (self.batch_size, self.ctx_len, 2)).float()
        
        output = self.model(colors, positions, H=15, W=15)
        
        assert output.dtype == torch.float32
    
    def test_color_embedding_bounds(self):
        """Test that color values must be within valid range."""
        # Valid colors (should work)
        colors = torch.randint(0, self.C, (self.batch_size, self.ctx_len))
        positions = torch.randint(0, 30, (self.batch_size, self.ctx_len, 2)).float()
        
        output = self.model(colors, positions, H=15, W=15)
        assert output.shape == (self.batch_size, self.ctx_len, self.d_model)
        
        # Invalid colors (should raise error)
        invalid_colors = torch.full((self.batch_size, self.ctx_len), self.C)  # >= C
        with pytest.raises(IndexError):
            self.model(invalid_colors, positions, H=15, W=15)
    
    def test_positional_embeddings_creation(self):
        """Test positional embeddings creation with 4 transformations."""
        positions = torch.tensor([[[2, 3], [5, 7]]], dtype=torch.float)  # [1, 2, 2]
        H, W = 10, 12
        
        pos_emb = self.model.create_positional_embeddings(positions, H, W)
        
        # Should have correct shape
        assert pos_emb.shape == (1, 2, self.d_model)
        
        # Should not be all zeros
        assert not torch.allclose(pos_emb, torch.zeros_like(pos_emb))
    
    def test_four_coordinate_transformations(self):
        """Test that 4 coordinate transformations are correctly applied."""
        # Create simple test case
        i, j = 2, 3
        H, W = 5, 7
        positions = torch.tensor([[[i, j]]], dtype=torch.float)  # [1, 1, 2]
        
        pos_emb = self.model.create_positional_embeddings(positions, H, W)
        
        # Extract the embedding for verification
        embedding = pos_emb[0, 0]  # [d_model]
        
        # Should be composed of 4 parts, each of size d_positional_input
        d_pos = self.model.d_positional_input
        part1 = embedding[:d_pos]
        part2 = embedding[d_pos:2*d_pos]
        part3 = embedding[2*d_pos:3*d_pos]
        part4 = embedding[3*d_pos:]
        
        # Verify each part has correct size
        assert part1.shape == (d_pos,)
        assert part2.shape == (d_pos,)
        assert part3.shape == (d_pos,)
        assert part4.shape == (d_pos,)
        
        # Manually create expected embeddings for comparison
        expected_part1 = create_2d_positional_embedding(i, j, d_pos)
        expected_part2 = create_2d_positional_embedding(
            self.H_max + H - 1 - i, j, d_pos
        )
        expected_part3 = create_2d_positional_embedding(
            self.H_max + H - 1 - i, self.W_max + W - 1 - j, d_pos
        )
        expected_part4 = create_2d_positional_embedding(
            i, self.W_max + W - 1 - j, d_pos
        )
        
        # Should match expected values
        assert torch.allclose(part1, expected_part1, atol=1e-6)
        assert torch.allclose(part2, expected_part2, atol=1e-6)
        assert torch.allclose(part3, expected_part3, atol=1e-6)
        assert torch.allclose(part4, expected_part4, atol=1e-6)
    
    def test_embedding_addition(self):
        """Test that color and positional embeddings are added correctly."""
        colors = torch.zeros((1, 1), dtype=torch.long)  # Color 0
        positions = torch.zeros((1, 1, 2), dtype=torch.float)  # Position (0,0)
        
        # Get individual embeddings
        color_emb = self.model.color_embedding(colors)  # [1, 1, d_model]
        pos_emb = self.model.create_positional_embeddings(positions, H=10, W=10)  # [1, 1, d_model]
        
        # Forward pass should add them
        output = self.model(colors, positions, H=10, W=10)
        
        # The transformer may modify the result, but before transformer it should be sum
        expected_input_to_transformer = color_emb + pos_emb
        
        # We can't directly test this since transformer modifies it,
        # but we can test that both embeddings contribute
        assert not torch.allclose(output, color_emb)
        assert not torch.allclose(output, pos_emb)
    
    def test_different_grid_sizes(self):
        """Test with different H and W values."""
        colors = torch.randint(0, self.C, (self.batch_size, self.ctx_len))
        positions = torch.randint(0, 10, (self.batch_size, self.ctx_len, 2)).float()
        
        # Test different grid sizes
        for H, W in [(5, 5), (10, 15), (20, 25)]:
            output = self.model(colors, positions, H=H, W=W)
            assert output.shape == (self.batch_size, self.ctx_len, self.d_model)
    
    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        colors = torch.randint(0, self.C, (self.batch_size, self.ctx_len))
        positions = torch.randint(0, 30, (self.batch_size, self.ctx_len, 2)).float()
        
        output = self.model(colors, positions, H=15, W=15)
        
        # Each batch item should be different (with high probability)
        assert not torch.allclose(output[0], output[1])
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        colors = torch.randint(0, self.C, (self.batch_size, self.ctx_len))
        positions = torch.randint(0, 30, (self.batch_size, self.ctx_len, 2)).float()
        
        output = self.model(colors, positions, H=15, W=15)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        assert self.model.color_embedding.weight.grad is not None
        assert not torch.allclose(
            self.model.color_embedding.weight.grad, 
            torch.zeros_like(self.model.color_embedding.weight.grad)
        )


class TestGinConfiguration:
    """Test gin configuration functionality."""
    
    def test_gin_configurable_decorator(self):
        """Test that EncodingModule is gin configurable."""
        # Clear any existing gin configuration
        gin.clear_config()
        
        # Configure via gin
        gin.parse_config([
            'EncodingModule.d_model = 128',
            'EncodingModule.n_layers = 6',
            'EncodingModule.n_heads = 4',
            'EncodingModule.dropout = 0.2'
        ])
        
        # Create model with gin configuration
        model = EncodingModule()
        
        assert model.d_model == 128
        assert model.n_layers == 6
        # Note: n_heads and dropout are passed to transformer layers,
        # we can't directly access them but gin should handle them
    
    def test_gin_override_with_explicit_params(self):
        """Test that explicit parameters override gin configuration."""
        gin.clear_config()
        
        # Set gin configuration
        gin.parse_config(['EncodingModule.d_model = 128'])
        
        # Override with explicit parameter
        model = EncodingModule(d_model=256)
        
        assert model.d_model == 256  # Should use explicit value, not gin
    
    def test_gin_partial_configuration(self):
        """Test gin configuration with only some parameters."""
        gin.clear_config()
        
        # Configure only some parameters
        gin.parse_config([
            'EncodingModule.d_model = 512',
            'EncodingModule.C = 15'
        ])
        
        model = EncodingModule()
        
        assert model.d_model == 512
        # Other parameters should use defaults
        assert model.n_layers == 6  # default
        assert model.color_embedding.num_embeddings == 15


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_sequence(self):
        """Test with empty sequence (ctx_len=0)."""
        model = EncodingModule(d_model=256)
        
        colors = torch.empty((2, 0), dtype=torch.long)
        positions = torch.empty((2, 0, 2), dtype=torch.float)
        
        output = model(colors, positions, H=10, W=10)
        
        assert output.shape == (2, 0, 256)
    
    def test_single_element_sequence(self):
        """Test with single element sequence."""
        model = EncodingModule(d_model=256)
        
        colors = torch.tensor([[0], [5]], dtype=torch.long)
        positions = torch.tensor([[[2, 3]], [[10, 15]]], dtype=torch.float)
        
        output = model(colors, positions, H=20, W=20)
        
        assert output.shape == (2, 1, 256)
    
    def test_large_coordinates(self):
        """Test with large coordinate values."""
        model = EncodingModule(d_model=256, H_max=100, W_max=100)
        
        colors = torch.randint(0, 11, (1, 5))
        positions = torch.tensor([[[50, 60], [80, 90], [0, 0], [99, 99], [25, 75]]], 
                                dtype=torch.float)
        
        output = model(colors, positions, H=100, W=100)
        
        assert output.shape == (1, 5, 256)
        assert torch.isfinite(output).all()
    
    def test_zero_coordinates(self):
        """Test with zero coordinates."""
        model = EncodingModule(d_model=256)
        
        colors = torch.zeros((1, 3), dtype=torch.long)
        positions = torch.zeros((1, 3, 2), dtype=torch.float)
        
        output = model(colors, positions, H=10, W=10)
        
        assert output.shape == (1, 3, 256)
        assert torch.isfinite(output).all()
    
    def test_different_d_model_sizes(self):
        """Test with various d_model sizes (all divisible by 4)."""
        for d_model in [64, 128, 256, 512, 1024]:
            model = EncodingModule(d_model=d_model)
            
            colors = torch.randint(0, 11, (1, 5))
            positions = torch.randint(0, 30, (1, 5, 2)).float()
            
            output = model(colors, positions, H=15, W=15)
            
            assert output.shape == (1, 5, d_model)
            assert model.d_positional_input == d_model // 4


if __name__ == "__main__":
    pytest.main([__file__])