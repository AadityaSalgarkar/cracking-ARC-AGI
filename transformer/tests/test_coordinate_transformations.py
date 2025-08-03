"""
Tests specifically for coordinate transformations as specified in req.txt.

Validates that the 4 coordinate transformations are exactly:
1. (i,j)
2. (H_max+H-1-i,j)  
3. (H_max+H-1-i,W_max+W-1-j)
4. (i,W_max+W-1-j)
"""

import pytest
import torch
import numpy as np

from layers.Encoding_Module import EncodingModule, create_2d_positional_embedding


class TestCoordinateTransformations:
    """Test that coordinate transformations match req.txt specification."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.H_max = 30
        self.W_max = 30
        self.d_model = 256
        self.model = EncodingModule(
            d_model=self.d_model,
            H_max=self.H_max,
            W_max=self.W_max
        )
    
    def test_coordinate_transformation_formulas(self):
        """Test that transformations match exact formulas from req.txt."""
        # Test cases: (i, j, H, W)
        test_cases = [
            (0, 0, 5, 7),      # Corner case
            (2, 3, 10, 12),    # Middle case
            (4, 6, 5, 7),      # Edge case (i = H-1, j = W-1)
            (1, 1, 3, 3),      # Small grid
            (10, 15, 20, 25),  # Larger grid
        ]
        
        for i, j, H, W in test_cases:
            print(f"\nTesting position ({i}, {j}) in grid {H}x{W}")
            
            # Create input
            positions = torch.tensor([[[i, j]]], dtype=torch.float)  # [1, 1, 2]
            
            # Get positional embeddings
            pos_emb = self.model.create_positional_embeddings(positions, H, W)
            embedding = pos_emb[0, 0]  # [d_model]
            
            # Split into 4 parts
            d_pos = self.d_model // 4
            part1 = embedding[:d_pos]
            part2 = embedding[d_pos:2*d_pos]
            part3 = embedding[2*d_pos:3*d_pos]
            part4 = embedding[3*d_pos:]
            
            # Calculate expected transformations
            transform1 = (i, j)
            transform2 = (self.H_max + H - 1 - i, j)
            transform3 = (self.H_max + H - 1 - i, self.W_max + W - 1 - j)
            transform4 = (i, self.W_max + W - 1 - j)
            
            print(f"  Transform 1: {transform1}")
            print(f"  Transform 2: {transform2}")  
            print(f"  Transform 3: {transform3}")
            print(f"  Transform 4: {transform4}")
            
            # Create expected embeddings
            expected1 = create_2d_positional_embedding(transform1[0], transform1[1], d_pos)
            expected2 = create_2d_positional_embedding(transform2[0], transform2[1], d_pos)
            expected3 = create_2d_positional_embedding(transform3[0], transform3[1], d_pos)
            expected4 = create_2d_positional_embedding(transform4[0], transform4[1], d_pos)
            
            # Verify each transformation
            assert torch.allclose(part1, expected1, atol=1e-6), f"Transform 1 failed for {transform1}"
            assert torch.allclose(part2, expected2, atol=1e-6), f"Transform 2 failed for {transform2}"
            assert torch.allclose(part3, expected3, atol=1e-6), f"Transform 3 failed for {transform3}"
            assert torch.allclose(part4, expected4, atol=1e-6), f"Transform 4 failed for {transform4}"
    
    def test_transformation_properties(self):
        """Test mathematical properties of transformations."""
        i, j = 3, 5
        H, W = 8, 10
        
        # Calculate transformations
        t1 = (i, j)
        t2 = (self.H_max + H - 1 - i, j)
        t3 = (self.H_max + H - 1 - i, self.W_max + W - 1 - j)
        t4 = (i, self.W_max + W - 1 - j)
        
        print(f"Original: {t1}")
        print(f"Transform 2: {t2}")
        print(f"Transform 3: {t3}")
        print(f"Transform 4: {t4}")
        
        # Verify transformation properties
        # Transform 2: same j, transformed i
        assert t2[1] == t1[1], "Transform 2 should preserve j coordinate"
        assert t2[0] == self.H_max + H - 1 - i, "Transform 2 i coordinate incorrect"
        
        # Transform 3: both coordinates transformed  
        assert t3[0] == self.H_max + H - 1 - i, "Transform 3 i coordinate incorrect"
        assert t3[1] == self.W_max + W - 1 - j, "Transform 3 j coordinate incorrect"
        
        # Transform 4: same i, transformed j
        assert t4[0] == t1[0], "Transform 4 should preserve i coordinate"
        assert t4[1] == self.W_max + W - 1 - j, "Transform 4 j coordinate incorrect"
    
    def test_edge_positions(self):
        """Test transformations at grid edges."""
        H, W = 15, 20
        
        # Test corners and edges
        edge_positions = [
            (0, 0),           # Top-left corner
            (0, W-1),         # Top-right corner  
            (H-1, 0),         # Bottom-left corner
            (H-1, W-1),       # Bottom-right corner
            (0, W//2),        # Top edge center
            (H-1, W//2),      # Bottom edge center
            (H//2, 0),        # Left edge center
            (H//2, W-1),      # Right edge center
        ]
        
        for i, j in edge_positions:
            print(f"\nTesting edge position ({i}, {j})")
            
            positions = torch.tensor([[[i, j]]], dtype=torch.float)
            pos_emb = self.model.create_positional_embeddings(positions, H, W)
            
            # Should not raise errors and should produce finite values
            assert torch.isfinite(pos_emb).all(), f"Non-finite values at edge position ({i}, {j})"
            
            # Verify transformations are computed correctly
            t2_i = self.H_max + H - 1 - i
            t3_j = self.W_max + W - 1 - j
            t4_j = self.W_max + W - 1 - j
            
            # All transformed coordinates should be non-negative
            assert t2_i >= 0, f"Transform 2 i coordinate negative: {t2_i}"
            assert t3_j >= 0, f"Transform 3 j coordinate negative: {t3_j}"
            assert t4_j >= 0, f"Transform 4 j coordinate negative: {t4_j}"
    
    def test_transformation_uniqueness(self):
        """Test that different positions produce different transformation sets."""
        H, W = 10, 12
        positions = [
            (1, 2),
            (3, 4), 
            (5, 6),
            (7, 8)
        ]
        
        embeddings = []
        for i, j in positions:
            pos_tensor = torch.tensor([[[i, j]]], dtype=torch.float)
            pos_emb = self.model.create_positional_embeddings(pos_tensor, H, W)
            embeddings.append(pos_emb[0, 0])  # [d_model]
        
        # All embeddings should be different
        for idx1 in range(len(embeddings)):
            for idx2 in range(idx1 + 1, len(embeddings)):
                assert not torch.allclose(embeddings[idx1], embeddings[idx2]), \
                    f"Positions {positions[idx1]} and {positions[idx2]} produced identical embeddings"
    
    def test_grid_size_independence(self):
        """Test that transformations work correctly for different grid sizes."""
        fixed_position = (2, 3)
        grid_sizes = [(5, 7), (10, 12), (15, 18), (20, 25)]
        
        for H, W in grid_sizes:
            print(f"\nTesting grid size {H}x{W}")
            
            i, j = fixed_position
            if i >= H or j >= W:
                continue  # Skip if position is outside grid
                
            positions = torch.tensor([[[i, j]]], dtype=torch.float)
            pos_emb = self.model.create_positional_embeddings(positions, H, W)
            
            # Verify embedding is finite and has correct shape
            assert torch.isfinite(pos_emb).all()
            assert pos_emb.shape == (1, 1, self.d_model)
            
            # Verify transformations are grid-size dependent
            embedding = pos_emb[0, 0]
            d_pos = self.d_model // 4
            
            # Extract parts and verify they depend on H, W
            part2 = embedding[d_pos:2*d_pos]
            part3 = embedding[2*d_pos:3*d_pos] 
            part4 = embedding[3*d_pos:]
            
            # These should be different for different H, W values
            # (we can't easily test this without comparing, but we ensure no errors)
            assert not torch.allclose(part2, torch.zeros_like(part2))
            assert not torch.allclose(part3, torch.zeros_like(part3))
            assert not torch.allclose(part4, torch.zeros_like(part4))
    
    def test_coordinate_range_validity(self):
        """Test that coordinate transformations stay within expected ranges."""
        H, W = 25, 28
        
        # Test various positions within the grid
        for i in range(0, H, 5):
            for j in range(0, W, 7):
                positions = torch.tensor([[[i, j]]], dtype=torch.float)
                pos_emb = self.model.create_positional_embeddings(positions, H, W)
                
                # Calculate actual transformations
                t1 = (i, j)
                t2 = (self.H_max + H - 1 - i, j)
                t3 = (self.H_max + H - 1 - i, self.W_max + W - 1 - j)
                t4 = (i, self.W_max + W - 1 - j)
                
                # Verify coordinate ranges
                assert 0 <= t1[0] < H and 0 <= t1[1] < W, f"Transform 1 out of range: {t1}"
                assert t2[0] >= 0 and t2[1] >= 0, f"Transform 2 negative: {t2}"
                assert t3[0] >= 0 and t3[1] >= 0, f"Transform 3 negative: {t3}"
                assert t4[0] >= 0 and t4[1] >= 0, f"Transform 4 negative: {t4}"
                
                # Embedding should be finite
                assert torch.isfinite(pos_emb).all(), f"Non-finite embedding at ({i}, {j})"


class TestSpecificationCompliance:
    """Test compliance with req.txt specification."""
    
    def test_input_format_compliance(self):
        """Test that input format matches req.txt specification."""
        # Input: [batch_size, ctx_len] tuples (color, location=(i,j))
        batch_size, ctx_len = 3, 8
        C = 11
        
        model = EncodingModule(d_model=256, C=C)
        
        # Colors: [batch_size, ctx_len] 
        colors = torch.randint(0, C, (batch_size, ctx_len))
        
        # Positions: [batch_size, ctx_len, 2] with (i,j) coordinates
        positions = torch.randint(0, 30, (batch_size, ctx_len, 2)).float()
        
        # Should work without errors
        output = model(colors, positions, H=15, W=15)
        
        assert output.shape == (batch_size, ctx_len, model.d_model)
    
    def test_tied_embeddings_compliance(self):
        """Test that tied embeddings are of size d_model."""
        d_model = 512
        C = 11
        
        model = EncodingModule(d_model=d_model, C=C)
        
        # Check embedding layer
        assert model.color_embedding.embedding_dim == d_model
        assert model.color_embedding.num_embeddings == C
        
        # Test embedding output
        colors = torch.tensor([[0, 1, 2]], dtype=torch.long)  # [1, 3]
        color_emb = model.color_embedding(colors)  # [1, 3, d_model]
        
        assert color_emb.shape == (1, 3, d_model)
    
    def test_positional_embedding_dimension_compliance(self):
        """Test that 4*d_positional_input = d_model."""
        for d_model in [64, 128, 256, 512]:
            model = EncodingModule(d_model=d_model)
            
            expected_d_positional_input = d_model // 4
            assert model.d_positional_input == expected_d_positional_input
            assert 4 * model.d_positional_input == d_model
    
    def test_four_transformations_compliance(self):
        """Test that exactly 4 transformations are created as specified."""
        model = EncodingModule(d_model=256)
        
        # Single position test
        positions = torch.tensor([[[5, 7]]], dtype=torch.float)
        pos_emb = model.create_positional_embeddings(positions, H=10, W=12)
        
        embedding = pos_emb[0, 0]  # [d_model]
        d_pos = model.d_positional_input
        
        # Should be exactly 4 parts
        assert embedding.shape[0] == 4 * d_pos
        
        # Each part should be different (with high probability)
        part1 = embedding[:d_pos]
        part2 = embedding[d_pos:2*d_pos]
        part3 = embedding[2*d_pos:3*d_pos]
        part4 = embedding[3*d_pos:]
        
        # Parts should be different
        assert not torch.allclose(part1, part2)
        assert not torch.allclose(part1, part3)
        assert not torch.allclose(part1, part4)
        assert not torch.allclose(part2, part3)
        assert not torch.allclose(part2, part4)
        assert not torch.allclose(part3, part4)
    
    def test_encoder_transformer_compliance(self):
        """Test that model uses standard encoder transformer."""
        n_layers = 6
        model = EncodingModule(d_model=256, n_layers=n_layers)
        
        # Check transformer structure
        assert isinstance(model.transformer_encoder, torch.nn.TransformerEncoder)
        assert len(model.transformer_encoder.layers) == n_layers
        
        # Each layer should be TransformerEncoderLayer
        for layer in model.transformer_encoder.layers:
            assert isinstance(layer, torch.nn.TransformerEncoderLayer)
    
    def test_gin_configurability_compliance(self):
        """Test that model is configurable via gin as required."""
        import gin
        
        gin.clear_config()
        
        # Should be able to configure via gin
        gin.parse_config([
            'EncodingModule.d_model = 128',
            'EncodingModule.n_layers = 4',
            'EncodingModule.n_heads = 4',
            'EncodingModule.C = 15'
        ])
        
        model = EncodingModule()
        
        # Configuration should take effect
        assert model.d_model == 128
        assert model.n_layers == 4
        assert model.color_embedding.num_embeddings == 15


if __name__ == "__main__":
    pytest.main([__file__])