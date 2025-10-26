"""Unit tests for Flowing Context module."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from gravitas_k.models.flowing_context import FlowingContext, TopKSegments


class TestFlowingContext:
    """Test cases for Flowing Context module."""
    
    @pytest.fixture
    def fc_module(self):
        """Create a Flowing Context module for testing."""
        return FlowingContext(
            hidden_size=768,
            num_segments=4,
            scan_type="bigru",
            beta=10.0,
            tau=0.65,
        )
        
    def test_initialization(self, fc_module):
        """Test module initialization."""
        assert fc_module.hidden_size == 768
        assert fc_module.num_segments == 4
        assert fc_module.beta == 10.0
        assert fc_module.tau == 0.65
        
    def test_forward_pass(self, fc_module):
        """Test forward pass through FC module."""
        batch_size = 2
        seq_len = 128
        hidden_size = 768
        
        # Create dummy input
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass
        soft_mask, segments, attention_bias = fc_module(
            hidden_states,
            attention_mask,
        )
        
        # Check outputs
        assert soft_mask.shape == (batch_size, seq_len)
        assert len(segments) == batch_size
        assert attention_bias.shape == (batch_size, 1, seq_len, seq_len)
        
    def test_scanner_output_range(self, fc_module):
        """Test that scanner outputs are in valid range."""
        batch_size = 2
        seq_len = 64
        hidden_size = 768
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Get scanner output
        relevance_scores = fc_module.scanner(hidden_states)
        
        # Check range [0, 1] after sigmoid
        assert torch.all(relevance_scores >= 0)
        assert torch.all(relevance_scores <= 1)
        
    def test_segment_selection_nms(self):
        """Test segment selection with NMS."""
        selector = TopKSegments(num_segments=3, min_distance=10)
        
        # Create scores with clear peaks
        scores = torch.tensor([
            0.1, 0.2, 0.9, 0.3, 0.2,  # Peak at 2
            0.1, 0.1, 0.2, 0.3, 0.8,  # Peak at 9
            0.1, 0.1, 0.1, 0.1, 0.1,
            0.7, 0.6, 0.5, 0.4, 0.3,  # Peak at 15
        ])
        
        selected = selector(scores)
        
        # Check that selected segments respect min_distance
        selected_list = selected.tolist()
        for i in range(len(selected_list) - 1):
            assert abs(selected_list[i+1] - selected_list[i]) >= selector.min_distance
            
    def test_attention_bias_creation(self, fc_module):
        """Test attention bias matrix creation."""
        batch_size = 1
        seq_len = 32
        
        soft_mask = torch.rand(batch_size, seq_len)
        segments = [torch.tensor([5, 15, 25])]
        
        bias = fc_module._create_attention_bias(soft_mask, segments, seq_len)
        
        # Check that bias is applied around segments
        assert bias.shape == (batch_size, 1, seq_len, seq_len)
        
        # Bias should be non-zero around segment positions
        for seg_idx in segments[0]:
            assert torch.any(bias[0, 0, :, seg_idx] != 0)
            
    def test_lambda_coefficient_learning(self, fc_module):
        """Test that lambda coefficient is learnable."""
        initial_lambda = fc_module.lambda_coef.item()
        
        # Create dummy loss and backward
        batch_size = 2
        seq_len = 64
        hidden_size = 768
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        _, _, attention_bias = fc_module(hidden_states)
        
        loss = attention_bias.mean()
        loss.backward()
        
        # Check that lambda has gradient
        assert fc_module.lambda_coef.grad is not None
        assert fc_module.lambda_coef.grad != 0
        
    def test_different_scan_types(self):
        """Test different scanner types."""
        hidden_size = 768
        
        for scan_type in ["bigru", "conv", "hybrid"]:
            fc = FlowingContext(
                hidden_size=hidden_size,
                scan_type=scan_type,
            )
            
            # Test forward pass
            hidden_states = torch.randn(1, 32, hidden_size)
            soft_mask, segments, attention_bias = fc(hidden_states)
            
            assert soft_mask is not None
            assert segments is not None
            assert attention_bias is not None


class TestTopKSegments:
    """Test cases for TopK segment selection."""
    
    def test_basic_selection(self):
        """Test basic top-k selection."""
        selector = TopKSegments(num_segments=3, min_distance=5)
        
        scores = torch.rand(50)
        selected = selector(scores)
        
        # Should select at most num_segments
        assert len(selected) <= selector.num_segments
        
    def test_with_mask(self):
        """Test selection with mask."""
        selector = TopKSegments(num_segments=3, min_distance=5)
        
        scores = torch.rand(50)
        mask = torch.zeros(50, dtype=torch.bool)
        mask[:25] = True  # Only first half is valid
        
        selected = selector(scores, mask)
        
        # All selected should be in valid region
        for idx in selected:
            assert idx < 25
            
    def test_empty_selection(self):
        """Test when all scores are masked out."""
        selector = TopKSegments(num_segments=3, min_distance=5)
        
        scores = torch.rand(50)
        mask = torch.zeros(50, dtype=torch.bool)  # All masked
        
        selected = selector(scores, mask)
        
        # Should return empty
        assert len(selected) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
