"""
Flowing Context (FC) Module: Scan→Focus mechanism for efficient attention.

This module implements a two-stage attention mechanism:
1. Scan: O(N) complexity scan to identify important segments
2. Focus: Apply attention bias to selected segments and limit costly operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math


class FlowingContext(nn.Module):
    """
    Flowing Context module that implements Scan→Focus attention mechanism.
    
    Args:
        hidden_size: Dimension of hidden states
        num_segments: Maximum number of segments to focus on (k_seg)
        scan_type: Type of scanner ('bigru', 'conv', 'hybrid')
        beta: Initial attention bias strength
        tau: Temperature for soft masking
        lambda_init: Initial attention bias coefficient
        min_segment_distance: Minimum distance between selected segments
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_segments: int = 4,
        scan_type: str = "bigru",
        beta: float = 10.0,
        tau: float = 0.65,
        lambda_init: float = 0.1,
        min_segment_distance: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_segments = num_segments
        self.beta = beta
        self.tau = tau
        self.min_segment_distance = min_segment_distance
        
        # Learnable lambda coefficient for attention bias annealing
        self.lambda_coef = nn.Parameter(torch.tensor(lambda_init))
        
        # Scanner network
        if scan_type == "bigru":
            self.scanner = BiGRUScanner(hidden_size)
        elif scan_type == "conv":
            self.scanner = ConvScanner(hidden_size)
        elif scan_type == "hybrid":
            self.scanner = HybridScanner(hidden_size)
        else:
            raise ValueError(f"Unknown scan_type: {scan_type}")
            
        # Segment selection with NMS
        self.segment_selector = TopKSegments(
            num_segments=num_segments,
            min_distance=min_segment_distance
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Forward pass of Flowing Context.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            return_diagnostics: Whether to return diagnostic information
            
        Returns:
            soft_mask: Soft attention mask [batch_size, seq_len]
            segments: List of selected segment indices for each batch
            attention_bias: Attention bias to be added to logits [batch_size, 1, seq_len, seq_len]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Stage 1: Scan - get relevance scores
        relevance_scores = self.scanner(hidden_states, attention_mask)  # [B, L]
        
        # Apply temperature-controlled sigmoid for soft masking
        soft_mask = torch.sigmoid((relevance_scores - 0.5) / self.tau)  # [B, L]
        
        # Stage 2: Select top-k segments with NMS
        segments = []
        for b in range(batch_size):
            mask = attention_mask[b] if attention_mask is not None else None
            selected_indices = self.segment_selector(relevance_scores[b], mask)
            segments.append(selected_indices)
        
        # Stage 3: Create attention bias
        attention_bias = self._create_attention_bias(
            soft_mask, segments, seq_len
        )
        
        if return_diagnostics:
            diagnostics = {
                "relevance_scores": relevance_scores,
                "soft_mask": soft_mask,
                "segments": segments,
                "lambda_coef": self.lambda_coef.item(),
                "mean_mask": soft_mask.mean().item(),
            }
            return soft_mask, segments, attention_bias, diagnostics
            
        return soft_mask, segments, attention_bias
    
    def _create_attention_bias(
        self,
        soft_mask: torch.Tensor,
        segments: List[torch.Tensor],
        seq_len: int
    ) -> torch.Tensor:
        """Create attention bias matrix from soft mask and selected segments."""
        batch_size = soft_mask.shape[0]
        device = soft_mask.device
        
        # Initialize bias matrix
        bias = torch.zeros(batch_size, 1, seq_len, seq_len, device=device)
        
        # Apply monotonic mapping: lambda * g(m) where g is a monotonic function
        # Here we use g(m) = beta * m^2 for stronger emphasis on high-relevance tokens
        enhanced_mask = self.beta * (soft_mask ** 2)
        
        # Add bias for each segment
        for b, segment_indices in enumerate(segments):
            if len(segment_indices) > 0:
                # Create segment mask
                segment_mask = torch.zeros(seq_len, device=device)
                for idx in segment_indices:
                    # Apply Gaussian-like weight around segment center
                    distances = torch.abs(torch.arange(seq_len, device=device) - idx)
                    weights = torch.exp(-distances.float() / 8.0)  # Decay parameter
                    segment_mask += weights
                    
                segment_mask = torch.clamp(segment_mask, 0, 1)
                
                # Combine with soft mask
                combined_mask = enhanced_mask[b] * segment_mask
                
                # Apply to bias matrix (both as query and key positions)
                bias[b, 0, :, :] = self.lambda_coef * combined_mask.unsqueeze(0)
                
        return bias


class BiGRUScanner(nn.Module):
    """Bidirectional GRU scanner for relevance scoring."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(
            hidden_size, 
            hidden_size // 2,
            batch_first=True,
            bidirectional=True
        )
        self.projection = nn.Linear(hidden_size, 1)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Run BiGRU
        gru_out, _ = self.gru(hidden_states)  # [B, L, H]
        
        # Project to relevance scores
        relevance = self.projection(gru_out).squeeze(-1)  # [B, L]
        
        # Apply mask if provided
        if attention_mask is not None:
            relevance = relevance.masked_fill(~attention_mask.bool(), -1e9)
            
        return torch.sigmoid(relevance)


class ConvScanner(nn.Module):
    """Convolutional scanner for local pattern detection."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_size // 2, 1, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Transpose for conv1d [B, L, H] -> [B, H, L]
        x = hidden_states.transpose(1, 2)
        
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        relevance = self.conv3(x).squeeze(1)  # [B, L]
        
        if attention_mask is not None:
            relevance = relevance.masked_fill(~attention_mask.bool(), -1e9)
            
        return torch.sigmoid(relevance)


class HybridScanner(nn.Module):
    """Hybrid scanner combining BiGRU and Conv for both sequential and local patterns."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gru_scanner = BiGRUScanner(hidden_size)
        self.conv_scanner = ConvScanner(hidden_size)
        self.fusion = nn.Linear(2, 1)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        gru_scores = self.gru_scanner(hidden_states, attention_mask)
        conv_scores = self.conv_scanner(hidden_states, attention_mask)
        
        # Fusion via learned weighted average
        combined = torch.stack([gru_scores, conv_scores], dim=-1)  # [B, L, 2]
        relevance = self.fusion(combined).squeeze(-1)  # [B, L]
        
        return torch.sigmoid(relevance)


class TopKSegments(nn.Module):
    """Select top-k segments with Non-Maximum Suppression (NMS)."""
    
    def __init__(self, num_segments: int = 4, min_distance: int = 16):
        super().__init__()
        self.num_segments = num_segments
        self.min_distance = min_distance
        
    def forward(
        self,
        scores: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Select top-k segments with NMS.
        
        Args:
            scores: Relevance scores [seq_len]
            mask: Valid position mask [seq_len]
            
        Returns:
            Selected segment indices
        """
        device = scores.device
        seq_len = scores.shape[0]
        
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -float('inf'))
            
        selected = []
        scores_copy = scores.clone()
        
        for _ in range(self.num_segments):
            # Find maximum
            max_idx = torch.argmax(scores_copy)
            
            if scores_copy[max_idx] == -float('inf'):
                break
                
            selected.append(max_idx.item())
            
            # Suppress nearby positions
            start = max(0, max_idx - self.min_distance)
            end = min(seq_len, max_idx + self.min_distance + 1)
            scores_copy[start:end] = -float('inf')
            
        return torch.tensor(selected, device=device, dtype=torch.long)
