"""
Probabilistic Region Collapse (PRC): Prior injection via kNN and sampling.

This module implements concept space sampling through:
1. kNN search in concept space (W_out or HVAE latent space)
2. Statistical estimation (μ, σ) from neighbors
3. Reparameterized sampling
4. Gated fusion with main stream
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
import numpy as np
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. PRC will use PyTorch-based kNN (slower).")


class ProbabilisticRegionCollapse(nn.Module):
    """
    PRC module for controlled prior injection from concept space.
    
    Args:
        concept_dim: Dimension of concept space
        k_neighbors: Number of nearest neighbors for statistics
        tau_prime: Initial triggering threshold
        sigma_max: Maximum sampling variance
        gamma_init: Initial gating coefficient
        use_faiss: Whether to use FAISS for kNN search
        index_type: FAISS index type ('flat', 'ivf', 'opq')
    """
    
    def __init__(
        self,
        concept_dim: int,
        k_neighbors: int = 5,
        tau_prime: float = 0.7,
        sigma_max: float = 0.5,
        gamma_init: float = 0.3,
        use_faiss: bool = True,
        index_type: str = "flat",
    ):
        super().__init__()
        self.concept_dim = concept_dim
        self.k_neighbors = k_neighbors
        self.tau_prime = tau_prime
        self.sigma_max = sigma_max
        
        # Learnable gating parameter
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        
        # Concept space (can be W_out or HVAE space)
        self.concept_bank = None
        self.index = None
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.index_type = index_type
        
        # Statistics networks
        self.mu_proj = nn.Linear(concept_dim * k_neighbors, concept_dim)
        self.sigma_proj = nn.Linear(concept_dim * k_neighbors, concept_dim)
        
        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, concept_dim),
            nn.Sigmoid()
        )
        
    def build_index(self, concept_bank: torch.Tensor):
        """
        Build search index from concept bank.
        
        Args:
            concept_bank: Concept vectors [num_concepts, concept_dim]
        """
        self.concept_bank = concept_bank.detach()
        
        if self.use_faiss:
            # Build FAISS index
            vectors = concept_bank.cpu().numpy().astype('float32')
            
            if self.index_type == "flat":
                self.index = faiss.IndexFlatL2(self.concept_dim)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.concept_dim)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.concept_dim, min(100, vectors.shape[0] // 10)
                )
                self.index.train(vectors)
            elif self.index_type == "opq":
                # OPQ for better compression
                M = min(48, self.concept_dim // 2)  # Sub-vector count
                self.index = faiss.IndexOPQ(self.concept_dim, M, 8)
                self.index.train(vectors)
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
                
            self.index.add(vectors)
        else:
            # Will use PyTorch cdist for kNN
            pass
            
    def search_neighbors(
        self,
        queries: torch.Tensor,
        k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search k nearest neighbors.
        
        Args:
            queries: Query vectors [batch_size, seq_len, concept_dim]
            k: Number of neighbors (default: self.k_neighbors)
            
        Returns:
            distances: Distance to neighbors [batch_size, seq_len, k]
            indices: Neighbor indices [batch_size, seq_len, k]
        """
        if k is None:
            k = self.k_neighbors
            
        batch_size, seq_len, _ = queries.shape
        queries_flat = queries.reshape(-1, self.concept_dim)
        
        if self.use_faiss and self.index is not None:
            # FAISS search
            queries_np = queries_flat.cpu().numpy().astype('float32')
            distances_np, indices_np = self.index.search(queries_np, k)
            
            distances = torch.from_numpy(distances_np).to(queries.device)
            indices = torch.from_numpy(indices_np).to(queries.device)
        else:
            # PyTorch cdist-based search
            if self.concept_bank is None:
                raise ValueError("Concept bank not initialized")
                
            dists = torch.cdist(queries_flat, self.concept_bank)  # [B*L, num_concepts]
            distances, indices = torch.topk(dists, k, dim=-1, largest=False)
            
        # Reshape back
        distances = distances.reshape(batch_size, seq_len, k)
        indices = indices.reshape(batch_size, seq_len, k)
        
        return distances, indices
        
    def estimate_statistics(
        self,
        queries: torch.Tensor,
        neighbors: torch.Tensor,
        distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate (μ, σ) from neighbors.
        
        Args:
            queries: Query vectors [batch_size, seq_len, concept_dim]
            neighbors: Neighbor vectors [batch_size, seq_len, k, concept_dim]
            distances: Distances to neighbors [batch_size, seq_len, k]
            
        Returns:
            mu: Mean vector [batch_size, seq_len, concept_dim]
            sigma: Std vector [batch_size, seq_len, concept_dim]
        """
        batch_size, seq_len, k, dim = neighbors.shape
        
        # Flatten neighbors for projection
        neighbors_flat = neighbors.reshape(batch_size, seq_len, k * dim)
        
        # Estimate mu and sigma through learned projections
        mu = self.mu_proj(neighbors_flat)
        sigma_raw = self.sigma_proj(neighbors_flat)
        
        # Constrain sigma to reasonable range
        sigma = torch.sigmoid(sigma_raw) * self.sigma_max
        
        # Weight by inverse distance
        weights = F.softmax(-distances, dim=-1).unsqueeze(-1)  # [B, L, k, 1]
        
        # Weighted average as alternative mu
        weighted_mu = (neighbors * weights).sum(dim=2)
        
        # Combine learned and weighted estimates
        mu = 0.5 * mu + 0.5 * weighted_mu
        
        return mu, sigma
        
    def reparameterized_sample(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterized sampling from N(μ, σ).
        
        Args:
            mu: Mean [batch_size, seq_len, concept_dim]
            sigma: Std [batch_size, seq_len, concept_dim]
            
        Returns:
            Sampled vectors [batch_size, seq_len, concept_dim]
        """
        if self.training:
            # Reparameterization trick
            eps = torch.randn_like(mu)
            return mu + sigma * eps
        else:
            # Use mean during inference for stability
            return mu
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        trigger_mask: Optional[torch.Tensor] = None,
        sigma_ctrl: float = 1.0,
        gamma_ctrl: float = 1.0,
        level: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass of PRC.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            trigger_mask: Binary mask for triggering PRC [batch_size, seq_len]
            sigma_ctrl: External control for sampling variance
            gamma_ctrl: External control for gating strength
            level: HVAE level for hierarchical sampling (optional)
            
        Returns:
            output: Fused output with prior injection
            diagnostics: Diagnostic information
        """
        if self.concept_bank is None:
            # No concept bank, return input unchanged
            return hidden_states, {'triggered': False}
            
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Determine which positions to trigger
        if trigger_mask is None:
            # Default: trigger based on attention scores or confidence
            trigger_mask = torch.ones(batch_size, seq_len, device=device) > (1 - self.tau_prime)
            
        # Only process triggered positions
        num_triggered = trigger_mask.sum().item()
        if num_triggered == 0:
            return hidden_states, {'triggered': False, 'num_triggered': 0}
            
        # Project to concept space if needed (use persistent projection layers)
        if not hasattr(self, 'to_concept'):
            self.to_concept = nn.Linear(hidden_size, self.concept_dim, bias=False)
            self.from_concept = nn.Linear(self.concept_dim, hidden_size, bias=False)
        
        if hidden_size != self.concept_dim:
            queries = self.to_concept(hidden_states)
        else:
            queries = hidden_states
            
        # Search neighbors
        distances, indices = self.search_neighbors(queries)
        
        # Gather neighbor vectors
        neighbors = self.concept_bank[indices]  # [B, L, k, D]
        
        # Estimate statistics
        mu, sigma = self.estimate_statistics(queries, neighbors, distances)
        
        # Apply external control
        sigma = sigma * sigma_ctrl
        
        # Sample
        samples = self.reparameterized_sample(mu, sigma)
        
        # Project back if needed
        if hidden_size != self.concept_dim:
            samples = self.from_concept(samples)
            
        # Gated fusion
        gate_input = torch.cat([hidden_states, samples], dim=-1)
        gate = self.gate_net(gate_input) * self.gamma * gamma_ctrl
        
        # Apply gating only at triggered positions
        gate = gate * trigger_mask.unsqueeze(-1)
        
        # Fuse
        output = hidden_states + gate * (samples - hidden_states)
        
        diagnostics = {
            'triggered': True,
            'num_triggered': num_triggered,
            'mean_distance': distances.mean().item(),
            'mean_sigma': sigma.mean().item(),
            'mean_gate': gate.mean().item(),
        }
        
        return output, diagnostics


class HierarchicalPRC(nn.Module):
    """
    Hierarchical PRC that works with HVAE at multiple levels.
    
    Args:
        levels: List of (level_name, dimension) tuples
        base_k: Base number of neighbors
        share_stats: Whether to share statistics networks across levels
    """
    
    def __init__(
        self,
        levels: list,
        base_k: int = 5,
        share_stats: bool = False
    ):
        super().__init__()
        self.levels = levels
        self.num_levels = len(levels)
        
        # Create PRC for each level
        self.prcs = nn.ModuleDict()
        for level_name, dim in levels:
            self.prcs[level_name] = ProbabilisticRegionCollapse(
                concept_dim=dim,
                k_neighbors=base_k
            )
            
        # Level selector
        self.level_selector = nn.Linear(levels[0][1], self.num_levels)
        
    def select_level(self, hidden_states: torch.Tensor) -> int:
        """Select which level to use based on hidden states."""
        pooled = hidden_states.mean(dim=1).mean(dim=0)  # Global pooling
        logits = self.level_selector(pooled)
        level_idx = torch.argmax(F.softmax(logits, dim=-1))
        return level_idx.item()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        level: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward with automatic or specified level selection.
        
        Args:
            hidden_states: Input hidden states
            level: Specific level to use (optional)
            **kwargs: Additional arguments for PRC
            
        Returns:
            output: Fused output
            diagnostics: Including selected level
        """
        if level is None:
            level = self.select_level(hidden_states)
            
        level_name = self.levels[level][0]
        prc = self.prcs[level_name]
        
        output, diagnostics = prc(hidden_states, **kwargs)
        diagnostics['selected_level'] = level_name
        
        return output, diagnostics
