"""
Hierarchical VAE (HVAE) for Concept Manifold.

Implements multi-level latent variable structure for hierarchical knowledge representation:
- Directory (768d) → Page (256d) → Line (128d) → Phrase (64d) → Word (32d)
- Parent-child conditional encoding
- KL annealing and Free-Bits for training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import math


class HierarchicalVAE(nn.Module):
    """
    Hierarchical VAE for concept manifold with multi-level latent variables.
    
    Implements a hierarchy: Directory → Page → Line → Phrase → Word
    Each level conditions on its parent level.
    
    Args:
        input_dim: Input dimension (e.g., 768 for BERT-like models)
        levels: List of (name, dimension) tuples defining hierarchy
        kl_weight: Initial KL divergence weight
        free_bits: Free bits threshold for KL
        use_alignment_loss: Whether to use parent-child alignment loss
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        levels: Optional[List[Tuple[str, int]]] = None,
        kl_weight: float = 1e-4,
        free_bits: float = 0.5,
        use_alignment_loss: bool = True,
    ):
        super().__init__()
        
        # Default hierarchy: Directory → Page → Line → Phrase → Word
        if levels is None:
            levels = [
                ('directory', 768),
                ('page', 256),
                ('line', 128),
                ('phrase', 64),
                ('word', 32),
            ]
            
        self.input_dim = input_dim
        self.levels = levels
        self.num_levels = len(levels)
        self.kl_weight = kl_weight
        self.free_bits = free_bits
        self.use_alignment_loss = use_alignment_loss
        
        # Build encoders and decoders for each level
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        
        for i, (level_name, level_dim) in enumerate(levels):
            # Encoder: q(z_l | z_{l-1}, x)
            if i == 0:
                # First level conditions only on input
                encoder_input_dim = input_dim
            else:
                # Other levels condition on parent
                parent_dim = levels[i-1][1]
                encoder_input_dim = input_dim + parent_dim
                
            self.encoders[level_name] = ConditionalEncoder(
                input_dim=encoder_input_dim,
                latent_dim=level_dim
            )
            
            # Decoder: p(x | z_l, z_{l-1})
            if i == 0:
                decoder_input_dim = level_dim
            else:
                parent_dim = levels[i-1][1]
                decoder_input_dim = level_dim + parent_dim
                
            self.decoders[level_name] = ConditionalDecoder(
                latent_dim=decoder_input_dim,
                output_dim=input_dim
            )
            
        # Prior networks for each level p(z_l | z_{l-1})
        self.priors = nn.ModuleDict()
        for i, (level_name, level_dim) in enumerate(levels):
            if i == 0:
                # First level has standard normal prior
                self.priors[level_name] = StandardNormalPrior(level_dim)
            else:
                # Other levels have conditional prior
                parent_dim = levels[i-1][1]
                self.priors[level_name] = ConditionalPrior(
                    parent_dim=parent_dim,
                    latent_dim=level_dim
                )
                
        # KL annealing schedule
        self.register_buffer('_step', torch.tensor(0))
        self.annealing_steps = 10000
        
    def encode_level(
        self,
        x: torch.Tensor,
        level: int,
        parent_z: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to a specific level.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            level: Level index to encode to
            parent_z: Parent level latent (optional)
            
        Returns:
            mu: Mean of posterior [batch_size, seq_len, level_dim]
            log_var: Log variance of posterior [batch_size, seq_len, level_dim]
        """
        level_name = self.levels[level][0]
        encoder = self.encoders[level_name]
        
        if level == 0 or parent_z is None:
            # First level or no parent
            encoder_input = x
        else:
            # Condition on parent
            encoder_input = torch.cat([x, parent_z], dim=-1)
            
        mu, log_var = encoder(encoder_input)
        return mu, log_var
        
    def sample_level(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Sample from latent distribution using reparameterization trick.
        
        Args:
            mu: Mean [batch_size, seq_len, latent_dim]
            log_var: Log variance [batch_size, seq_len, latent_dim]
            temperature: Sampling temperature
            
        Returns:
            Sampled latent [batch_size, seq_len, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * log_var) * temperature
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
            
    def decode_level(
        self,
        z: torch.Tensor,
        level: int,
        parent_z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode from a specific level.
        
        Args:
            z: Latent variable [batch_size, seq_len, level_dim]
            level: Level index
            parent_z: Parent level latent (optional)
            
        Returns:
            Reconstructed output [batch_size, seq_len, input_dim]
        """
        level_name = self.levels[level][0]
        decoder = self.decoders[level_name]
        
        if level == 0 or parent_z is None:
            decoder_input = z
        else:
            decoder_input = torch.cat([z, parent_z], dim=-1)
            
        return decoder(decoder_input)
        
    def compute_kl_divergence(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        prior_mu: Optional[torch.Tensor] = None,
        prior_log_var: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute KL divergence with optional conditional prior.
        
        Args:
            mu: Posterior mean
            log_var: Posterior log variance
            prior_mu: Prior mean (None for standard normal)
            prior_log_var: Prior log variance (None for standard normal)
            
        Returns:
            KL divergence [batch_size, seq_len]
        """
        if prior_mu is None:
            # KL with standard normal
            kl = -0.5 * torch.sum(
                1 + log_var - mu.pow(2) - log_var.exp(),
                dim=-1
            )
        else:
            # KL between two Gaussians
            kl = 0.5 * torch.sum(
                prior_log_var - log_var +
                (log_var.exp() + (mu - prior_mu).pow(2)) / prior_log_var.exp() - 1,
                dim=-1
            )
            
        # Apply free bits
        kl = torch.maximum(kl, torch.tensor(self.free_bits, device=kl.device))
        
        return kl
        
    def forward(
        self,
        x: torch.Tensor,
        target_level: Optional[int] = None,
        return_all_levels: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HVAE.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            target_level: Specific level to encode to (None for all)
            return_all_levels: Whether to return all level outputs
            
        Returns:
            Dictionary containing:
                - 'recon': Reconstructed output
                - 'kl_loss': Total KL divergence loss
                - 'latents': Dict of latent variables per level
                - 'losses': Dict of losses per level
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Storage for outputs
        latents = {}
        mus = {}
        log_vars = {}
        kl_losses = {}
        recon_losses = {}
        
        # Forward through hierarchy
        parent_z = None
        for i, (level_name, level_dim) in enumerate(self.levels):
            # Skip if we only want a specific level
            if target_level is not None and i != target_level:
                continue
                
            # Encode
            mu, log_var = self.encode_level(x, i, parent_z)
            mus[level_name] = mu
            log_vars[level_name] = log_var
            
            # Sample
            z = self.sample_level(mu, log_var)
            latents[level_name] = z
            
            # Compute KL divergence
            if i == 0:
                # Standard normal prior
                kl = self.compute_kl_divergence(mu, log_var)
            else:
                # Conditional prior
                parent_name = self.levels[i-1][0]
                prior = self.priors[level_name]
                prior_mu, prior_log_var = prior(latents[parent_name])
                kl = self.compute_kl_divergence(mu, log_var, prior_mu, prior_log_var)
                
            kl_losses[level_name] = kl.mean()
            
            # Update parent for next level
            parent_z = z
            
        # Decode from the deepest level (or target level)
        if target_level is not None:
            decode_level = target_level
            decode_name = self.levels[target_level][0]
        else:
            decode_level = self.num_levels - 1
            decode_name = self.levels[-1][0]
            
        # Hierarchical decoding (from deepest to shallowest)
        recon = self.decode_level(latents[decode_name], decode_level)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # Total KL loss with annealing
        kl_weight = self.get_annealed_kl_weight()
        total_kl = sum(kl_losses.values()) * kl_weight
        
        # Alignment loss between parent-child pairs
        alignment_loss = torch.tensor(0.0, device=device)
        if self.use_alignment_loss and len(latents) > 1:
            for i in range(1, len(self.levels)):
                if i > len(latents) - 1:
                    break
                parent_name = self.levels[i-1][0]
                child_name = self.levels[i][0]
                if parent_name in latents and child_name in latents:
                    # Cosine similarity loss
                    parent_pooled = latents[parent_name].mean(dim=1)
                    child_pooled = latents[child_name].mean(dim=1)
                    
                    # Project child to parent dimension for comparison
                    child_proj = F.linear(
                        child_pooled,
                        torch.randn(self.levels[i-1][1], self.levels[i][1], device=device) / math.sqrt(self.levels[i][1])
                    )
                    
                    cos_sim = F.cosine_similarity(parent_pooled, child_proj, dim=-1)
                    alignment_loss = alignment_loss + (1 - cos_sim).mean()
                    
        # Total loss
        total_loss = recon_loss + total_kl + 0.1 * alignment_loss
        
        # Update step counter
        if self.training:
            self._step += 1
            
        outputs = {
            'recon': recon,
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': total_kl,
            'alignment_loss': alignment_loss,
            'latents': latents,
            'kl_losses': kl_losses,
        }
        
        return outputs
        
    def get_annealed_kl_weight(self) -> float:
        """Get KL weight with annealing schedule."""
        if self._step < self.annealing_steps:
            # Linear annealing
            return self.kl_weight * (self._step.float() / self.annealing_steps)
        else:
            return self.kl_weight
            
    def hierarchical_search(
        self,
        query: torch.Tensor,
        start_level: int = 0
    ) -> List[Dict]:
        """
        Perform hierarchical search from coarse to fine.
        
        Args:
            query: Query vector [batch_size, input_dim]
            start_level: Level to start search from
            
        Returns:
            List of search results at each level
        """
        results = []
        
        # Start from specified level and go deeper
        for i in range(start_level, self.num_levels):
            level_name = self.levels[i][0]
            
            # Encode query to this level
            if i == 0:
                parent_z = None
            else:
                # Use result from previous level as parent
                parent_z = results[-1]['latent']
                
            mu, log_var = self.encode_level(query.unsqueeze(1), i, parent_z)
            z = self.sample_level(mu, log_var, temperature=0)  # Deterministic
            
            results.append({
                'level': level_name,
                'latent': z.squeeze(1),
                'dimension': self.levels[i][1],
            })
            
        return results


class ConditionalEncoder(nn.Module):
    """Conditional encoder for a single level."""
    
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        hidden_dim = (input_dim + latent_dim) // 2
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_var_head = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.network(x)
        mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-10, max=10)
        return mu, log_var


class ConditionalDecoder(nn.Module):
    """Conditional decoder for a single level."""
    
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        hidden_dim = (latent_dim + output_dim) // 2
        
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class StandardNormalPrior(nn.Module):
    """Standard normal prior for top level."""
    
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        
    def forward(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = torch.zeros(batch_size, self.latent_dim, device=device)
        log_var = torch.zeros(batch_size, self.latent_dim, device=device)
        return mu, log_var


class ConditionalPrior(nn.Module):
    """Conditional prior p(z_l | z_{l-1})."""
    
    def __init__(self, parent_dim: int, latent_dim: int):
        super().__init__()
        hidden_dim = (parent_dim + latent_dim) // 2
        
        self.network = nn.Sequential(
            nn.Linear(parent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_var_head = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, parent_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.network(parent_z)
        mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        log_var = torch.clamp(log_var, min=-10, max=10)
        return mu, log_var
