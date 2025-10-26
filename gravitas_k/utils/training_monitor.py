"""
Training Monitor for Gravitas-K

Provides real-time monitoring and visualization of training metrics:
- FlowingContext: segment hit rates, attention bias distributions
- CCB: expert activation frequencies, workflow path statistics
- PRC: trigger positions, σ/γ distributions
- HVAE: KL divergence per level, Free-Bits utilization
- Dual-Stream: α(s) distributions, anchor vs emergence norms
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import torch
import numpy as np
from pathlib import Path


class TrainingMonitor:
    """
    Real-time training monitor with metrics logging and visualization.
    
    Integrates with TensorBoard and W&B for comprehensive tracking.
    """
    
    def __init__(
        self,
        log_dir: str = './logs',
        run_name: Optional[str] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        window_size: int = 100,  # Moving average window
    ):
        """
        Args:
            log_dir: Directory to save logs
            run_name: Name of this training run
            use_tensorboard: Whether to use TensorBoard logging
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
            window_size: Window size for moving averages
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_name = run_name or f"run_{int(time.time())}"
        self.run_dir = self.log_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.window_size = window_size
        
        # Metric storage with moving averages
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.global_step = 0
        
        # Initialize loggers
        self.tb_writer = None
        self.wandb_run = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / 'tensorboard'))
                print(f"✓ TensorBoard logging enabled: {self.run_dir / 'tensorboard'}")
            except ImportError:
                print("⚠ TensorBoard not installed. Install with: pip install tensorboard")
        
        if use_wandb and wandb_project:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=self.run_name,
                    dir=str(self.run_dir),
                )
                print(f"✓ W&B logging enabled: {wandb_project}/{self.run_name}")
            except ImportError:
                print("⚠ W&B not installed. Install with: pip install wandb")
    
    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """Log a scalar metric."""
        if step is None:
            step = self.global_step
        
        self.metrics[name].append(value)
        
        if self.tb_writer:
            self.tb_writer.add_scalar(name, value, step)
        
        if self.wandb_run:
            self.wandb_run.log({name: value}, step=step)
    
    def log_histogram(self, name: str, values: torch.Tensor, step: Optional[int] = None):
        """Log a histogram of values."""
        if step is None:
            step = self.global_step
        
        if self.tb_writer:
            self.tb_writer.add_histogram(name, values.cpu().numpy(), step)
        
        if self.wandb_run:
            import wandb
            self.wandb_run.log({name: wandb.Histogram(values.cpu().numpy())}, step=step)
    
    def log_image(self, name: str, image: torch.Tensor, step: Optional[int] = None):
        """Log an image (e.g., attention heatmap)."""
        if step is None:
            step = self.global_step
        
        if self.tb_writer:
            self.tb_writer.add_image(name, image, step)
        
        if self.wandb_run:
            import wandb
            self.wandb_run.log({name: wandb.Image(image.cpu().numpy())}, step=step)
    
    def log_flowing_context_metrics(
        self,
        m_i: torch.Tensor,  # Soft mask
        S: torch.Tensor,  # Selected segments
        B_attn_fc: torch.Tensor,  # Attention bias
        step: Optional[int] = None,
    ):
        """
        Log FlowingContext-specific metrics.
        
        Args:
            m_i: Soft segment importance mask (batch, seq_len)
            S: Selected segment mask (batch, seq_len)
            B_attn_fc: Attention bias (batch, 1, 1, seq_len)
        """
        if step is None:
            step = self.global_step
        
        # Segment hit rate
        hit_rate = S.float().mean().item()
        self.log_scalar('fc/segment_hit_rate', hit_rate, step)
        
        # Attention bias statistics
        bias_mean = B_attn_fc.mean().item()
        bias_std = B_attn_fc.std().item()
        bias_max = B_attn_fc.max().item()
        
        self.log_scalar('fc/bias_mean', bias_mean, step)
        self.log_scalar('fc/bias_std', bias_std, step)
        self.log_scalar('fc/bias_max', bias_max, step)
        
        # Segment importance distribution
        self.log_histogram('fc/segment_importance', m_i, step)
    
    def log_ccb_metrics(
        self,
        workflow_plan: Dict[str, Any],
        expert_outputs: Dict[str, Optional[torch.Tensor]],
        step: Optional[int] = None,
    ):
        """
        Log CCB (A-P-C-V-S) metrics.
        
        Args:
            workflow_plan: WorkflowPlan object with task type, confidence, controls
            expert_outputs: Dictionary of expert outputs
        """
        if step is None:
            step = self.global_step
        
        # Task type distribution
        if hasattr(workflow_plan, 'task_type'):
            self.log_scalar(f'ccb/task_type/{workflow_plan.task_type}', 1.0, step)
        
        # Confidence score
        if hasattr(workflow_plan, 'confidence'):
            self.log_scalar('ccb/confidence', workflow_plan.confidence, step)
        
        # Expert activation
        for expert_name, output in expert_outputs.items():
            if output is not None:
                self.log_scalar(f'ccb/expert_active/{expert_name}', 1.0, step)
                # Log expert output norms
                output_norm = output.norm(dim=-1).mean().item()
                self.log_scalar(f'ccb/expert_norm/{expert_name}', output_norm, step)
        
        # Control parameters
        if hasattr(workflow_plan, 'sigma_ctrl'):
            self.log_scalar('ccb/sigma_ctrl', workflow_plan.sigma_ctrl, step)
        if hasattr(workflow_plan, 'gamma_ctrl'):
            self.log_scalar('ccb/gamma_ctrl', workflow_plan.gamma_ctrl, step)
        if hasattr(workflow_plan, 'fc_strength'):
            self.log_scalar('ccb/fc_strength', workflow_plan.fc_strength, step)
        if hasattr(workflow_plan, 'alpha_dual'):
            self.log_scalar('ccb/alpha_dual', workflow_plan.alpha_dual, step)
    
    def log_prc_metrics(
        self,
        trigger_mask: torch.Tensor,
        sampled_z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        step: Optional[int] = None,
    ):
        """
        Log PRC (Probabilistic Region Collapse) metrics.
        
        Args:
            trigger_mask: Boolean mask of triggered positions
            sampled_z: Sampled latent vectors
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        if step is None:
            step = self.global_step
        
        # Trigger rate
        trigger_rate = trigger_mask.float().mean().item()
        self.log_scalar('prc/trigger_rate', trigger_rate, step)
        
        # Latent statistics
        z_norm = sampled_z.norm(dim=-1).mean().item()
        mu_norm = mu.norm(dim=-1).mean().item()
        std = torch.exp(0.5 * logvar).mean().item()
        
        self.log_scalar('prc/z_norm', z_norm, step)
        self.log_scalar('prc/mu_norm', mu_norm, step)
        self.log_scalar('prc/std', std, step)
        
        # Distribution of sampled vectors
        self.log_histogram('prc/sampled_z', sampled_z, step)
    
    def log_hvae_metrics(
        self,
        latents: List[torch.Tensor],
        mus: List[torch.Tensor],
        logvars: List[torch.Tensor],
        kl_losses: List[float],
        step: Optional[int] = None,
    ):
        """
        Log HVAE (Hierarchical VAE) metrics.
        
        Args:
            latents: List of latent vectors per level
            mus: List of means per level
            logvars: List of log variances per level
            kl_losses: List of KL divergence per level
        """
        if step is None:
            step = self.global_step
        
        level_names = ['directory', 'page', 'line', 'phrase', 'word']
        
        for i, (latent, mu, logvar, kl) in enumerate(zip(latents, mus, logvars, kl_losses)):
            level_name = level_names[i] if i < len(level_names) else f'level_{i}'
            
            # KL divergence
            self.log_scalar(f'hvae/kl_{level_name}', kl, step)
            
            # Latent norms
            latent_norm = latent.norm(dim=-1).mean().item()
            mu_norm = mu.norm(dim=-1).mean().item()
            
            self.log_scalar(f'hvae/latent_norm_{level_name}', latent_norm, step)
            self.log_scalar(f'hvae/mu_norm_{level_name}', mu_norm, step)
            
            # Posterior collapse detection
            std = torch.exp(0.5 * logvar).mean().item()
            self.log_scalar(f'hvae/std_{level_name}', std, step)
            
            # Free bits utilization
            if kl > 0:
                free_bits_util = min(kl / 0.5, 1.0)  # Assuming 0.5 free bits
                self.log_scalar(f'hvae/free_bits_util_{level_name}', free_bits_util, step)
    
    def log_dual_stream_metrics(
        self,
        h_anchor: torch.Tensor,
        h_emergence: torch.Tensor,
        alpha: torch.Tensor,
        step: Optional[int] = None,
    ):
        """
        Log Dual-Stream metrics.
        
        Args:
            h_anchor: Anchor stream hidden states
            h_emergence: Emergence stream hidden states
            alpha: Mixing coefficient α(s)
        """
        if step is None:
            step = self.global_step
        
        # Stream norms
        anchor_norm = h_anchor.norm(dim=-1).mean().item()
        emergence_norm = h_emergence.norm(dim=-1).mean().item()
        
        self.log_scalar('dual_stream/anchor_norm', anchor_norm, step)
        self.log_scalar('dual_stream/emergence_norm', emergence_norm, step)
        
        # Mixing coefficient statistics
        alpha_mean = alpha.mean().item()
        alpha_std = alpha.std().item()
        
        self.log_scalar('dual_stream/alpha_mean', alpha_mean, step)
        self.log_scalar('dual_stream/alpha_std', alpha_std, step)
        
        # Distribution of alpha
        self.log_histogram('dual_stream/alpha', alpha, step)
    
    def log_training_step(
        self,
        loss: float,
        learning_rate: float,
        grad_norm: Optional[float] = None,
        step: Optional[int] = None,
    ):
        """Log basic training metrics."""
        if step is None:
            step = self.global_step
        
        self.log_scalar('train/loss', loss, step)
        self.log_scalar('train/learning_rate', learning_rate, step)
        
        if grad_norm is not None:
            self.log_scalar('train/grad_norm', grad_norm, step)
        
        self.global_step += 1
    
    def get_moving_average(self, name: str) -> Optional[float]:
        """Get moving average of a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None
        return float(np.mean(self.metrics[name]))
    
    def save_checkpoint_metrics(self, checkpoint_path: str, metrics: Dict[str, Any]):
        """Save metrics alongside a checkpoint."""
        metrics_path = Path(checkpoint_path).with_suffix('.metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def close(self):
        """Close all loggers."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            self.wandb_run.finish()

