"""
Cognitive Core Block (CCB): A-P-C-V-S structured reasoning module.

Implements the Arbiter-Proposer-Challenger-Verifier-Synthesizer workflow
for structured cognitive processing, replacing traditional FFN layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class WorkflowPlan:
    """Workflow execution plan from Arbiter."""
    task_type: str  # 'creative', 'logical', 'deliberative'
    confidence: float  # Task confidence score [0, 1]
    enable_challenger: bool
    enable_full_deliberation: bool
    sigma_ctrl: float  # PRC sampling variance control
    gamma_ctrl: float  # PRC gating control
    fc_strength: float  # Flowing Context strength
    alpha_dual: float  # Dual-stream mixing coefficient


class CognitiveCoreBlock(nn.Module):
    """
    Cognitive Core Block implementing A-P-C-V-S workflow.
    
    Replaces FFN layers with structured reasoning components:
    - Arbiter: Plans workflow and controls
    - Proposer: Generates initial proposals
    - Challenger: Questions and provides alternatives
    - Verifier: Validates proposals
    - Synthesizer: Combines outputs
    
    Args:
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        num_experts: Number of expert modules
        dropout: Dropout probability
        enable_dual_stream: Whether to use dual-stream attention
        enable_think_logs: Whether to generate <think> logs
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        num_experts: int = 4,
        dropout: float = 0.1,
        enable_dual_stream: bool = True,
        enable_think_logs: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or hidden_size * 4
        self.num_experts = num_experts
        self.enable_dual_stream = enable_dual_stream
        self.enable_think_logs = enable_think_logs
        
        # Arbiter: Lightweight task classifier and controller
        self.arbiter = Arbiter(hidden_size)
        
        # Proposer: Initial proposal generation (can reuse FFN weights)
        self.proposer = Proposer(hidden_size, self.intermediate_size)
        
        # Challenger: Questions and alternatives
        self.challenger = Challenger(hidden_size, self.intermediate_size // 4)
        
        # Verifier: Validation and fact-checking
        self.verifier = Verifier(hidden_size, self.intermediate_size // 2)
        
        # Synthesizer: Adaptive fusion of all outputs
        self.synthesizer = Synthesizer(hidden_size, num_experts)
        
        # Dual-stream components if enabled
        if enable_dual_stream:
            self.anchor_ema = nn.Parameter(torch.tensor(0.9))  # EMA coefficient
            self.dual_mixer = DualStreamMixer(hidden_size)
            
        # Think log formatter
        if enable_think_logs:
            self.think_formatter = ThinkLogFormatter()
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        h_anchor_in: Optional[torch.Tensor] = None,
        task_meta: Optional[Dict] = None,
        prc_samples: Optional[torch.Tensor] = None,
        fc_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through CCB.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            h_anchor_in: Anchor stream input from previous layer
            task_meta: Task metadata for Arbiter
            prc_samples: PRC prior samples if available
            fc_mask: Flowing Context mask
            
        Returns:
            h_out: Output hidden states
            h_anchor_out: Anchor stream output for next layer
            think_logs: Structured thinking logs (if enabled)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Step 1: Arbiter plans the workflow
        workflow_plan = self.arbiter(hidden_states, task_meta)
        
        # Step 2: Proposer generates initial proposals
        proposals = self.proposer(hidden_states, prc_samples)
        
        # Step 3: Challenger questions (if enabled by Arbiter)
        challenges = None
        if workflow_plan.enable_challenger:
            challenges = self.challenger(hidden_states, proposals)
            
        # Step 4: Verifier validates
        verification = self.verifier(hidden_states, proposals, challenges)
        
        # Step 5: Synthesizer combines all outputs
        expert_outputs = {
            'proposer': proposals,
            'challenger': challenges,
            'verifier': verification,
        }
        synthesized = self.synthesizer(
            hidden_states,
            expert_outputs,
            workflow_plan
        )
        
        # Apply dual-stream if enabled
        h_out = synthesized
        h_anchor_out = hidden_states  # Default passthrough
        
        if self.enable_dual_stream and h_anchor_in is not None:
            h_out, h_anchor_out = self.dual_mixer(
                synthesized,
                h_anchor_in,
                workflow_plan.alpha_dual
            )
            
        # Generate think logs if enabled
        think_logs = None
        if self.enable_think_logs:
            think_logs = self.think_formatter.format(
                workflow_plan,
                expert_outputs,
                verification
            )
            
        return h_out, h_anchor_out, think_logs


class Arbiter(nn.Module):
    """Arbiter module for workflow planning and control."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.task_classifier = nn.Linear(hidden_size, 3)  # creative/logical/deliberative
        self.confidence_head = nn.Linear(hidden_size, 1)
        self.control_head = nn.Linear(hidden_size, 5)  # Various control parameters
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        task_meta: Optional[Dict] = None
    ) -> WorkflowPlan:
        # Pool hidden states for classification
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Classify task type
        task_logits = self.task_classifier(pooled)
        task_probs = F.softmax(task_logits, dim=-1)
        task_idx = torch.argmax(task_probs, dim=-1)
        task_types = ['creative', 'logical', 'deliberative']
        task_type = task_types[task_idx[0].item()]
        
        # Compute confidence
        confidence = torch.sigmoid(self.confidence_head(pooled)).squeeze(-1)
        
        # Generate control parameters
        controls = torch.sigmoid(self.control_head(pooled))
        
        # Create workflow plan
        plan = WorkflowPlan(
            task_type=task_type,
            confidence=confidence[0].item(),
            enable_challenger=confidence[0].item() < 0.7,  # Enable when uncertain
            enable_full_deliberation=task_type == 'deliberative',
            sigma_ctrl=controls[0, 0].item(),
            gamma_ctrl=controls[0, 1].item(),
            fc_strength=controls[0, 2].item(),
            alpha_dual=controls[0, 3].item(),
        )
        
        return plan


class Proposer(nn.Module):
    """Proposer module for initial proposal generation."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # Similar to FFN, can be initialized from original weights
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.SiLU()  # SwiGLU-style activation
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        prc_samples: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Incorporate PRC samples if available
        if prc_samples is not None:
            hidden_states = hidden_states + 0.1 * prc_samples
            
        # SwiGLU-style computation
        gate = self.activation(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        intermediate = gate * up
        output = self.down_proj(intermediate)
        
        return output


class Challenger(nn.Module):
    """Challenger module for questioning and alternatives."""
    
    def __init__(self, hidden_size: int, challenge_size: int):
        super().__init__()
        # Residual network to compute difference
        self.diff_net = nn.Sequential(
            nn.Linear(hidden_size * 2, challenge_size),
            nn.ReLU(),
            nn.Linear(challenge_size, challenge_size),
            nn.ReLU(),
            nn.Linear(challenge_size, hidden_size)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        proposals: torch.Tensor
    ) -> torch.Tensor:
        # Compute difference between proposal and expectation
        combined = torch.cat([hidden_states, proposals], dim=-1)
        challenge = self.diff_net(combined)
        
        # Return alternative path
        return hidden_states + challenge


class Verifier(nn.Module):
    """Verifier module for validation and fact-checking."""
    
    def __init__(self, hidden_size: int, verify_size: int):
        super().__init__()
        self.verify_net = nn.Sequential(
            nn.Linear(hidden_size, verify_size),
            nn.ReLU(),
            nn.Linear(verify_size, verify_size // 2),
            nn.ReLU(),
            nn.Linear(verify_size // 2, hidden_size)
        )
        # Binary verification head
        self.verification_gate = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        proposals: torch.Tensor,
        challenges: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Verify proposals
        to_verify = proposals
        if challenges is not None:
            # Also consider challenges
            to_verify = proposals + 0.5 * challenges
            
        verified = self.verify_net(to_verify)
        
        # Apply gating based on verification
        gate = torch.sigmoid(self.verification_gate(verified))
        output = gate * verified + (1 - gate) * hidden_states
        
        return output


class Synthesizer(nn.Module):
    """Synthesizer module for adaptive fusion of expert outputs."""
    
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.fusion_weights = nn.Linear(hidden_size, num_experts)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_outputs: Dict[str, Optional[torch.Tensor]],
        workflow_plan: WorkflowPlan
    ) -> torch.Tensor:
        # Collect valid expert outputs
        valid_outputs = []
        for name, output in expert_outputs.items():
            if output is not None:
                valid_outputs.append(output)
                
        if not valid_outputs:
            return hidden_states
            
        # Stack expert outputs
        stacked = torch.stack(valid_outputs, dim=0)  # [num_valid, B, L, H]
        
        # Compute fusion weights based on workflow plan
        pooled = hidden_states.mean(dim=1)  # [B, H]
        weights = F.softmax(self.fusion_weights(pooled), dim=-1)  # [B, num_experts]
        
        # Weighted fusion
        num_valid = len(valid_outputs)
        weights = weights[:, :num_valid]  # [B, num_valid]
        # Reshape weights for broadcasting: [B, num_valid, 1, 1]
        weights = weights.unsqueeze(2).unsqueeze(3)
        # Weighted sum: stacked is [num_valid, B, L, H], weights is [B, num_valid, 1, 1]
        # Transpose stacked to [B, num_valid, L, H] for proper broadcasting
        stacked = stacked.permute(1, 0, 2, 3)  # [B, num_valid, L, H]
        fused = (stacked * weights).sum(dim=1)  # [B, L, H]
        
        # Final projection
        output = self.output_proj(fused)
        
        return output + hidden_states  # Residual connection


class DualStreamMixer(nn.Module):
    """Dual-stream mixer for anchor and emergence streams."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.anchor_proj = nn.Linear(hidden_size, hidden_size)
        self.emergence_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self,
        h_emergence: torch.Tensor,
        h_anchor_in: torch.Tensor,
        alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process anchor stream (stable, factual)
        h_anchor_processed = self.anchor_proj(h_anchor_in)
        
        # Process emergence stream (creative, detailed)
        h_emergence_processed = self.emergence_proj(h_emergence)
        
        # Mix streams based on alpha
        h_out = alpha * h_anchor_processed + (1 - alpha) * h_emergence_processed
        
        # Update anchor for next layer (EMA-style)
        h_anchor_out = 0.9 * h_anchor_in + 0.1 * h_emergence_processed
        
        return h_out, h_anchor_out


class ThinkLogFormatter:
    """Format structured thinking logs for <think> output."""
    
    def format(
        self,
        workflow_plan: WorkflowPlan,
        expert_outputs: Dict,
        verification: torch.Tensor
    ) -> Dict:
        """Format thinking process into structured logs."""
        logs = {
            'workflow': {
                'task_type': workflow_plan.task_type,
                'confidence': workflow_plan.confidence,
                'challenger_enabled': workflow_plan.enable_challenger,
            },
            'reasoning_path': [],
        }
        
        # Add reasoning steps
        if expert_outputs.get('proposer') is not None:
            logs['reasoning_path'].append({
                'step': 'proposal',
                'confidence': workflow_plan.confidence,
            })
            
        if expert_outputs.get('challenger') is not None:
            logs['reasoning_path'].append({
                'step': 'challenge',
                'reason': 'low_confidence' if workflow_plan.confidence < 0.7 else 'deliberation',
            })
            
        logs['reasoning_path'].append({
            'step': 'verification',
            'passed': True,  # Simplified for now
        })
        
        return logs
