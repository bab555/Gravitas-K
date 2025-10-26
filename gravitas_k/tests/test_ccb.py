"""Unit tests for Cognitive Core Block (CCB) module."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from gravitas_k.models.ccb import (
    CognitiveCoreBlock, 
    WorkflowPlan,
    Arbiter,
    Proposer,
    Challenger,
    Verifier,
    Synthesizer,
)


class TestCognitiveCoreBlock:
    """Test cases for CCB module."""
    
    @pytest.fixture
    def ccb_module(self):
        """Create a CCB module for testing."""
        return CognitiveCoreBlock(
            hidden_size=768,
            intermediate_size=3072,
            num_experts=4,
            enable_dual_stream=True,
            enable_think_logs=True,
        )
        
    def test_initialization(self, ccb_module):
        """Test module initialization."""
        assert ccb_module.hidden_size == 768
        assert ccb_module.intermediate_size == 3072
        assert ccb_module.num_experts == 4
        assert ccb_module.enable_dual_stream
        assert ccb_module.enable_think_logs
        
        # Check sub-modules exist
        assert isinstance(ccb_module.arbiter, Arbiter)
        assert isinstance(ccb_module.proposer, Proposer)
        assert isinstance(ccb_module.challenger, Challenger)
        assert isinstance(ccb_module.verifier, Verifier)
        assert isinstance(ccb_module.synthesizer, Synthesizer)
        
    def test_forward_pass(self, ccb_module):
        """Test forward pass through CCB."""
        batch_size = 2
        seq_len = 64
        hidden_size = 768
        
        # Create inputs
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        h_anchor_in = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        h_out, h_anchor_out, think_logs = ccb_module(
            hidden_states=hidden_states,
            h_anchor_in=h_anchor_in,
        )
        
        # Check outputs
        assert h_out.shape == (batch_size, seq_len, hidden_size)
        assert h_anchor_out.shape == (batch_size, seq_len, hidden_size)
        assert think_logs is not None
        
    def test_arbiter_workflow_planning(self):
        """Test Arbiter workflow planning."""
        arbiter = Arbiter(hidden_size=768)
        
        batch_size = 2
        seq_len = 32
        hidden_size = 768
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Get workflow plan
        plan = arbiter(hidden_states)
        
        # Check plan attributes
        assert isinstance(plan, WorkflowPlan)
        assert plan.task_type in ['creative', 'logical', 'deliberative']
        assert 0 <= plan.confidence <= 1
        assert isinstance(plan.enable_challenger, bool)
        assert isinstance(plan.enable_full_deliberation, bool)
        
    def test_proposer_with_prc(self):
        """Test Proposer with PRC samples."""
        proposer = Proposer(hidden_size=768, intermediate_size=3072)
        
        batch_size = 2
        seq_len = 32
        hidden_size = 768
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        prc_samples = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward with PRC
        output = proposer(hidden_states, prc_samples)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        
    def test_challenger_alternative_generation(self):
        """Test Challenger alternative generation."""
        challenger = Challenger(hidden_size=768, challenge_size=192)
        
        batch_size = 2
        seq_len = 32
        hidden_size = 768
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        proposals = torch.randn(batch_size, seq_len, hidden_size)
        
        # Generate challenge
        challenge = challenger(hidden_states, proposals)
        
        assert challenge.shape == (batch_size, seq_len, hidden_size)
        # Challenge should be different from proposals
        assert not torch.allclose(challenge, proposals)
        
    def test_verifier_validation(self):
        """Test Verifier validation."""
        verifier = Verifier(hidden_size=768, verify_size=384)
        
        batch_size = 2
        seq_len = 32
        hidden_size = 768
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        proposals = torch.randn(batch_size, seq_len, hidden_size)
        challenges = torch.randn(batch_size, seq_len, hidden_size)
        
        # Verify with challenges
        output = verifier(hidden_states, proposals, challenges)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        
    def test_synthesizer_fusion(self):
        """Test Synthesizer fusion of expert outputs."""
        synthesizer = Synthesizer(hidden_size=768, num_experts=3)
        
        batch_size = 2
        seq_len = 32
        hidden_size = 768
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        expert_outputs = {
            'proposer': torch.randn(batch_size, seq_len, hidden_size),
            'challenger': torch.randn(batch_size, seq_len, hidden_size),
            'verifier': torch.randn(batch_size, seq_len, hidden_size),
        }
        
        plan = WorkflowPlan(
            task_type='logical',
            confidence=0.8,
            enable_challenger=True,
            enable_full_deliberation=False,
            sigma_ctrl=0.5,
            gamma_ctrl=0.3,
            fc_strength=0.7,
            alpha_dual=0.6,
        )
        
        # Synthesize
        output = synthesizer(hidden_states, expert_outputs, plan)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        
    def test_dual_stream_mixing(self, ccb_module):
        """Test dual-stream mixing."""
        batch_size = 2
        seq_len = 32
        hidden_size = 768
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        h_anchor_in = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward with dual stream
        h_out, h_anchor_out, _ = ccb_module(
            hidden_states=hidden_states,
            h_anchor_in=h_anchor_in,
        )
        
        # Check that anchor is updated
        assert not torch.allclose(h_anchor_out, h_anchor_in)
        
    def test_think_log_generation(self, ccb_module):
        """Test thinking log generation."""
        batch_size = 1
        seq_len = 32
        hidden_size = 768
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        _, _, think_logs = ccb_module(hidden_states=hidden_states)
        
        # Check think logs structure
        assert isinstance(think_logs, dict)
        assert 'workflow' in think_logs
        assert 'reasoning_path' in think_logs
        
        workflow = think_logs['workflow']
        assert 'task_type' in workflow
        assert 'confidence' in workflow
        
    def test_without_challenger(self):
        """Test CCB without challenger enabled."""
        ccb = CognitiveCoreBlock(
            hidden_size=768,
            intermediate_size=3072,
            enable_dual_stream=False,
            enable_think_logs=False,
        )
        
        batch_size = 2
        seq_len = 32
        hidden_size = 768
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Mock arbiter to disable challenger
        with torch.no_grad():
            # This would normally be done by the arbiter based on confidence
            ccb.arbiter.confidence_head.weight.data.fill_(1.0)  # High confidence
            
        h_out, h_anchor_out, think_logs = ccb(hidden_states=hidden_states)
        
        assert h_out.shape == (batch_size, seq_len, hidden_size)


class TestWorkflowPlan:
    """Test WorkflowPlan dataclass."""
    
    def test_creation(self):
        """Test WorkflowPlan creation."""
        plan = WorkflowPlan(
            task_type='creative',
            confidence=0.75,
            enable_challenger=True,
            enable_full_deliberation=False,
            sigma_ctrl=0.4,
            gamma_ctrl=0.6,
            fc_strength=0.8,
            alpha_dual=0.5,
        )
        
        assert plan.task_type == 'creative'
        assert plan.confidence == 0.75
        assert plan.enable_challenger
        assert not plan.enable_full_deliberation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
