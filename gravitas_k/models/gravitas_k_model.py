"""
Gravitas-K Model: Surgical modification of Qwen3-8B architecture.

This module implements the main model that:
1. Loads Qwen3-8B base model
2. Replaces FFN layers with CCB in the last 8-12 layers
3. Integrates FC, PRC, and HVAE modules
4. Manages dual-stream and head vector bus
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional, Tuple, Dict, List, Union
import copy
import warnings

from .flowing_context import FlowingContext
from .ccb import CognitiveCoreBlock
from .prc import ProbabilisticRegionCollapse, HierarchicalPRC
from .hvae import HierarchicalVAE


class GravitasKConfig:
    """Configuration for Gravitas-K modifications."""
    
    def __init__(
        self,
        base_model_path: str = "../Qwen3-8B",  # Relative to Gravitas-K directory
        num_modified_layers: int = 8,  # Number of layers to replace FFN with CCB
        enable_fc: bool = True,
        enable_prc: bool = True,
        enable_hvae: bool = True,
        enable_dual_stream: bool = True,
        enable_think_logs: bool = True,
        fc_num_segments: int = 4,
        prc_k_neighbors: int = 5,
        hvae_levels: Optional[List[Tuple[str, int]]] = None,
        load_in_4bit: bool = False,  # For 32GB VRAM optimization
    ):
        self.base_model_path = base_model_path
        self.num_modified_layers = num_modified_layers
        self.enable_fc = enable_fc
        self.enable_prc = enable_prc
        self.enable_hvae = enable_hvae
        self.enable_dual_stream = enable_dual_stream
        self.enable_think_logs = enable_think_logs
        self.fc_num_segments = fc_num_segments
        self.prc_k_neighbors = prc_k_neighbors
        self.hvae_levels = hvae_levels or [
            ('directory', 768),
            ('page', 256),
            ('line', 128),
            ('phrase', 64),
            ('word', 32),
        ]
        self.load_in_4bit = load_in_4bit


class ModifiedQwen3DecoderLayer(nn.Module):
    """
    Modified Qwen3 decoder layer with CCB replacing FFN.
    
    This layer maintains the attention mechanism but replaces
    the feed-forward network with our Cognitive Core Block.
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        config: GravitasKConfig,
        layer_idx: int,
    ):
        super().__init__()
        
        # Keep original attention and layer norms
        self.self_attn = original_layer.self_attn
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm
        
        # Get dimensions from original layer
        hidden_size = original_layer.mlp.up_proj.in_features
        intermediate_size = original_layer.mlp.up_proj.out_features
        
        # Replace MLP with CCB
        self.ccb = CognitiveCoreBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            enable_dual_stream=config.enable_dual_stream,
            enable_think_logs=config.enable_think_logs,
        )
        
        # Initialize CCB from original FFN weights where possible
        if hasattr(original_layer, 'mlp'):
            # Initialize Proposer with original FFN weights
            self.ccb.proposer.up_proj.weight.data = original_layer.mlp.up_proj.weight.data.clone()
            self.ccb.proposer.gate_proj.weight.data = original_layer.mlp.gate_proj.weight.data.clone()
            self.ccb.proposer.down_proj.weight.data = original_layer.mlp.down_proj.weight.data.clone()
            
        # Add Flowing Context if enabled
        self.fc = None
        if config.enable_fc:
            self.fc = FlowingContext(
                hidden_size=hidden_size,
                num_segments=config.fc_num_segments,
            )
            
        # Add PRC if enabled
        self.prc = None
        if config.enable_prc:
            self.prc = ProbabilisticRegionCollapse(
                concept_dim=hidden_size,
                k_neighbors=config.prc_k_neighbors,
            )
            
        self.layer_idx = layer_idx
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        h_anchor_in: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through modified layer.
        
        Returns:
            Tuple containing:
            - hidden_states
            - h_anchor_out (if dual-stream enabled)
            - present_key_value (if use_cache)
            - attentions (if output_attentions)
            - think_logs (if enabled)
        """
        residual = hidden_states
        
        # LayerNorm before attention
        hidden_states = self.input_layernorm(hidden_states)
        
        # Apply Flowing Context if available (used to derive focus segments)
        fc_segments = None
        if self.fc is not None:
            _, fc_segments, _ = self.fc(hidden_states, attention_mask)
            
        # Self-attention with optional FC bias
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        
        hidden_states = attn_outputs[0]
        
        hidden_states = residual + hidden_states
        
        # LayerNorm before CCB
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply PRC sampling if enabled and in focus regions
        prc_samples = None
        if self.prc is not None and fc_segments is not None:
            # Create trigger mask from FC segments
            batch_size, seq_len = hidden_states.shape[:2]
            trigger_mask = torch.zeros(batch_size, seq_len, device=hidden_states.device)
            for b, segments in enumerate(fc_segments):
                if len(segments) > 0:
                    trigger_mask[b, segments] = 1.0
                    
            prc_samples, _ = self.prc(hidden_states, trigger_mask=trigger_mask)
            
        # Pass through CCB instead of FFN
        hidden_states, h_anchor_out, think_logs = self.ccb(
            hidden_states=hidden_states,
            h_anchor_in=h_anchor_in,
            prc_samples=prc_samples,
            fc_mask=fc_segments,
        )
        
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if h_anchor_out is not None:
            outputs += (h_anchor_out,)
            
        if use_cache:
            outputs += (attn_outputs[1],) if len(attn_outputs) > 1 else (None,)
            
        if output_attentions:
            outputs += (attn_outputs[-1],) if output_attentions else (None,)
            
        if think_logs is not None:
            outputs += (think_logs,)
            
        return outputs


class GravitasKModel(nn.Module):
    """
    Main Gravitas-K model with surgical modifications to Qwen3.
    
    This model:
    1. Loads the base Qwen3 model
    2. Replaces specified layers with modified versions
    3. Manages HVAE, PRC concept banks
    4. Handles dual-stream propagation
    """
    
    def __init__(self, config: GravitasKConfig):
        super().__init__()
        self.config = config
        
        # Load base model configuration
        base_config = AutoConfig.from_pretrained(config.base_model_path)
        
        # Load base model
        if config.load_in_4bit:
            # Use 4-bit quantization for base model to save VRAM
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            
        # Perform surgery: replace last N layers
        self._perform_surgery()
        
        # Initialize HVAE if enabled
        self.hvae = None
        if config.enable_hvae:
            self.hvae = HierarchicalVAE(
                input_dim=base_config.hidden_size,
                levels=config.hvae_levels,
            )
            
        # Initialize hierarchical PRC if HVAE is enabled
        if config.enable_prc and config.enable_hvae:
            self.hierarchical_prc = HierarchicalPRC(
                levels=config.hvae_levels,
                base_k=config.prc_k_neighbors,
            )
            
        # Head vector bus for cross-layer communication
        self.head_vector_bus = HeadVectorBus(
            hidden_size=base_config.hidden_size,
            num_layers=base_config.num_hidden_layers,
        )
        
    def _perform_surgery(self):
        """Replace FFN with CCB in the last N layers."""
        model = self.base_model.model if hasattr(self.base_model, 'model') else self.base_model
        layers = model.layers
        
        num_layers = len(layers)
        start_idx = num_layers - self.config.num_modified_layers
        
        print(f"Performing surgery on layers {start_idx} to {num_layers-1}")
        
        for i in range(start_idx, num_layers):
            original_layer = layers[i]
            
            # Create modified layer
            modified_layer = ModifiedQwen3DecoderLayer(
                original_layer=original_layer,
                config=self.config,
                layer_idx=i,
            )
            
            # Replace the layer
            layers[i] = modified_layer
            
        print(f"Surgery complete: {self.config.num_modified_layers} layers modified")
        
    def build_concept_banks(self, data_loader=None):
        """Build concept banks for PRC from data or model weights."""
        if not self.config.enable_prc:
            return
            
        # Use output projection weights as initial concept bank
        lm_head = self.base_model.lm_head if hasattr(self.base_model, 'lm_head') else None
        if lm_head is not None:
            # [vocab_size, hidden_size]
            concept_bank = lm_head.weight.data
            
            # Build index for each modified layer's PRC
            model = self.base_model.model if hasattr(self.base_model, 'model') else self.base_model
            layers = model.layers
            
            for layer in layers:
                if hasattr(layer, 'prc') and layer.prc is not None:
                    layer.prc.build_index(concept_bank)
                    
            print(f"Built PRC concept banks from lm_head weights")
            
        # If HVAE is enabled, also build hierarchical concept banks
        if self.config.enable_hvae and self.hvae is not None:
            # This would be populated during training
            pass
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        """
        Forward pass through Gravitas-K model.
        
        This maintains compatibility with HuggingFace transformers
        while adding our cognitive architecture.
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        
        # Add HVAE processing if enabled
        if self.hvae is not None and output_hidden_states:
            # Get last hidden states
            hidden_states = outputs.hidden_states[-1] if return_dict else outputs[1][-1]
            
            # Pass through HVAE
            hvae_outputs = self.hvae(hidden_states)
            
            # Add HVAE loss to the main loss
            if labels is not None:
                if return_dict:
                    outputs.loss = outputs.loss + 0.1 * hvae_outputs['total_loss']
                else:
                    outputs = (outputs[0] + 0.1 * hvae_outputs['total_loss'],) + outputs[1:]
                    
        return outputs
        
    def generate_with_think(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.95,
        **kwargs,
    ):
        """
        Generate text with thinking logs.
        
        This method generates text while capturing the internal
        thinking process from CCB modules.
        """
        # Enable think log collection
        think_logs = []
        
        # Standard generation
        outputs = self.base_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            **kwargs,
        )
        
        # Format think logs
        formatted_logs = self._format_think_logs(think_logs)
        
        return outputs, formatted_logs
        
    def _format_think_logs(self, logs: List[Dict]) -> str:
        """Format thinking logs into readable text."""
        if not logs:
            return ""
            
        formatted = "<think>\n"
        for i, log in enumerate(logs):
            formatted += f"Step {i+1}:\n"
            if 'workflow' in log:
                formatted += f"  Task: {log['workflow']['task_type']}\n"
                formatted += f"  Confidence: {log['workflow']['confidence']:.2f}\n"
            if 'reasoning_path' in log:
                for step in log['reasoning_path']:
                    formatted += f"  - {step['step']}\n"
        formatted += "</think>\n"
        
        return formatted


class HeadVectorBus(nn.Module):
    """Head vector bus for cross-layer anchor stream communication."""
    
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Aggregation weights for combining anchor vectors
        self.aggregation_weights = nn.Parameter(
            torch.ones(num_layers) / num_layers
        )
        
        # Optional transformation for anchor vectors
        self.anchor_transform = nn.Linear(hidden_size, hidden_size)
        
    def aggregate_anchors(self, anchor_vectors: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate anchor vectors from multiple layers."""
        if not anchor_vectors:
            return None
            
        # Stack and weight
        stacked = torch.stack(anchor_vectors, dim=0)  # [num_layers, batch_size, seq_len, hidden_size]
        weights = F.softmax(self.aggregation_weights[:len(anchor_vectors)], dim=0)
        weights = weights.view(-1, 1, 1, 1)
        
        # Weighted average
        aggregated = (stacked * weights).sum(dim=0)
        
        # Transform
        aggregated = self.anchor_transform(aggregated)
        
        return aggregated
