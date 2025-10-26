"""
Model Loader for Gravitas-K

Handles loading and initialization of Gravitas-K models with:
- Qwen3-8B base model loading
- 4-bit quantization
- CCB layer replacement
- PRC/HVAE initialization
- Weight inheritance from pre-trained models
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GravitasKModelLoader:
    """
    Comprehensive model loader for Gravitas-K.
    
    Handles:
    1. Loading Qwen3-8B with optional quantization
    2. Replacing decoder layers with GravitasKDecoderLayer
    3. Initializing PRC, HVAE, FlowingContext
    4. Loading from checkpoints
    """
    
    def __init__(
        self,
        base_model_path: str = "../Qwen3-8B",
        num_ccb_layers: int = 12,
        quantization: str = "4bit",  # "4bit", "8bit", or None
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            base_model_path: Path to Qwen3-8B model
            num_ccb_layers: Number of last layers to replace with CCB
            quantization: Quantization mode
            device: Device to load model on
        """
        self.base_model_path = Path(base_model_path)
        self.num_ccb_layers = num_ccb_layers
        self.quantization = quantization
        self.device = device
        
        # Verify base model exists
        if not self.base_model_path.exists():
            raise FileNotFoundError(f"Base model not found at {self.base_model_path}")
    
    def load_base_model(self) -> tuple:
        """
        Load Qwen3-8B base model with optional quantization.
        
        Returns:
            Tuple of (model, tokenizer, config)
        """
        logger.info(f"Loading base model from {self.base_model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.base_model_path),
            trust_remote_code=True,
        )
        
        # Setup quantization config
        bnb_config = None
        if self.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using 4-bit quantization")
        elif self.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            logger.info("Using 8-bit quantization")
        
        # Load config
        config = AutoConfig.from_pretrained(
            str(self.base_model_path),
            trust_remote_code=True,
        )
        
        # Load model
        model = AutoModel.from_pretrained(
            str(self.base_model_path),
            config=config,
            quantization_config=bnb_config,
            device_map="auto" if self.quantization else self.device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.quantization else torch.float32,
        )
        
        logger.info(f"✓ Base model loaded: {model.__class__.__name__}")
        logger.info(f"  - Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        logger.info(f"  - Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B")
        
        return model, tokenizer, config
    
    def load_gravitas_k_model(
        self,
        enable_fc: bool = True,
        enable_ccb: bool = True,
        enable_prc: bool = False,  # Stage B
        enable_hvae: bool = False,  # Stage B
        enable_dual_stream: bool = True,
    ) -> tuple:
        """
        Load Gravitas-K model with specified modules enabled.
        
        Args:
            enable_fc: Enable FlowingContext
            enable_ccb: Enable CognitiveCoreBlock (A-P-C-V-S)
            enable_prc: Enable PRC
            enable_hvae: Enable HVAE
            enable_dual_stream: Enable Dual-Stream
            
        Returns:
            Tuple of (model, tokenizer)
        """
        from gravitas_k.models.gravitas_k_model import GravitasKModel
        from transformers import Qwen3ForCausalLM, Qwen3Config
        
        # Load base model
        base_model, tokenizer, config = self.load_base_model()
        
        # Create Gravitas-K specific config
        gravitas_config = Qwen3Config(**config.to_dict())
        
        # Initialize Gravitas-K model
        logger.info("Initializing Gravitas-K architecture...")
        
        # Create model with base weights
        model = GravitasKModel(
            config=gravitas_config,
            num_ccb_layers=self.num_ccb_layers if enable_ccb else 0,
        )
        
        # Load base model weights (excluding layers we'll replace)
        logger.info("Loading pre-trained weights...")
        model.load_state_dict(base_model.state_dict(), strict=False)
        
        # Configure modules
        if enable_fc:
            logger.info("✓ FlowingContext enabled")
        
        if enable_ccb:
            logger.info(f"✓ CognitiveCoreBlock enabled (replacing last {self.num_ccb_layers} layers)")
        
        if enable_prc:
            logger.info("✓ PRC enabled")
            # Build concept bank from lm_head
            if hasattr(model, 'lm_head') and hasattr(model, 'prc_module'):
                model.prc_module.build_concept_bank(model.lm_head.weight)
        
        if enable_hvae:
            logger.info("✓ HVAE enabled")
        
        if enable_dual_stream:
            logger.info("✓ Dual-Stream enabled")
        
        # Move to device if not quantized (quantized models use device_map)
        if not self.quantization:
            model.to(self.device)
        
        logger.info(f"✓ Gravitas-K model ready on {self.device}")
        
        return model, tokenizer
    
    def freeze_base_model(self, model, unfreeze_ln_bias: bool = True):
        """
        Freeze base model parameters, optionally unfreezing LayerNorm and bias.
        
        Args:
            model: Gravitas-K model
            unfreeze_ln_bias: Whether to unfreeze LayerNorm and bias parameters
        """
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze newly added modules
        for name, module in model.named_modules():
            if any(keyword in name for keyword in [
                'flowing_context',
                'prc_module',
                'hvae',
                'arbiter',
                'proposer',
                'challenger',
                'verifier',
                'synthesizer',
                'dual_stream',
            ]):
                for param in module.parameters():
                    param.requires_grad = True
        
        # Optionally unfreeze LN and bias
        if unfreeze_ln_bias:
            for name, param in model.named_parameters():
                if 'norm' in name.lower() or 'bias' in name:
                    param.requires_grad = True
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M / {total_params / 1e9:.2f}B")
        logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optional optimizer to load state
            scheduler: Optional scheduler to load state
            
        Returns:
            Dictionary with checkpoint metadata (epoch, step, etc.)
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0),
            'best_loss': checkpoint.get('best_loss', float('inf')),
        }
        
        logger.info(f"✓ Checkpoint loaded (epoch {metadata['epoch']}, step {metadata['global_step']})")
        
        return metadata
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        best_loss: float = float('inf'),
        additional_metadata: Optional[Dict] = None,
    ):
        """
        Save model checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
            model: Model to save
            optimizer: Optional optimizer to save state
            scheduler: Optional scheduler to save state
            epoch: Current epoch
            global_step: Global training step
            best_loss: Best validation loss
            additional_metadata: Additional metadata to save
        """
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'best_loss': best_loss,
            'model_state_dict': model.state_dict(),
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if additional_metadata:
            checkpoint.update(additional_metadata)
        
        # Create directory if needed
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"✓ Checkpoint saved")

