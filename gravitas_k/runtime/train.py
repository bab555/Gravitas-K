"""
Training script for Gravitas-K model.

This script handles the training loop with all the special considerations
for our modified architecture including CCB, FC, PRC, and HVAE.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf
from pathlib import Path
import logging
from typing import Dict, Optional
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from gravitas_k.models.gravitas_k_model import GravitasKModel, GravitasKConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GravitasKDataset(Dataset):
    """Dataset for Gravitas-K training with thinking tags support."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 8192,
        add_thinking_tags: bool = True,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_thinking_tags = add_thinking_tags
        
        # Load data (simplified for now)
        self.examples = self._load_examples()
        
    def _load_examples(self):
        """Load training examples from disk."""
        examples = []
        
        # This is a placeholder - in practice, load from your dataset
        # For now, create some dummy examples
        for i in range(100):
            examples.append({
                'input': f"Question {i}: What is the capital of France?",
                'output': f"Answer {i}: The capital of France is Paris.",
                'thinking': f"I need to recall geographical knowledge about France."
            })
            
        return examples
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format with thinking tags if enabled
        if self.add_thinking_tags and 'thinking' in example:
            text = f"{example['input']}\n<think>{example['thinking']}</think>\n{example['output']}"
        else:
            text = f"{example['input']}\n{example['output']}"
            
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': tokens['input_ids'].squeeze(),  # For causal LM
        }


class GravitasKTrainer:
    """Trainer for Gravitas-K model."""
    
    def __init__(self, config_path: str):
        # Load configuration
        self.config = OmegaConf.load(config_path)
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            mixed_precision='bf16' if self.config.training.bf16 else None,
            log_with=['wandb', 'tensorboard'] if self.config.experiment.use_wandb else ['tensorboard'],
        )
        
        # Set up logging
        if self.accelerator.is_main_process:
            if self.config.experiment.use_wandb:
                wandb.init(
                    project=self.config.experiment.wandb_project,
                    name=self.config.experiment.name,
                    config=OmegaConf.to_container(self.config),
                    tags=self.config.experiment.tags,
                )
                
        # Initialize model
        self.model = self._init_model()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.data.tokenizer_path,
            trust_remote_code=True,
        )
        
        # Initialize datasets
        self.train_dataset = GravitasKDataset(
            self.config.data.train_data_path,
            self.tokenizer,
            self.config.training.max_seq_length,
            self.config.data.add_thinking_tags,
        )
        
        self.eval_dataset = GravitasKDataset(
            self.config.data.eval_data_path,
            self.tokenizer,
            self.config.training.max_seq_length,
            self.config.data.add_thinking_tags,
        )
        
        # Initialize data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.micro_batch_size,
            shuffle=True,
            num_workers=2,
        )
        
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.training.micro_batch_size,
            shuffle=False,
            num_workers=2,
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        
        # Prepare for distributed training
        self.model, self.optimizer, self.train_loader, self.eval_loader = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.eval_loader
            )
            
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
    def _init_model(self):
        """Initialize Gravitas-K model."""
        # Get current stage configuration
        stage = self.config.experiment.name.split('_')[-1]  # e.g., 'stage_a' -> 'a'
        stage_config = self.config.stages.get(f'stage_{stage}', self.config.stages.stage_a)
        
        # Create model configuration
        model_config = GravitasKConfig(
            base_model_path=self.config.model.base_model_path,
            num_modified_layers=self.config.model.num_modified_layers,
            enable_fc=self.config.model.enable_fc,
            enable_prc=stage_config.get('enable_prc', self.config.model.enable_prc),
            enable_hvae=stage_config.get('enable_hvae', self.config.model.enable_hvae),
            enable_dual_stream=self.config.model.enable_dual_stream,
            enable_think_logs=self.config.model.enable_think_logs,
            fc_num_segments=self.config.model.fc_num_segments,
            prc_k_neighbors=self.config.model.prc_k_neighbors,
            hvae_levels=self.config.hvae_levels,
            load_in_4bit=self.config.model.load_in_4bit,
        )
        
        # Create model
        model = GravitasKModel(model_config)
        
        # Build concept banks if PRC is enabled
        if model_config.enable_prc:
            model.build_concept_banks()
            
        # Enable gradient checkpointing if specified
        if self.config.training.gradient_checkpointing:
            model.base_model.gradient_checkpointing_enable()
            
        return model
        
    def _init_optimizer(self):
        """Initialize optimizer with different learning rates for different modules."""
        # Separate parameters by type
        new_params = []  # CCB, FC, PRC, HVAE
        frozen_params = []  # Attention layers
        ln_params = []  # Layer norms
        
        for name, param in self.model.named_parameters():
            if any(module in name for module in ['ccb', 'fc', 'prc', 'hvae']):
                new_params.append(param)
            elif 'layernorm' in name.lower() or 'ln' in name.lower():
                ln_params.append(param)
            else:
                frozen_params.append(param)
                
        # Create parameter groups
        param_groups = [
            {'params': new_params, 'lr': self.config.training.learning_rate_new},
            {'params': ln_params, 'lr': self.config.training.learning_rate_ln},
            {'params': frozen_params, 'lr': self.config.training.learning_rate_frozen},
        ]
        
        # Initialize optimizer
        if self.config.training.optimizer == 'adamw_8bit':
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    param_groups,
                    weight_decay=self.config.training.weight_decay,
                )
            except ImportError:
                logger.warning("8-bit optimizer not available, using standard AdamW")
                optimizer = torch.optim.AdamW(
                    param_groups,
                    weight_decay=self.config.training.weight_decay,
                )
        else:
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.config.training.weight_decay,
            )
            
        return optimizer
        
    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=self.config.training.max_steps,
        )
        return scheduler
        
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        self.model.train()
        progress_bar = tqdm(
            range(self.config.training.max_steps),
            desc="Training",
            disable=not self.accelerator.is_main_process,
        )
        
        for epoch in range(100):  # Large number, will break on max_steps
            for batch_idx, batch in enumerate(self.train_loader):
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                    )
                    
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping
                    if self.config.training.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.max_grad_norm,
                        )
                        
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update progress
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    # Logging
                    if self.global_step % self.config.training.logging_steps == 0:
                        self._log_metrics({'train/loss': loss.item()})
                        
                    # Evaluation
                    if self.global_step % self.config.training.eval_steps == 0:
                        eval_loss = self.evaluate()
                        self._log_metrics({'eval/loss': eval_loss})
                        
                        # Save best model
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint('best')
                            
                    # Regular checkpoint
                    if self.global_step % self.config.training.save_steps == 0:
                        self.save_checkpoint(f'step_{self.global_step}')
                        
                    # Check if done
                    if self.global_step >= self.config.training.max_steps:
                        logger.info("Reached maximum training steps")
                        return
                        
        logger.info("Training completed")
        
    def evaluate(self):
        """Evaluation loop."""
        logger.info("Running evaluation...")
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(
                self.eval_loader,
                desc="Evaluating",
                disable=not self.accelerator.is_main_process,
            ):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )
                
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        self.model.train()
        
        return avg_loss
        
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        if self.accelerator.is_main_process:
            checkpoint_dir = Path(f"./checkpoints/{self.config.experiment.name}/{name}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.accelerator.unwrap_model(self.model).save_pretrained(checkpoint_dir)
            
            # Save training state
            state = {
                'global_step': self.global_step,
                'best_eval_loss': self.best_eval_loss,
                'config': OmegaConf.to_container(self.config),
            }
            torch.save(state, checkpoint_dir / 'training_state.pt')
            
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
    def _log_metrics(self, metrics: Dict):
        """Log metrics to wandb and tensorboard."""
        if self.accelerator.is_main_process:
            # Add step to metrics
            metrics['step'] = self.global_step
            
            # Log to wandb
            if self.config.experiment.use_wandb:
                wandb.log(metrics, step=self.global_step)
                
            # Log to tensorboard
            if self.config.experiment.use_tensorboard:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(self.config.experiment.tb_log_dir)
                for key, value in metrics.items():
                    if key != 'step':
                        writer.add_scalar(key, value, self.global_step)
                writer.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Gravitas-K model')
    parser.add_argument(
        '--config',
        type=str,
        default='gravitas_k/runtime/config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = GravitasKTrainer(args.config)
    trainer.train()
    
    # Final evaluation
    final_eval_loss = trainer.evaluate()
    logger.info(f"Final evaluation loss: {final_eval_loss:.4f}")
    
    # Save final checkpoint
    trainer.save_checkpoint('final')
    

if __name__ == '__main__':
    main()
