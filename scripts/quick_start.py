"""
Quick Start Script for Gravitas-K

Provides simple entry points for:
1. Training a Gravitas-K model
2. Launching interactive demo
3. Running evaluation
"""

import sys
from pathlib import Path
import argparse
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from gravitas_k.utils.model_loader import GravitasKModelLoader
from gravitas_k.utils.gradio_interface import GravitasKDemo
from gravitas_k.utils.training_monitor import TrainingMonitor


def train(args):
    """Start training."""
    print("🚀 Starting Gravitas-K training...")
    
    # Load model
    loader = GravitasKModelLoader(
        base_model_path=args.base_model,
        num_ccb_layers=args.num_ccb_layers,
        quantization=args.quantization,
    )
    
    model, tokenizer = loader.load_gravitas_k_model(
        enable_fc=True,
        enable_ccb=True,
        enable_prc=args.stage in ['B', 'C'],
        enable_hvae=args.stage in ['B', 'C'],
        enable_dual_stream=True,
    )
    
    # Freeze base model
    loader.freeze_base_model(model, unfreeze_ln_bias=True)
    
    # Initialize monitor
    monitor = TrainingMonitor(
        log_dir=args.log_dir,
        run_name=args.run_name,
        use_tensorboard=True,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )
    
    print(f"✓ Model loaded and ready for Stage {args.stage} training")
    print(f"✓ Logs will be saved to: {monitor.run_dir}")
    print("\n⚠ Note: Full training loop is in gravitas_k/runtime/train.py")
    print("  Run with: python -m gravitas_k.runtime.train --config config.yaml")


def demo(args):
    """Launch interactive demo."""
    print("🎨 Launching Gravitas-K interactive demo...")
    
    # Load model
    loader = GravitasKModelLoader(
        base_model_path=args.base_model,
        num_ccb_layers=12,
        quantization="4bit" if not args.no_quantization else None,
    )
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        model, tokenizer = loader.load_gravitas_k_model()
        loader.load_checkpoint(args.checkpoint, model)
    else:
        print("Loading base Gravitas-K model (no checkpoint)")
        model, tokenizer = loader.load_gravitas_k_model()
    
    # Create demo
    demo = GravitasKDemo(
        model=model,
        tokenizer=tokenizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        export_dir=args.export_dir,
    )
    
    print(f"✓ Demo ready!")
    print(f"  - Port: {args.port}")
    print(f"  - Share: {args.share}")
    print(f"  - Export directory: {args.export_dir}")
    
    demo.launch(share=args.share, server_port=args.port)


def evaluate(args):
    """Run evaluation."""
    print("📊 Running Gravitas-K evaluation...")
    print("\n⚠ Note: Full evaluation suite is in gravitas_k/runtime/eval.py")
    print("  Run with: python -m gravitas_k.runtime.eval --config config.yaml --checkpoint path/to/checkpoint")


def main():
    parser = argparse.ArgumentParser(description="Gravitas-K Quick Start")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Start training')
    train_parser.add_argument('--base-model', type=str, default='../Qwen3-8B', help='Path to base model')
    train_parser.add_argument('--num-ccb-layers', type=int, default=12, help='Number of CCB layers')
    train_parser.add_argument('--quantization', type=str, choices=['4bit', '8bit', 'none'], default='4bit')
    train_parser.add_argument('--stage', type=str, choices=['A', 'B', 'C'], default='A', help='Training stage')
    train_parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    train_parser.add_argument('--run-name', type=str, default=None, help='Run name')
    train_parser.add_argument('--use-wandb', action='store_true', help='Use W&B logging')
    train_parser.add_argument('--wandb-project', type=str, default='Gravitas-K', help='W&B project name')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Launch interactive demo')
    demo_parser.add_argument('--base-model', type=str, default='../Qwen3-8B', help='Path to base model')
    demo_parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    demo_parser.add_argument('--port', type=int, default=7860, help='Server port')
    demo_parser.add_argument('--share', action='store_true', help='Create public link')
    demo_parser.add_argument('--export-dir', type=str, default='./exported_samples', help='Sample export directory')
    demo_parser.add_argument('--no-quantization', action='store_true', help='Disable quantization')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Run evaluation')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'demo':
        demo(args)
    elif args.command == 'eval':
        evaluate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

