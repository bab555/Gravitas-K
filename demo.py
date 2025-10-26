"""
Demo script for Gravitas-K model.

This script demonstrates the basic usage of Gravitas-K model
including loading, inference, and thinking log generation.
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from gravitas_k.models.gravitas_k_model import GravitasKModel, GravitasKConfig


def main():
    """Main demo function."""
    print("=" * 60)
    print("Gravitas-K Demo")
    print("=" * 60)
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize configuration
    print("\nInitializing Gravitas-K configuration...")
    config = GravitasKConfig(
        base_model_path="../Qwen3-8B",
        num_modified_layers=8,
        enable_fc=True,
        enable_prc=True,
        enable_hvae=True,
        enable_dual_stream=True,
        enable_think_logs=True,
        load_in_4bit=True,  # For 32GB VRAM optimization
    )
    
    # Load model
    print("Loading Gravitas-K model (this may take a few minutes)...")
    try:
        model = GravitasKModel(config)
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Make sure Qwen3-8B model is available at ../Qwen3-8B/")
        return
        
    # Build concept banks for PRC
    print("Building concept banks for PRC...")
    model.build_concept_banks()
    print("✓ Concept banks built")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_path,
        trust_remote_code=True,
    )
    
    # Demo prompts
    prompts = [
        "Explain the concept of quantum entanglement in simple terms.",
        "Write a creative story about a robot learning to paint.",
        "Solve this step by step: If a train travels 120 km in 1.5 hours, what is its average speed?",
    ]
    
    print("\n" + "=" * 60)
    print("Running inference examples...")
    print("=" * 60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Prompt: {prompt}")
        print("-" * 40)
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        
        # Move to device
        input_ids = inputs["input_ids"].to(device)
        
        # Generate with thinking logs
        print("Generating response...")
        with torch.no_grad():
            outputs, think_logs = model.generate_with_think(
                input_ids=input_ids,
                max_new_tokens=200,
                temperature=0.6,
                top_p=0.95,
            )
            
        # Decode output
        response = tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        # Display results
        if think_logs:
            print("\nThinking Process:")
            print(think_logs)
            
        print("\nResponse:")
        print(response)
        print("-" * 40)
        
    # Demonstrate component inspection
    print("\n" + "=" * 60)
    print("Model Architecture Summary")
    print("=" * 60)
    
    # Count modified layers
    model_layers = model.base_model.model.layers if hasattr(model.base_model, 'model') else model.base_model.layers
    modified_count = sum(1 for layer in model_layers if hasattr(layer, 'ccb'))
    
    print(f"Total layers: {len(model_layers)}")
    print(f"Modified layers (with CCB): {modified_count}")
    print(f"Flowing Context enabled: {config.enable_fc}")
    print(f"PRC enabled: {config.enable_prc}")
    print(f"HVAE enabled: {config.enable_hvae}")
    print(f"Dual-stream enabled: {config.enable_dual_stream}")
    
    # Show HVAE hierarchy if enabled
    if config.enable_hvae and model.hvae is not None:
        print("\nHVAE Hierarchy:")
        for level_name, level_dim in model.hvae.levels:
            print(f"  - {level_name}: {level_dim}d")
            
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def interactive_mode():
    """Interactive chat mode with Gravitas-K."""
    print("=" * 60)
    print("Gravitas-K Interactive Mode")
    print("=" * 60)
    print("Type 'quit' to exit")
    print("-" * 60)
    
    # Initialize model (simplified for demo)
    config = GravitasKConfig(
        base_model_path="../Qwen3-8B",
        num_modified_layers=4,  # Fewer layers for faster loading
        load_in_4bit=True,
    )
    
    print("Loading model...")
    model = GravitasKModel(config)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_path,
        trust_remote_code=True,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        if not user_input:
            continue
            
        # Generate response
        print("Gravitas-K: ", end="", flush=True)
        
        inputs = tokenizer(user_input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        with torch.no_grad():
            outputs = model.base_model.generate(
                input_ids=input_ids,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
            
        response = tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        print(response)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gravitas-K Demo")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode"
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    else:
        main()
