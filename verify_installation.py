"""
Verification script to check if Gravitas-K is properly installed.
"""

import sys
import torch
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are available."""
    print("Checking dependencies...")
    
    dependencies = {
        "torch": None,
        "transformers": None,
        "accelerate": None,
        "safetensors": None,
        "omegaconf": None,
        "tqdm": None,
    }
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            dependencies[dep] = module.__version__ if hasattr(module, '__version__') else "installed"
            print(f"  ✓ {dep}: {dependencies[dep]}")
        except ImportError:
            print(f"  ✗ {dep}: NOT INSTALLED")
            dependencies[dep] = None
            
    return all(v is not None for v in dependencies.values())


def check_cuda():
    """Check CUDA availability and version."""
    print("\nChecking CUDA...")
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  ✓ GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            
        # Check for RTX 5090D requirements
        cuda_version = torch.version.cuda
        if cuda_version:
            major, minor = map(int, cuda_version.split('.')[:2])
            if major >= 12 and minor >= 8:
                print(f"  ✓ CUDA 12.8+ detected (RTX 5090D compatible)")
            else:
                print(f"  ⚠ CUDA version {cuda_version} (RTX 5090D requires 12.8+)")
                
        return True
    else:
        print("  ✗ CUDA not available")
        return False


def check_model_files():
    """Check if Qwen3-8B model files are available."""
    print("\nChecking model files...")
    
    model_path = Path("../Qwen3-8B")
    
    if not model_path.exists():
        print(f"  ✗ Model directory not found: {model_path.absolute()}")
        return False
        
    required_files = [
        "config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
    ]
    
    model_files = ["model-00001-of-00005.safetensors",
                   "model-00002-of-00005.safetensors",
                   "model-00003-of-00005.safetensors",
                   "model-00004-of-00005.safetensors",
                   "model-00005-of-00005.safetensors"]
    
    all_found = True
    
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} NOT FOUND")
            all_found = False
            
    # Check for model weight files
    model_found = False
    for file in model_files:
        if (model_path / file).exists():
            model_found = True
            break
            
    if model_found:
        print(f"  ✓ Model weights found")
    else:
        print(f"  ✗ Model weights NOT FOUND")
        all_found = False
        
    return all_found


def check_gravitas_modules():
    """Check if Gravitas-K modules can be imported."""
    print("\nChecking Gravitas-K modules...")
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    modules = [
        "gravitas_k.models.flowing_context",
        "gravitas_k.models.ccb",
        "gravitas_k.models.prc",
        "gravitas_k.models.hvae",
        "gravitas_k.models.gravitas_k_model",
    ]
    
    all_imported = True
    
    for module_name in modules:
        try:
            module = __import__(module_name, fromlist=[''])
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            all_imported = False
            
    return all_imported


def check_memory():
    """Check available system and GPU memory."""
    print("\nChecking memory...")
    
    # System RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"  ✓ System RAM: {mem.total / 1024**3:.1f} GB total, {mem.available / 1024**3:.1f} GB available")
    except ImportError:
        print("  ⚠ psutil not installed, cannot check system RAM")
        
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            print(f"  ✓ GPU {i} memory: {mem_total:.1f} GB total, {mem_allocated:.1f} GB allocated, {mem_reserved:.1f} GB reserved")
            
            if mem_total >= 32:
                print(f"    ✓ Sufficient VRAM for full model")
            elif mem_total >= 24:
                print(f"    ⚠ Limited VRAM, 4-bit quantization recommended")
            else:
                print(f"    ⚠ Low VRAM, may need CPU offloading")


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Gravitas-K Installation Verification")
    print("=" * 60)
    
    checks = {
        "Dependencies": check_dependencies(),
        "CUDA": check_cuda(),
        "Model Files": check_model_files(),
        "Gravitas Modules": check_gravitas_modules(),
    }
    
    # Memory check (informational only)
    check_memory()
    
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{check_name}: {status}")
        if not passed:
            all_passed = False
            
    print("=" * 60)
    
    if all_passed:
        print("✓ All checks passed! Gravitas-K is ready to use.")
        print("\nNext steps:")
        print("1. Run the demo: python demo.py")
        print("2. Start training: python -m gravitas_k.runtime.train")
    else:
        print("✗ Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        
        if not checks["Model Files"]:
            print("\nModel files not found. Please download Qwen3-8B:")
            print("  huggingface-cli download Qwen/Qwen3-8B --local-dir ../Qwen3-8B")
            
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
