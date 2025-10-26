# Gravitas-K: Cognitive Architecture for Structured Reasoning

Gravitas-K is an advanced cognitive architecture that enhances large language models with structured reasoning capabilities through the integration of multiple innovative components.

## 🌟 Key Features

- **Cognitive Core Block (CCB)**: Replaces traditional FFN layers with A-P-C-V-S (Arbiter-Proposer-Challenger-Verifier-Synthesizer) workflow
- **Flowing Context (FC)**: Efficient Scan→Focus attention mechanism for long context processing
- **Probabilistic Region Collapse (PRC)**: Controlled prior injection from concept space
- **Hierarchical VAE (HVAE)**: Multi-level concept manifold (Directory→Page→Line→Phrase→Word)
- **Dual-Stream Attention**: Separate anchor (stable) and emergence (creative) information streams
- **Thinking Mode Integration**: Leverages Qwen3's native `<think>` tags for interpretable reasoning

## 📋 Requirements

- Python 3.10+
- PyTorch 2.5+ with CUDA 12.8+ (for RTX 5090D support)
- 32GB+ VRAM
- Qwen3-8B model weights

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bab555/Gravitas-K.git
cd Gravitas-K

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Model Setup

1. Ensure Qwen3-8B model is available in the parent directory:
```
../Qwen3-8B/
├── config.json
├── model-*.safetensors
├── tokenizer.json
└── ...
```

2. Run the demo:
```python
from gravitas_k.models.gravitas_k_model import GravitasKModel, GravitasKConfig

# Initialize configuration
config = GravitasKConfig(
    base_model_path="../Qwen3-8B",
    num_modified_layers=8,
    enable_fc=True,
    enable_prc=True,
    enable_hvae=True,
    load_in_4bit=True,  # For 32GB VRAM
)

# Load model
model = GravitasKModel(config)

# Build concept banks for PRC
model.build_concept_banks()

# Generate with thinking logs
outputs, think_logs = model.generate_with_think(
    input_ids,
    max_new_tokens=1024,
    temperature=0.6,
)
```

## 🏋️ Training

### Stage A: Core Pathway Validation
```bash
python -m gravitas_k.runtime.train --config gravitas_k/runtime/config.yaml
```

### Configuration
Edit `gravitas_k/runtime/config.yaml` to adjust:
- Model architecture settings
- Training hyperparameters
- Stage-specific configurations
- Hardware optimization settings

## 📊 Evaluation

```bash
python -m gravitas_k.runtime.eval \
    --model_path ./checkpoints/gravitas_k_stage_a/best \
    --eval_data ./data/eval/eval_data.json \
    --output ./results/eval_results.json
```

## 🏗️ Architecture

### Module Structure
```
gravitas_k/
├── models/
│   ├── flowing_context.py    # FC: Scan→Focus mechanism
│   ├── ccb.py                # Cognitive Core Block (A-P-C-V-S)
│   ├── prc.py                # Probabilistic Region Collapse
│   ├── hvae.py               # Hierarchical VAE
│   └── gravitas_k_model.py  # Main model with Qwen3 surgery
├── runtime/
│   ├── train.py              # Training script
│   ├── eval.py               # Evaluation script
│   └── config.yaml           # Configuration
└── tests/                    # Unit tests
```

### Training Stages

1. **Stage A**: FC + Dual-stream + CCB(A-P-V), no Challenger
2. **Stage B**: Add PRC (limited) and HVAE page-level
3. **Stage C**: Full A-P-C-V-S with all HVAE levels
4. **Stage D**: Optimization and fine-tuning

## 🧪 Testing

```bash
# Run all tests
pytest gravitas_k/tests/

# Run specific test module
pytest gravitas_k/tests/test_flowing_context.py -v
```

## 📈 Monitoring

Training progress can be monitored via:
- TensorBoard: `tensorboard --logdir ./logs/tensorboard`
- Weights & Biases: Configure in `config.yaml`

## 📝 Citation

If you use Gravitas-K in your research, please cite:

```bibtex
@software{gravitas-k,
  title = {Gravitas-K: Cognitive Architecture for Structured Reasoning},
  author = {Gravitas Team},
  year = {2024},
  url = {https://github.com/bab555/Gravitas-K}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Based on Qwen3-8B model by Alibaba Cloud
- Inspired by cognitive science and structured reasoning research
- Built with PyTorch and Hugging Face Transformers
