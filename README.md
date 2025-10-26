# Gravitas-K

**Gravitas-K** is a novel cognitive architecture for Large Language Models, combining structured reasoning (KSF's Conceptual Manifold) with dynamic attention mechanisms (Gravitas) built on top of Qwen3-8B.

## 🌟 Key Features

### Core Modules
- **FlowingContext (FC)**: Scan→Focus mechanism for adaptive attention
- **Cognitive Core Block (CCB)**: A-P-C-V-S workflow (Arbiter, Proposer, Challenger, Verifier, Synthesizer)
- **Probabilistic Region Collapse (PRC)**: Dynamic knowledge retrieval from conceptual manifold
- **Hierarchical VAE (HVAE)**: Multi-level knowledge representation (768D→256D→128D→64D→32D)
- **Dual-Stream Architecture**: Anchor vs Emergence stream processing
- **Head Vector Transfer**: Cross-layer concept propagation

### Special Capabilities
- **Native Thinking Mode**: Leverages Qwen3's `<think>` tags for interpretable reasoning
- **Structured Knowledge**: Hierarchical concept storage and retrieval
- **Noise Robustness**: Trained on augmented data for real-world reliability
- **Real-time Monitoring**: TensorBoard & W&B integration with detailed metrics

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.11+
# CUDA 12.8+ (for RTX 5090D or similar)
# 32GB+ VRAM recommended
```

### Installation
```bash
# Clone the repository
git clone https://github.com/bab555/Gravitas-K.git
cd Gravitas-K

# Install dependencies
pip install -r requirements.txt

# (Optional) Install flash-attention for your CUDA version
# pip install flash-attn --no-build-isolation
```

### Download Base Model
Place Qwen3-8B in a sibling directory:
```
parent_dir/
├── Qwen3-8B/          # Qwen3 base model
└── Gravitas-K/        # This repository
```

### Launch Interactive Demo
```bash
python scripts/quick_start.py demo \
    --base-model ../Qwen3-8B \
    --port 7860 \
    --share  # Optional: create public link
```

Visit `http://localhost:7860` to interact with the model!

### Start Training
```bash
# Stage A: FC + Dual-Stream + CCB (A-P-V)
python scripts/quick_start.py train \
    --base-model ../Qwen3-8B \
    --stage A \
    --use-wandb \
    --wandb-project Gravitas-K

# For full training control, use:
python -m gravitas_k.runtime.train --config gravitas_k/runtime/config.yaml
```

## 📚 Documentation

### Project Structure
```
Gravitas-K/
├── gravitas_k/
│   ├── models/              # Core model implementations
│   │   ├── flowing_context.py
│   │   ├── ccb.py
│   │   ├── prc.py
│   │   ├── hvae.py
│   │   └── gravitas_k_model.py
│   ├── data/                # Data processing
│   │   ├── hierarchical_parser.py
│   │   ├── think_alignment.py
│   │   └── noise_augmentation.py
│   ├── runtime/             # Training & evaluation
│   │   ├── config.yaml
│   │   ├── train.py
│   │   └── eval.py
│   ├── utils/               # Utilities
│   │   ├── model_loader.py
│   │   ├── training_monitor.py
│   │   └── gradio_interface.py
│   └── tests/               # Unit tests
├── scripts/
│   └── quick_start.py       # Quick start scripts
├── requirements.txt
└── README.md
```

### Training Stages

**Stage A** (Basic Architecture)
- FlowingContext + Dual-Stream + CCB(A-P-V)
- Focus: Establishing baseline reasoning workflow
- Duration: ~10 epochs

**Stage B** (Knowledge Integration)
- + PRC (limited trigger) + HVAE (page-level)
- Focus: Hierarchical knowledge retrieval
- Duration: ~15 epochs

**Stage C** (Full Reasoning)
- + Challenger + HVAE (full hierarchy)
- Focus: Deliberative reasoning and multi-level concepts
- Duration: ~20 epochs

See [GRAVITAS-K_工程实现总指南_Qwen3-8B.md](../GRAVITAS-K_工程实现总指南_Qwen3-8B.md) for comprehensive implementation details.

## 🧪 Testing

Run unit tests:
```bash
pytest gravitas_k/tests/ -v
```

Current test coverage:
- ✅ FlowingContext: 10/10 tests passing
- ✅ CognitiveCoreBlock: 11/11 tests passing

## 📊 Monitoring

### TensorBoard
```bash
tensorboard --logdir logs/
```

### Weights & Biases
Configure in `gravitas_k/runtime/config.yaml`:
```yaml
logging:
  wandb_project: "Gravitas-K"
  wandb_run_name: "qwen3-8b-stage-A"
```

## 🎯 Evaluation Metrics

- **Long-form QA**: Hit@k, MRR, GFR
- **Noise Robustness**: ΔP@1 under perturbations
- **Cognitive Efficiency**: Score/FLOPs ratio
- **Think-Chain Quality**: Explanation coherence, structural consistency

## 🤝 Contributing

We welcome contributions! Please see our development roadmap in [GRAVITAS-K_后续工作计划.md](../GRAVITAS-K_后续工作计划.md).

## 📄 License

- Gravitas-K code: MIT License
- Qwen3-8B base model: Apache 2.0 License

## 🔗 References

- [Qwen3 Paper & Repository](https://github.com/QwenLM/Qwen3)
- KSF Paper: "Conceptual Manifold and Structural Resonance for Knowledge Integration"
- Gravitas Paper: "Gravitas: Cognitive Architecture for Structured Reasoning in LLMs"

## 📮 Contact

- GitHub Issues: [https://github.com/bab555/Gravitas-K/issues](https://github.com/bab555/Gravitas-K/issues)
- Project Maintainer: [@bab555](https://github.com/bab555)

---

**Status**: 🚧 Early Development (TRL-3 Validation Phase)

**Last Updated**: 2025-01-26
