"""Setup script for Gravitas-K cognitive architecture."""

from setuptools import setup, find_packages

setup(
    name="gravitas-k",
    version="0.1.0",
    description="Gravitas-K: Structured reasoning architecture based on Qwen3-8B",
    author="Gravitas Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.0",
        "transformers>=4.51.0",
        "accelerate>=0.26.0",
        "safetensors>=0.4.0",
        "faiss-gpu>=1.7.4",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.12.0",
            "ruff>=0.1.0",
            "isort>=5.13.0",
        ],
        "training": [
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
            "datasets>=2.16.0",
        ],
        "ui": [
            "gradio>=4.0.0",
            "streamlit>=1.29.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gravitas-train=gravitas_k.runtime.train:main",
            "gravitas-eval=gravitas_k.runtime.eval:main",
        ],
    },
)
