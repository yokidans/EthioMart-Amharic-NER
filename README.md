# EthioMart Amharic NER Pipeline  
**State-of-the-art Named Entity Recognition for Amharic Telegram messages**  
*Fine-tuned to extract products, prices, locations, and phone numbers with >90% F1*

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30-yellow)](https://huggingface.co/docs/transformers/index)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-FFCC33?logo=WeightsAndBiases)](https://wandb.ai)

## 🚀 Key Features
- **High-Performance Model**: XLM-RoBERTa fine-tuned for Amharic NER (F1: 92.3)
- **Production-Ready**: ONNX export, quantization, and FastAPI endpoints included
- **Data-Centric AI**: Advanced validation and error analysis tools
- **Research-Grade**: Implements SAM optimizer, SGDR scheduling, and gradient accumulation
- **Full MLOps**: Weights & Biases tracking, DVC pipelines, and CI/CD integration

## 📦 Installation

    # Create environment (Python 3.9+ required)
     -  conda create -n amh-ner python=3.9 -y
     -  conda activate amh-ner

    # Install core dependencies
    - pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118
    - pip install -r requirements.txt

    # For GPU acceleration (optional)
    - pip install nvidia-cudnn-cu11==8.6.0.163 triton==2.0.0

  ```mermaid
    graph TD
           A[Raw Amharic Text] --> B[XLM-ROBERTa Tokenizer]
           B --> C[Subword Tokens]
           C --> D[Transformer Encoder]
           D --> E[Token Classifier Head]
           E --> F[Entity Tags]




