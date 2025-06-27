# EthioMart Amharic NER Project
**Building a FinTech-Ready Vendor Intelligence System**  
*Fine-tuned NER for Telegram Commerce Data with Micro-Lending Analytics*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-%23EE4C2C)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30-yellow)](https://huggingface.co/docs/transformers)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-FFCC33?logo=WeightsAndBiases)](https://wandb.ai)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

## 🔍 Project Overview
### Business Context
  ```mermaid
          graph LR
                A[Telegram Channels] --> B[EthioMart Platform]
                B --> C[Vendor Analytics]
                C --> D[Micro-Lending Decisions]
```
## Core Problem:

"Transform messy Telegram posts into a smart FinTech engine that reveals which vendors are best candidates for loans."
   Key Entities:
           PRODUCT: የቤት ንጥረ ነገር (Home appliances)
           PRICE: 2500 ብር (ETB)
           LOC: መከለር (Mekelle)
           PHONE: 0911223344

  ## 🛠️ Technical Architecture
  🎯 Task Breakdown
##  Task 1: Data Pipeline
### Elite Implementation
 Key Features:
      - Multi-channel async scraping
      - Image OCR with Tesseract-Amharic
      - Atomic writes with checksum validation
  ## Task 2: Annotation Protocol
  ### CoNLL Format Standard
   ```bash
          ሽያጭ    B-PRODUCT
           አለኝ    O
           በመከለር  B-LOC
           2500    B-PRICE
            ብር     I-PRICE
  ```
 ### Quality Control:

  - IOB2 validation script
  - Inter-annotator agreement > 0.85
  - Entity consistency checks

 ## Task 3: Model Fine-Tuning
   ### Hyperparameters

 ``` yml
   training:
  model: xlm-roberta-large
  batch_size: 16
  grad_accum: 4
  lr: 2e-5
  epochs: 5
  warmup: 0.1
optim:
  use_sam: true  # Sharpness-Aware Minimization
  scheduler: cosine_with_restarts
```
## Task 4: Model Comparison
## 🏆 Model Benchmark Results

| Model            | F1-Score | VRAM   | Latency |
|------------------|----------|--------|---------|
| **XLM-RoBERTa-L** | 92.3     | 24 GB  | 58 ms   |
| **AfroXLMR**      | 89.7     | 18 GB  | 63 ms   |
| **mBERT**         | 85.2     | 16 GB  | 47 ms   |

**Key Insights**:
- 🥇 **XLM-RoBERTa-L** delivers best accuracy (F1 92.3) but requires most VRAM
- ⚡ **mBERT** offers fastest inference (47ms) with acceptable accuracy tradeoff
- ⚖️ **AfroXLMR** provides balanced memory/performance for mid-range GPUs

## Task 5: Model Interpretability
### SHAP Analysis
  ```python
   explainer = shap.Explainer(model)
   shap_values = explainer([sample_text])
   shap.plots.text(shap_values)

```
### Key Insights:
  - Price detection relies on ብር/ETB markers
  - Location sensitivity to preposition patterns
  - Product name ambiguity in short descriptions

## Task 6: Vendor Scorecard
### Lending Algorithm
  ```python
  def calculate_score(vendor: Vendor) -> float:
     return (
        0.4 * norm_avg_views + 
        0.3 * posting_frequency + 
        0.2 * price_consistency + 
        0.1 * entity_diversity
    )
```
## 📊 Vendor Scorecard Sample Output

| Vendor         | Avg Views | Posts/Week | Avg Price | Score |
|----------------|-----------|------------|-----------|-------|
| ሻገር           | 1,240     | 28         | 560 ETB   | 82.1  |
| ኢትዮ ኤሌክ       | 890       | 15         | 1,200 ETB | 68.4  |

**Metric Definitions**:
- **Avg Views**: 30-day average post engagements  
- **Posts/Week**: Consistent posting frequency  
- **Avg Price**: Median product price in ETB  
- **Score**: Weighted lending risk score (0-100)

# 🛠️ Development Setup
```bash
 conda create -n ethiomart python=3.9 -y
conda activate ethiomart

# Install with GPU support
pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Run training
python -m src.modeling.train_ner \
    --data_dir data/processed \
    --model xlm-roberta-large \
    --use_sam \
    --batch_size 16


 




  
