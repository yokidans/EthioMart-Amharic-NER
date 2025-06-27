# Model Comparison & Selection Notebook - Complete Version
# With Version Compatibility and Advanced Error Handling

# Import required libraries
import os
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import traceback
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import load_dataset, DatasetDict, Dataset
from seqeval.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("model_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configuration Class with Version Awareness
class ModelComparisonConfig:
    def __init__(self):
        # Core label mapping
        self.label_list = ["O", "B-PRODUCT", 'B-LOC', "I-PRODUCT", "B-PRICE", "I-PRICE", "B-PHONE", "I-PHONE"]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        
        # Model candidates to compare
        self.model_candidates = {
            "xlm-roberta-base": {
                "name": "xlm-roberta-base",
                "description": "Large multilingual model optimized for cross-lingual tasks",
                "expected_perf": "High accuracy, moderate speed",
                "size": "~2.5GB"
            },
            "distilbert-base-multilingual-cased": {
                "name": "distilbert-base-multilingual-cased",
                "description": "Distilled version of multilingual BERT, faster inference",
                "expected_perf": "Good accuracy, fast speed",
                "size": "~500MB"
            },
            "bert-base-multilingual-cased": {
                "name": "bert-base-multilingual-cased",
                "description": "Original multilingual BERT model",
                "expected_perf": "Good accuracy, moderate speed",
                "size": "~1.5GB"
            },
            "google/rembert": {
                "name": "google/rembert",
                "description": "Rethinking embedding and transformer model, handles long sequences well",
                "expected_perf": "High accuracy, slower speed",
                "size": "~5GB"
            }
        }
        
        # Training config
        self.max_seq_length = 128
        self.batch_size = 16
        self.num_train_epochs = 3
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        
        # Paths
        self.data_dir = Path("data/labeled")
        self.output_dir = Path("models/comparison")
        self.results_file = self.output_dir / "comparison_results.json"
        
        # Version info
        self.transformers_version = self._get_transformers_version()
        self.use_legacy_args = self.transformers_version < (4, 0, 0)
        
        # Validation
        self.validate()
    
    def validate(self):
        """Validate configuration"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_transformers_version(self):
        """Get transformers version as tuple"""
        from transformers import __version__
        return tuple(map(int, __version__.split('.')[:3]))

config = ModelComparisonConfig()

# Data Preparation
def load_and_prepare_data():
    """Load and prepare the dataset for all models"""
    logger.info("Loading and preparing dataset...")
    
    def parse_conll_file(file_path):
        examples = []
        current_example = {"tokens": [], "tags": []}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Sentence boundary
                    if current_example["tokens"]:
                        examples.append(current_example)
                        current_example = {"tokens": [], "tags": []}
                else:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        token, tag = parts
                        current_example["tokens"].append(token)
                        current_example["tags"].append(tag if tag in config.label2id else "O")
            
            if current_example["tokens"]:
                examples.append(current_example)
        
        return examples
    
    train_data = parse_conll_file(config.data_dir / 'train.conll')
    val_data = parse_conll_file(config.data_dir / 'val.conll')
    
    dataset = DatasetDict({
        'train': Dataset.from_list(train_data),
        'val': Dataset.from_list(val_data)
    })
    
    logger.info(f"Loaded {len(dataset['train'])} train and {len(dataset['val'])} val examples")
    return dataset

dataset = load_and_prepare_data()

# Core Training Functions
def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize text and align labels with tokens"""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding='max_length',
        max_length=config.max_seq_length,
        is_split_into_words=True,
        return_offsets_mapping=False
    )
    
    labels = []
    for i, tags in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                try:
                    label_ids.append(config.label2id[tags[word_idx]])
                except IndexError:
                    label_ids.append(-100)
            else:
                previous_tag = tags[previous_word_idx]
                if previous_tag.startswith("B-"):
                    new_tag = "I-" + previous_tag[2:]
                    label_ids.append(config.label2id.get(new_tag, -100))
                elif previous_tag.startswith("I-"):
                    label_ids.append(config.label2id[tags[previous_word_idx]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    """Compute evaluation metrics using seqeval"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [config.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = classification_report(true_labels, true_predictions, output_dict=True)
    return {
        "precision": results["weighted avg"]["precision"],
        "recall": results["weighted avg"]["recall"],
        "f1": results["weighted avg"]["f1-score"],
        "accuracy": results.get("accuracy", 0)
    }

# Version-Compatible Training Arguments
def get_training_args(model_output_dir, use_legacy=False):
    """Get version-compatible training arguments"""
    common_args = {
        "output_dir": str(model_output_dir),
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "num_train_epochs": config.num_train_epochs,
        "weight_decay": config.weight_decay,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "fp16": torch.cuda.is_available(),
        "warmup_ratio": config.warmup_ratio,
        "gradient_accumulation_steps": 2,
        "logging_dir": str(model_output_dir / "logs"),
        "logging_steps": 50,
        "report_to": "none"
    }
    
    if use_legacy:
        return TrainingArguments(
            eval_strategy="epoch",
            save_strategy="epoch",
            **common_args
        )
    else:
        return TrainingArguments(
            evaluation_strategy="epoch",
            save_strategy="epoch",
            **common_args
        )

# Error Handling System
class ErrorHandler:
    """Advanced error handling and recovery system"""
    
    @staticmethod
    def handle_model_loading(model_name, max_retries=3):
        """Robust model loading with retries and fallbacks"""
        last_error = None
        for attempt in range(max_retries):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForTokenClassification.from_pretrained(
                    model_name,
                    num_labels=len(config.label_list),
                    id2label=config.id2label,
                    label2id=config.label2id
                )
                return tokenizer, model
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(2 ** attempt)
                
        try:
            logger.warning("Trying local_files_only as last resort")
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(config.label_list),
                id2label=config.id2label,
                label2id=config.label2id,
                local_files_only=True
            )
            return tokenizer, model
        except Exception as e:
            raise RuntimeError(f"All loading attempts failed. Last error: {str(last_error)}") from e

    @staticmethod
    def handle_training_error(trainer, error):
        """Handle training errors and attempt recovery"""
        logger.error(f"Training error: {str(error)}")
        
        if "CUDA out of memory" in str(error):
            logger.warning("Reducing batch size due to CUDA OOM")
            trainer.args.per_device_train_batch_size = max(1, trainer.args.per_device_train_batch_size // 2)
            trainer.args.gradient_accumulation_steps *= 2
            return "retry"
        
        elif "loss is nan" in str(error).lower():
            logger.warning("Encountered NaN loss, trying gradient clipping")
            trainer.args.max_grad_norm = 1.0
            return "retry"
            
        return "fail"

# Performance Measurement Functions
def measure_inference_speed(model, tokenizer, num_samples=100):
    """Measure average inference time per sample"""
    device = model.device
    sample_texts = [" ".join(ex["tokens"]) for ex in dataset["val"][:num_samples]]
    
    # Warmup
    for text in sample_texts[:5]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.max_seq_length).to(device)
        with torch.no_grad():
            _ = model(**inputs)
    
    # Measure
    start_time = time.time()
    for text in sample_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.max_seq_length).to(device)
        with torch.no_grad():
            _ = model(**inputs)
    
    return (time.time() - start_time) / num_samples

def calculate_model_size(model_dir):
    """Calculate model size in MB"""
    total_size = 0
    for path in Path(model_dir).rglob('*'):
        if path.is_file():
            total_size += path.stat().st_size
    return total_size / (1024 * 1024)

def fallback_evaluation(trainer, eval_dataset):
    """Robust fallback evaluation when standard eval fails"""
    try:
        trainer.args.per_device_eval_batch_size = max(1, trainer.args.per_device_eval_batch_size // 2)
        return trainer.evaluate()
    except Exception:
        predictions = trainer.predict(eval_dataset)
        return compute_metrics((predictions.predictions, predictions.label_ids))

# Main Training Function
def train_and_evaluate_model(model_name):
    """Version-compatible training and evaluation"""
    try:
        model_output_dir = config.output_dir / model_name.replace("/", "_")
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Training and evaluating: {model_name}")
        
        # Load model with error handling
        tokenizer, model = ErrorHandler.handle_model_loading(model_name)
        
        # Tokenize dataset
        with tqdm(total=3, desc="Preprocessing") as pbar:
            tokenized_dataset = dataset.map(
                lambda x: tokenize_and_align_labels(x, tokenizer),
                batched=True,
                batch_size=32,
                remove_columns=dataset["train"].column_names
            )
            pbar.update(1)
            
            training_args = get_training_args(model_output_dir, config.use_legacy_args)
            pbar.update(1)
            
            data_collator = DataCollatorForTokenClassification(tokenizer)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["val"],
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer
            )
            pbar.update(1)
        
        # Training with resource management
        torch.cuda.empty_cache()
        start_time = time.time()
        
        try:
            train_results = trainer.train()
            training_time = time.time() - start_time
            
            try:
                eval_results = trainer.evaluate()
            except Exception as e:
                logger.warning(f"Standard evaluation failed: {str(e)}")
                eval_results = fallback_evaluation(trainer, tokenized_dataset["val"])
            
            inference_speed = measure_inference_speed(model, tokenizer)
            model_size = calculate_model_size(model_output_dir)
            
            return {
                "model_name": model_name,
                "training_time": training_time,
                "inference_speed": inference_speed,
                "model_size": model_size,
                "metrics": eval_results,
                "output_dir": str(model_output_dir),
                "transformers_version": str(config.transformers_version)
            }
            
        finally:
            del trainer
            del model
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"Error with model {model_name}: {str(e)}", exc_info=True)
        return {
            "model_name": model_name,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Results Analysis and Visualization
def analyze_results(results):
    """Analyze and visualize comparison results"""
    successful_results = [r for r in results if "error" not in r]
    
    df_data = []
    for result in successful_results:
        df_data.append({
            "Model": result["model_name"],
            "F1-Score": result["metrics"]["eval_f1"],
            "Precision": result["metrics"]["eval_precision"],
            "Recall": result["metrics"]["eval_recall"],
            "Training Time (min)": result["training_time"] / 60,
            "Inference Speed (ms)": result["inference_speed"] * 1000,
            "Model Size (MB)": result["model_size"]
        })
    
    df = pd.DataFrame(df_data)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.barplot(data=df, x="Model", y="F1-Score")
    plt.title("F1-Score Comparison")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    plt.subplot(2, 2, 2)
    sns.barplot(data=df, x="Model", y="Training Time (min)")
    plt.title("Training Time Comparison (minutes)")
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    sns.barplot(data=df, x="Model", y="Inference Speed (ms)")
    plt.title("Inference Speed Comparison (milliseconds)")
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    sns.barplot(data=df, x="Model", y="Model Size (MB)")
    plt.title("Model Size Comparison (MB)")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(config.output_dir / "model_comparison.png")
    plt.show()
    
    print("\nModel Comparison Summary:")
    print(df.to_markdown(index=False))
    
    df["Weighted Score"] = (
        0.5 * df["F1-Score"] + 
        0.2 * (1 - (df["Inference Speed (ms)"] / df["Inference Speed (ms)"].max())) +
        0.2 * (1 - (df["Model Size (MB)"] / df["Model Size (MB)"].max())) +
        0.1 * (1 - (df["Training Time (min)"] / df["Training Time (min)"].max()))
    )
    
    best_model = df.loc[df["Weighted Score"].idxmax()]
    print("\nBest Model Selection:")
    print(f"Based on weighted criteria (50% F1, 20% speed, 20% size, 10% training time):")
    print(f"Best model: {best_model['Model']}")
    print(f"F1-Score: {best_model['F1-Score']:.3f}")
    print(f"Inference Speed: {best_model['Inference Speed (ms)']:.2f} ms")
    print(f"Model Size: {best_model['Model Size (MB)']:.2f} MB")
    
    return df, best_model

# Production Recommendations
def generate_production_recommendation(best_model):
    """Generate production deployment recommendations"""
    print("\nProduction Deployment Recommendations:")
    print("="*50)
    
    print(f"\n1. Selected Model: {best_model['Model']}")
    print(f"   - F1-Score: {best_model['F1-Score']:.3f}")
    print(f"   - Inference Speed: {best_model['Inference Speed (ms)']:.2f} ms per sample")
    print(f"   - Model Size: {best_model['Model Size (MB)']:.2f} MB")
    
    print("\n2. Deployment Considerations:")
    if best_model["Model Size (MB)"] > 500:
        print("   - Model is large (>500MB), consider:")
        print("     * Using model distillation for production")
        print("     * Deploying on GPU-enabled infrastructure")
    else:
        print("   - Model size is reasonable for production deployment")
    
    if best_model["Inference Speed (ms)"] > 50:
        print("   - Inference speed is moderate, consider:")
        print("     * Implementing caching for frequent queries")
        print("     * Using batch processing for better throughput")
    else:
        print("   - Inference speed is excellent for real-time applications")
    
    print("\n3. Monitoring Recommendations:")
    print("   - Implement performance monitoring for:")
    print("     * Model drift (F1-score over time)")
    print("     * Inference latency percentiles")
    print("     * Error rates by entity type")
    
    print("\n4. Future Improvements:")
    print("   - Consider ensemble approaches combining top models")
    print("   - Implement active learning to improve challenging cases")
    print("   - Explore domain-specific pretraining for better accuracy")

# Main Execution Flow
def main():
    """Complete execution pipeline"""
    try:
        # Run model comparison
        comparison_results = []
        for model_name, model_info in config.model_candidates.items():
            result = train_and_evaluate_model(model_name)
            comparison_results.append(result)
            
            # Save intermediate results
            with open(config.results_file, 'w') as f:
                json.dump(comparison_results, f, indent=2)
        
        # Analyze and visualize results
        results_df, best_model = analyze_results(comparison_results)
        
        # Generate recommendations
        generate_production_recommendation(best_model)
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()