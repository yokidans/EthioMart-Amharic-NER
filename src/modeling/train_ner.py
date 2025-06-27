# src/modeling/train_ner.py
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
from seqeval.metrics import classification_report as seqeval_classification_report
import logging
from pathlib import Path
import json
import warnings
import torch
from collections import defaultdict
import sys
import pandas as pd
from typing import Dict, List, Tuple

# Fix Unicode encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_ner.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define core label mapping
CORE_LABEL_LIST = ["O", "B-PRODUCT", 'B-LOC', "I-PRODUCT", "B-PRICE", "I-PRICE", "B-PHONE", "I-PHONE"]
label2id = {label: i for i, label in enumerate(CORE_LABEL_LIST)}
id2label = {i: label for i, label in enumerate(CORE_LABEL_LIST)}

# Configuration class for maintainability
class NERConfig:
    def __init__(self):
        self.valid_entities = ["PRODUCT", "PRICE", "PHONE", "LOC"]
        self.max_seq_length = 128
        self.tokenizer_name = "xlm-roberta-base"
        self.model_name = "xlm-roberta-base"
        self.output_dir = Path("models/fine_tuned")
        self.data_dir = Path("data/labeled")
        self.fallback_tag = "O"  # Default tag for unexpected labels
        
    def validate(self):
        """Validate configuration"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

config = NERConfig()

class DataValidator:
    @staticmethod
    def analyze_unexpected_tags(file_path: Path) -> Dict[str, int]:
        """Analyze and count unexpected tags in the dataset"""
        tag_counts = defaultdict(int)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and '\t' in line:
                    tag = line.split('\t')[1].strip()
                    if tag not in label2id:
                        tag_counts[tag] += 1
        return tag_counts
    
    @staticmethod
    def generate_data_quality_report(tag_counts: Dict[str, int], file_path: Path) -> None:
        """Generate comprehensive data quality report"""
        report = {
            "total_unexpected_tags": sum(tag_counts.values()),
            "unique_unexpected_tags": len(tag_counts),
            "tag_distribution": dict(tag_counts),
            "suggested_actions": []
        }
        
        if tag_counts:
            report["suggested_actions"].append(
                f"Consider adding frequent tags to label schema (e.g., '{max(tag_counts, key=tag_counts.get)}')"
            )
            report["suggested_actions"].append(
                "Review annotation guidelines for consistency"
            )
        
        report_path = file_path.parent / f"{file_path.stem}_quality_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Generated data quality report at {report_path}")

def validate_and_clean_tags(tags: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """
    Validate tags with automatic cleaning:
    1. Convert invalid tags to fallback tag
    2. Maintain statistics about conversions
    """
    cleaned_tags = []
    conversion_stats = defaultdict(int)
    
    for tag in tags:
        if tag not in label2id:
            conversion_stats[tag] += 1
            cleaned_tags.append(config.fallback_tag)
        else:
            cleaned_tags.append(tag)
    
    return cleaned_tags, conversion_stats

def parse_conll_file(file_path: Path) -> Tuple[List[Dict], Dict[str, int]]:
    """Parse CoNLL format file with enhanced validation and cleaning"""
    examples = []
    current_example = {"tokens": [], "tags": []}
    conversion_stats = defaultdict(int)
    line_num = 0
    
    try:
        # Pre-analysis of unexpected tags
        tag_counts = DataValidator.analyze_unexpected_tags(file_path)
        if tag_counts:
            DataValidator.generate_data_quality_report(tag_counts, file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                
                if not line:  # Sentence boundary
                    if current_example["tokens"]:
                        # Clean and validate tags
                        cleaned_tags, stats = validate_and_clean_tags(current_example["tags"])
                        for k, v in stats.items():
                            conversion_stats[k] += v
                        
                        current_example["tags"] = cleaned_tags
                        examples.append(current_example)
                        current_example = {"tokens": [], "tags": []}
                else:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        token, tag = parts
                        current_example["tokens"].append(token)
                        current_example["tags"].append(tag)
                    else:
                        logger.warning(f"Skipping malformed line {line_num}: {line}")
            
            # Add the last example if file doesn't end with newline
            if current_example["tokens"]:
                cleaned_tags, stats = validate_and_clean_tags(current_example["tags"])
                for k, v in stats.items():
                    conversion_stats[k] += v
                current_example["tags"] = cleaned_tags
                examples.append(current_example)
                
        if conversion_stats:
            logger.warning(f"Tag conversions in {file_path}: {dict(conversion_stats)}")
        
        logger.info(f"Parsed {file_path}: {len(examples)} examples")
        return examples, conversion_stats
        
    except Exception as e:
        logger.error(f"Error parsing {file_path} at line {line_num}: {str(e)}")
        raise

def log_data_stats(dataset: DatasetDict) -> None:
    """Enhanced dataset statistics with visualization-ready output"""
    stats = {}
    for split in dataset.keys():
        split_stats = {
            "num_examples": len(dataset[split]),
            "tag_distribution": defaultdict(int),
            "length_stats": {
                "min": float('inf'),
                "max": 0,
                "sum": 0
            }
        }
        
        for example in dataset[split]:
            # Tag statistics
            for tag in example["tags"]:
                split_stats["tag_distribution"][tag] += 1
            
            # Length statistics
            length = len(example["tokens"])
            split_stats["length_stats"]["min"] = min(split_stats["length_stats"]["min"], length)
            split_stats["length_stats"]["max"] = max(split_stats["length_stats"]["max"], length)
            split_stats["length_stats"]["sum"] += length
        
        # Calculate averages
        split_stats["length_stats"]["avg"] = split_stats["length_stats"]["sum"] / len(dataset[split])
        
        stats[split] = split_stats
    
    # Save comprehensive stats
    stats_path = config.output_dir / "dataset_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Log summary
    logger.info("\nDataset Statistics:")
    for split, data in stats.items():
        logger.info(f"\n{split.upper()} SET:")
        logger.info(f"Examples: {data['num_examples']}")
        logger.info(f"Length - Min: {data['length_stats']['min']}, Max: {data['length_stats']['max']}, Avg: {data['length_stats']['avg']:.1f}")
        
        total_tags = sum(data['tag_distribution'].values())
        logger.info("Tag Distribution:")
        for tag in CORE_LABEL_LIST:
            count = data['tag_distribution'].get(tag, 0)
            logger.info(f"{tag}: {count} ({count/total_tags:.1%})")

def load_data() -> DatasetDict:
    """Enhanced data loading with comprehensive validation"""
    config.validate()
    
    try:
        train_path = config.data_dir / 'train.conll'
        val_path = config.data_dir / 'val.conll'
        
        # Load with cleaning and validation
        train_data, train_conversions = parse_conll_file(train_path)
        val_data, val_conversions = parse_conll_file(val_path)
        
        # Create dataset
        dataset = DatasetDict({
            'train': Dataset.from_list(train_data),
            'val': Dataset.from_list(val_data)
        })
        
        # Enhanced validation
        if len(dataset['train']) == 0:
            raise ValueError("No training data found")
        
        # Log statistics
        log_data_stats(dataset)
        
        logger.info(f"Loaded {len(dataset['train'])} train and {len(dataset['val'])} val examples")
        return dataset
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}", exc_info=True)
        raise

# [Previous imports and configuration remain the same...]

def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize text and align labels with tokens, handling subword tokens properly"""
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
                    label_ids.append(label2id[tags[word_idx]])
                except IndexError:
                    logger.warning(
                        f"Tokenization mismatch in example {i}, word {word_idx}. "
                        f"Original tokens: {len(examples['tokens'][i])}, "
                        f"tags: {len(tags)}"
                    )
                    label_ids.append(-100)
            else:
                # Handle subword tokens
                previous_tag = tags[previous_word_idx]
                if previous_tag.startswith("B-"):
                    new_tag = "I-" + previous_tag[2:]
                    label_ids.append(label2id.get(new_tag, -100))
                elif previous_tag.startswith("I-"):
                    label_ids.append(label2id[tags[previous_word_idx]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx
        
        if len(label_ids) != len(tokenized_inputs["input_ids"][i]):
            logger.warning(
                f"Length mismatch in example {i}: "
                f"{len(label_ids)} labels vs {len(tokenized_inputs['input_ids'][i])} tokens"
            )
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    """Compute evaluation metrics using seqeval for proper NER evaluation"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = seqeval_classification_report(
        true_labels, 
        true_predictions, 
        output_dict=True,
        zero_division=0
    )
    
    metrics = {
        "precision": results.get("macro avg", {}).get("precision", 0),
        "recall": results.get("macro avg", {}).get("recall", 0),
        "f1": results.get("macro avg", {}).get("f1-score", 0),
    }
    
    if "accuracy" in results:
        metrics["accuracy"] = results["accuracy"]
    
    for entity in config.valid_entities:
        entity_key = f"B-{entity}"
        if entity_key in results:
            metrics[f"{entity}_precision"] = results[entity_key].get("precision", 0)
            metrics[f"{entity}_recall"] = results[entity_key].get("recall", 0)
            metrics[f"{entity}_f1"] = results[entity_key].get("f1-score", 0)
    
    return metrics

def train():
    """Main training function with comprehensive setup"""
    try:
        # Load and prepare data
        logger.info("Loading and validating dataset...")
        dataset = load_data()
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        # Tokenize dataset
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer),
            batched=True,
            batch_size=32,
            remove_columns=dataset["train"].column_names
        )
        
        # Verify tokenization
        logger.info("Example of tokenized input:")
        sample = tokenized_dataset["train"][0]
        logger.info(f"Input IDs length: {len(sample['input_ids'])}")
        logger.info(f"Labels length: {len(sample['labels'])}")
        
         # Load model
        logger.info("Loading model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=len(CORE_LABEL_LIST),
            id2label=id2label,
            label2id=label2id
        ).to(device)
        
        # Training arguments (updated for older Transformers versions)
        training_args = TrainingArguments(
            output_dir="models/fine_tuned",
            eval_strategy="epoch",  # Changed from evaluation_strategy for older versions
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            logging_dir="logs",
            logging_steps=50,
            report_to="none",
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            warmup_ratio=0.1,
            gradient_accumulation_steps=2,
        )
        
        # Initialize Trainer with DataCollator
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
        
        # Train and save model
        logger.info("Starting training...")
        train_results = trainer.train()
        
        # Save everything
        logger.info("Saving model and artifacts...")
        model_save_path = config.output_dir / "ethiomart_ner"
        trainer.save_model(str(model_save_path))
        tokenizer.save_pretrained(str(model_save_path))
        
        # Save training metrics
        with open(config.output_dir / "training_results.json", "w", encoding='utf-8') as f:
            json.dump(train_results.metrics, f, indent=2)
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        eval_results = trainer.evaluate()
        logger.info("Final evaluation results:")
        for k, v in eval_results.items():
            logger.info(f"{k}: {v}")
        
        # Save evaluation results
        with open(config.output_dir / "evaluation_results.json", "w", encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    torch.cuda.empty_cache()
    try:
        train()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise