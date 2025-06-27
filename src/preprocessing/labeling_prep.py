# src/preprocessing/labeling_prep.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import re

# Create necessary directories first
Path("logs").mkdir(parents=True, exist_ok=True)
Path("data/labeled").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/labeling_prep.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_conll_lines(lines):
    """Parse lines from the labeled data file into sentences and tags"""
    sentences = []
    current_sentence = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            # Split on whitespace (handles tabs or spaces)
            parts = re.split(r'\s+', line)
            if len(parts) >= 2:
                token = parts[0]
                tag = parts[1]
                current_sentence.append((token, tag))
    
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences

def convert_to_conll(input_path, output_path):
    """Convert the labeled data to standard CoNLL format"""
    try:
        logger.info(f"Reading input file: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        sentences = parse_conll_lines(lines)
        
        logger.info(f"Writing output to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                for token, tag in sentence:
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")  # Sentence separator
                
        logger.info(f"Converted {len(sentences)} sentences")
        
    except Exception as e:
        logger.error(f"Error converting {input_path}: {str(e)}")
        raise

def prepare_labeling():
    """Prepare labeled data for NER training in CoNLL format"""
    try:
        # Load labeled data
        input_path = "data/labeled/labeled_telegram_product_price_location.txt"
        logger.info(f"Preparing labeled data from {input_path}...")
        
        # Check if file exists
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found at {input_path}")
        
        # Read the file to count lines and get sample data
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        sentences = parse_conll_lines(lines)
        logger.info(f"Found {len(sentences)} sentences in input file")
        
        if len(sentences) == 0:
            raise ValueError("No valid sentences found in input file")
        
        # Split into train and validation sets
        logger.info("Splitting data...")
        train_sentences, val_sentences = train_test_split(sentences, test_size=0.2, random_state=42)
        
        # Convert to CoNLL format
        logger.info("Converting to CoNLL format...")
        convert_to_conll(input_path, "data/labeled/full.conll")
        
        # Write train and validation splits
        with open("data/labeled/train.conll", 'w', encoding='utf-8') as f:
            for sentence in train_sentences:
                for token, tag in sentence:
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")
        
        with open("data/labeled/val.conll", 'w', encoding='utf-8') as f:
            for sentence in val_sentences:
                for token, tag in sentence:
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")
        
        logger.info(f"Prepared {len(train_sentences)} training and {len(val_sentences)} validation sentences")
        
        # Count tags for information
        tag_counts = {}
        for sentence in sentences:
            for _, tag in sentence:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        logger.info(f"Tag distribution: {tag_counts}")
        
    except Exception as e:
        logger.error(f"Error in prepare_labeling: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    prepare_labeling()