# src/preprocessing/data_pipeline.py
import re
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import pytesseract
from PIL import Image
import json
from sklearn.model_selection import train_test_split
from collections import Counter
import emoji
import unicodedata
import tempfile
import os
import shutil

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/data_pipeline.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EthioMartPreprocessor:
    def __init__(self):
        """Initialize with enhanced validation"""
        self.raw_dir = Path("data/labeled").absolute()
        self.processed_dir = Path("data/processed").absolute()
        self.labeled_dir = Path("data/labeled").absolute()
        
        # Setup directories with atomic verification
        self._setup_directories()
        
        # Enhanced entity patterns
        self._init_regex_patterns()
        
        # Configure Tesseract for Amharic OCR
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.tesseract_config = '--psm 6 -l amh+eng'

    def _init_regex_patterns(self):
        """Compile optimized regex patterns for Amharic e-commerce"""
        self.price_pattern = re.compile(
            r'(?:^|\s)(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(ብር|br|birr|ዶላር|dollar)?\b',
            re.IGNORECASE
        )
        self.phone_pattern = re.compile(
            r'(?<!\d)(09\d{8}|9\d{8})(?!\d)'
        )
        self.product_pattern = re.compile(
            r'(?:^|\s)(ሽያጭ|ለመግዛት|ይግዙ|ይሸጥ|ለሽያጭ|ገዝተውልኙ)\s*([^\n\d]+?)(?=\s{2}|$)',
            re.IGNORECASE
        )
        self.sku_pattern = re.compile(
            r'^[A-Za-z0-9][A-Za-z0-9/-]{2,}$'
        )

    def _setup_directories(self):
        """Validate and create directory structure with atomic checks"""
        try:
            for dir_path in [self.raw_dir, self.processed_dir, self.labeled_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
                
                # Test write permissions
                test_file = dir_path / ".perm_test"
                try:
                    test_file.touch()
                    test_file.write_text("test")
                    test_file.unlink()
                except Exception as e:
                    raise PermissionError(f"Can't write to {dir_path}: {str(e)}")
            
            logger.info("Directory structure validated")
        except Exception as e:
            logger.critical(f"Filesystem initialization failed: {str(e)}")
            raise

    def _normalize_text(self, text: str) -> str:
        """Advanced text normalization"""
        if not text or not isinstance(text, str):
            return ""
            
        # Phase 1: Emoji and symbol handling
        text = emoji.replace_emoji(text, replace="[EMOJI]")
        text = unicodedata.normalize('NFKC', text)
        
        # Phase 2: Special character cleaning
        text = re.sub(r'[^\w\s\u1200-\u137F.,!?]', ' ', text)
            
        # Phase 3: Price standardization
        text = self.price_pattern.sub(r' \1 ብር ', text)
        
        # Final cleanup
        return ' '.join(text.split()).strip()

    def _fix_entity_tags(self, tokens: List[str], tags: List[str]) -> List[str]:
        """Apply all entity correction rules with position tracking"""
        new_tags = list(tags)  # Create a mutable copy
        
        # Rule 1: Complete price entities (number + unit)
        i = 0
        while i < len(tokens):
            if new_tags[i].startswith(("B-PRICE", "I-PRICE")):
                # Check if next token is a price unit
                if (i + 1 < len(tokens) and 
                    tokens[i+1] in ["ብር", "birr", "br"] and
                    new_tags[i+1] == "O"):
                    new_tags[i+1] = "I-PRICE"
                i += 2
            else:
                i += 1
                
        # Rule 2: Filter invalid product labels
        for i, token in enumerate(tokens):
            if (new_tags[i].startswith(("B-PRODUCT", "I-PRODUCT")) and 
                self.sku_pattern.match(token)):
                new_tags[i] = "O"
                
        # Rule 3: Enforce strict IOB2 format
        current_entity = None
        for i, tag in enumerate(new_tags):
            if tag.startswith("B-"):
                current_entity = tag[2:]
            elif tag.startswith("I-"):
                if current_entity != tag[2:]:
                    new_tags[i] = f"B-{tag[2:]}"  # Convert to B- if broken sequence
                    current_entity = tag[2:]
            else:
                current_entity = None
                
        return new_tags

    def _read_input_file(self, filepath: Path) -> List[List[Tuple[str, str]]]:
        """Robust file reading with format auto-detection"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Auto-detect delimiter
            first_line = content.split('\n')[0] if content else ""
            delimiter = '\t' if '\t' in first_line else r'\s+'
            
            sentences = []
            current_sentence = []
            
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue
                    
                parts = re.split(delimiter, line, maxsplit=1)
                if len(parts) == 2:
                    current_sentence.append((parts[0], parts[1]))
                    
            return sentences
            
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {str(e)}")
            raise

    def _validate_sentence(self, tokens: List[str], tags: List[str]) -> bool:
        """Comprehensive sentence validation"""
        if len(tokens) != len(tags):
            logger.warning(f"Length mismatch: {len(tokens)} tokens vs {len(tags)} tags")
            return False
            
        # Check IOB2 compliance
        prev_tag = None
        for i, tag in enumerate(tags):
            if tag.startswith("I-"):
                if not prev_tag or prev_tag[2:] != tag[2:]:
                    logger.debug(f"Orphan I- tag at position {i}: {tag}")
                    return False
            elif tag.startswith("B-"):
                if prev_tag and prev_tag.startswith("I-"):
                    logger.debug(f"B- tag after I- at position {i}")
                    return False
            prev_tag = tag
            
        return True

    def _atomic_write_conll(self, data: List, output_path: Path):
        """Transactional file writing with validation"""
        temp_path = output_path.with_suffix('.tmp')
        try:
            # Write to temporary file
            with open(temp_path, 'w', encoding='utf-8') as f:
                for sentence in data:
                    for token, tag in sentence:
                        f.write(f"{token}\t{tag}\n")
                    f.write("\n")
            
            # Atomic replace
            temp_path.replace(output_path)
            
            # Verify write
            if not output_path.exists():
                raise IOError(f"File not created: {output_path}")
                
            logger.info(f"Wrote {len(data)} sentences to {output_path}")
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def prepare_labeled_data(self, input_path: Path):
        """End-to-end labeled data processing"""
        logger.info(f"Processing input file: {input_path}")
        
        # Read and validate input
        sentences = self._read_input_file(input_path)
        if not sentences:
            raise ValueError("No valid sentences found in input file")
            
        # Process each sentence
        processed_sentences = []
        error_count = 0
        
        for sentence in sentences:
            try:
                tokens, tags = zip(*sentence)
                
                # Apply correction rules
                fixed_tags = self._fix_entity_tags(tokens, tags)
                
                # Validate after correction
                if not self._validate_sentence(tokens, fixed_tags):
                    error_count += 1
                    continue
                    
                processed_sentences.append(list(zip(tokens, fixed_tags)))
            except Exception as e:
                logger.warning(f"Error processing sentence: {str(e)}")
                error_count += 1
                
        logger.info(f"Processed {len(processed_sentences)} sentences ({error_count} errors)")
        
        # Split into train/validation
        train, val = train_test_split(
            processed_sentences,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )
        
        # Write outputs
        self._atomic_write_conll(train, self.labeled_dir / "train.conll")
        self._atomic_write_conll(val, self.labeled_dir / "val.conll")
        
        # Generate quality report
        self._generate_quality_report(train, val)

    def _generate_quality_report(self, train: List, val: List):
        """Comprehensive data quality analysis"""
        all_sentences = train + val
        
        # Tag distribution
        tag_counts = Counter(tag for sentence in all_sentences for _, tag in sentence)
        
        # Entity statistics
        entity_stats = {
            "PRICE": {"correct": 0, "total": 0},
            "PRODUCT": {"correct": 0, "total": 0},
            "LOC": {"correct": 0, "total": 0}
        }
        
        for sentence in all_sentences:
            _, tags = zip(*sentence)
            for i, tag in enumerate(tags):
                if tag.startswith("B-"):
                    entity_type = tag[2:]
                    if entity_type in entity_stats:
                        entity_stats[entity_type]["total"] += 1
                        # Check if properly terminated
                        if i+1 >= len(tags) or not tags[i+1].startswith(("I-", "B-")):
                            entity_stats[entity_type]["correct"] += 1
        
        # Write report
        report_path = self.processed_dir / "data_quality_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                "tag_distribution": tag_counts,
                "entity_consistency": entity_stats,
                "split_counts": {
                    "train": len(train),
                    "val": len(val)
                }
            }, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Quality report saved to {report_path}")

    def run_pipeline(self):
        """Orchestrate the complete processing pipeline"""
        try:
            # Verify input file exists
            input_file = self.raw_dir / "labeled_telegram_product_price_location.txt"
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
                
            # Process data
            self.prepare_labeled_data(input_file)
            
            # Final verification
            assert (self.labeled_dir / "train.conll").exists()
            assert (self.labeled_dir / "val.conll").exists()
            
            logger.info("Pipeline completed successfully")
        except Exception as e:
            logger.critical(f"Pipeline failed: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    processor = EthioMartPreprocessor()
    processor.run_pipeline()