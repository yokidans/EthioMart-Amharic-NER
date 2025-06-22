# src/data_labeling/conll_annotator.py
import pandas as pd
from pathlib import Path
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/data_labeling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CoNLLAnnotator:
    def __init__(self):
        self.input_dir = Path("data/processed")
        self.output_dir = Path("data/labeled")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Entity tags per task requirements
        self.entity_tags = {
            "PRODUCT": ["B-PRODUCT", "I-PRODUCT"],
            "PRICE": ["B-PRICE", "I-PRICE"], 
            "LOC": ["B-LOC", "I-LOC"]
        }

    def load_messages(self) -> List[str]:
        """Load messages from Task 1's processed data"""
        try:
            df = pd.read_csv(self.input_dir / "cleaned_messages.csv")
            return df["text"].dropna().tolist()
        except Exception as e:
            logger.error(f"Error loading messages: {str(e)}")
            raise

    def tokenize_amharic(self, text: str) -> List[str]:
        """Basic tokenizer (replace with NLP lib if available)"""
        return text.split()

    def validate_annotation(self, tokens: List[str], labels: List[str]) -> bool:
        """Quality control for each annotation"""
        if len(tokens) != len(labels):
            logger.error(f"Length mismatch: {len(tokens)} tokens vs {len(labels)} labels")
            return False
        
        # Check I-tags follow B-tags
        for i, label in enumerate(labels):
            if label.startswith("I-"):
                prev_label = labels[i-1] if i > 0 else None
                if not (prev_label and prev_label[2:] == label[2:]):
                    logger.error(f"Invalid I-tag at position {i}: {label}")
                    return False
        return True

    def save_conll(self, messages: List[Tuple[List[str], List[str]]]):
        """Save annotated data in CoNLL format"""
        output_file = self.output_dir / "amharic_ner.conll"
        
        with open(output_file, "w", encoding="utf-8") as f:
            for tokens, labels in messages:
                if not self.validate_annotation(tokens, labels):
                    continue
                
                for token, label in zip(tokens, labels):
                    f.write(f"{token}\t{label}\n")
                f.write("\n")  # Sentence separator
        
        logger.info(f"Saved {len(messages)} annotated messages to {output_file}")

    def annotate_sample(self, num_samples: int = 50):
        """Main workflow to label samples"""
        messages = self.load_messages()[:num_samples]
        annotated = []
        
        logger.info(f"Annotating {len(messages)} messages...")
        
        # Example annotation (replace with your manual labeling)
        for msg in messages:
            tokens = self.tokenize_amharic(msg)
            labels = ["O"] * len(tokens)  # Initialize as Outside
            
            # TODO: Replace this with your actual manual labeling logic
            # This is just a demonstration pattern
            if "ብር" in msg:
                price_idx = tokens.index("ብር")
                labels[price_idx-1] = "B-PRICE"
                labels[price_idx] = "I-PRICE"
            
            annotated.append((tokens, labels))
        
        self.save_conll(annotated)

if __name__ == "__main__":
    annotator = CoNLLAnnotator()
    annotator.annotate_sample(num_samples=50)