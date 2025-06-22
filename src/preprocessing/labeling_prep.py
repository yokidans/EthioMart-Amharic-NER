# src/preprocessing/labeling_prep.py
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_labeling():
    # Load your labeled data
    labeled = pd.read_csv("data/labeled/Labeled_telegram_product_price_location.txt", sep='\t')
    
    # Split for annotation
    train, val = train_test_split(labeled, test_size=0.2)
    
    # Convert to spaCy format
    train.to_json("data/labeled/train.jsonl", orient='records', lines=True)
    val.to_json("data/labeled/val.jsonl", orient='records', lines=True)
    
    print(f"Prepared {len(train)} training and {len(val)} validation samples")

if __name__ == "__main__":
    prepare_labeling()