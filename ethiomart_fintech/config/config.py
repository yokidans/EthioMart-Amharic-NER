import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ScorecardConfig:
    # Path configurations
    DATA_DIR: Path = Path("data/processed")
    NER_MODEL_PATH: Path = Path("models/fine_tuned/ethiomart_ner")
    OUTPUT_DIR: Path = Path("reports/scorecards")
    
    # Metric weights for final score
    METRIC_WEIGHTS: Dict[str, float] = {
        'avg_views': 0.4,
        'posting_frequency': 0.3,
        'avg_price': 0.2,
        'product_variety': 0.1
    }
    
    # Thresholds for categorization
    ACTIVITY_THRESHOLDS: Tuple[float, float] = (1.0, 3.0)  # (low, medium) posts/week
    ENGAGEMENT_THRESHOLDS: Tuple[float, float] = (500, 2000)  # (low, medium) avg views
    
    def validate(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if not self.DATA_DIR.exists():
            raise FileNotFoundError(f"Data directory not found: {self.DATA_DIR}")
        if not self.NER_MODEL_PATH.exists():
            raise FileNotFoundError(f"NER model not found: {self.NER_MODEL_PATH}")

config = ScorecardConfig()