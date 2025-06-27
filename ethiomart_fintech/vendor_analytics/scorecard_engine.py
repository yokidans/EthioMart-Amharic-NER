import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from .data_loader import VendorDataLoader
from .metrics_calculator import VendorMetricsCalculator
from transformers import pipeline
from .config import config
import logging

logger = logging.getLogger(__name__)

class VendorScorecardEngine:
    def __init__(self):
        config.validate()
        self.data_loader = VendorDataLoader()
        self.ner_model = self._load_ner_model()
        self.metrics_calculator = VendorMetricsCalculator(self.ner_model)
    
    def _load_ner_model(self):
        """Load the fine-tuned NER model"""
        try:
            return pipeline(
                "ner",
                model=str(config.NER_MODEL_PATH),
                aggregation_strategy="simple"
            )
        except Exception as e:
            logger.error(f"Failed to load NER model: {str(e)}")
            raise
    
    def generate_all_scorecards(self) -> Dict[str, Dict]:
        """Generate scorecards for all vendors"""
        scorecards = {}
        vendor_names = self.data_loader.get_vendor_names()
        
        for vendor in vendor_names:
            try:
                scorecards[vendor] = self.generate_vendor_scorecard(vendor)
            except Exception as e:
                logger.error(f"Error processing vendor {vendor}: {str(e)}")
                continue
                
        self._save_scorecards(scorecards)
        return scorecards
    
    def generate_vendor_scorecard(self, vendor_name: str) -> Dict[str, Any]:
        """Generate comprehensive scorecard for a single vendor"""
        posts_df = self.data_loader.get_processed_dataframe(vendor_name)
        if posts_df is None or posts_df.empty:
            raise ValueError(f"No valid posts found for vendor: {vendor_name}")
        
        # Process NER entities if not already present
        if 'entities' not in posts_df.columns:
            posts_df['entities'] = posts_df['text'].apply(
                lambda x: self.ner_model(x) if pd.notnull(x) else []
            )
        
        # Calculate all metrics
        metrics = self.metrics_calculator.calculate_all_metrics(vendor_name, posts_df)
        
        # Add top post details
        top_post_id = metrics['engagement_metrics']['top_post_id']
        top_post = posts_df[posts_df['post_id'] == top_post_id].iloc[0]
        metrics['top_post_details'] = {
            'text': top_post['text'],
            'views': top_post['views'],
            'date': top_post['timestamp'].strftime('%Y-%m-%d'),
            'products': [e['word'] for e in top_post['entities'] if e['entity_group'] == 'B-PRODUCT'],
            'prices': [e['word'] for e in top_post['entities'] if e['entity_group'] == 'B-PRICE']
        }
        
        return metrics
    
    def _save_scorecards(self, scorecards: Dict[str, Dict]):
        """Save scorecards to JSON files"""
        # Save individual vendor scorecards
        for vendor, data in scorecards.items():
            vendor_file = config.OUTPUT_DIR / f"{vendor}_scorecard.json"
            with open(vendor_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save combined summary
        summary = {
            vendor: {
                'lending_score': data['lending_score'],
                'avg_views': data['engagement_metrics']['avg_views'],
                'posting_frequency': data['activity_metrics']['posting_frequency'],
                'avg_price': data['business_metrics']['avg_price']
            }
            for vendor, data in scorecards.items()
        }
        
        summary_file = config.OUTPUT_DIR / "vendor_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(scorecards)} scorecards to {config.OUTPUT_DIR}")