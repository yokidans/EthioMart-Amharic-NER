from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from .config import config
import logging

logger = logging.getLogger(__name__)

class VendorMetricsCalculator:
    def __init__(self, ner_model):
        self.ner_model = ner_model
    
    def calculate_all_metrics(self, vendor_name: str, posts_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all metrics for a vendor"""
        if posts_df.empty:
            return {}
            
        metrics = {
            'vendor_name': vendor_name,
            'activity_metrics': self._calculate_activity_metrics(posts_df),
            'engagement_metrics': self._calculate_engagement_metrics(posts_df),
            'business_metrics': self._calculate_business_metrics(posts_df),
            'temporal_metrics': self._calculate_temporal_metrics(posts_df)
        }
        
        # Calculate final score
        metrics['lending_score'] = self._calculate_lending_score(metrics)
        
        return metrics
    
    def _calculate_activity_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate posting frequency metrics"""
        # Group by week and count posts
        weekly_counts = df.groupby(['year', 'week']).size()
        
        return {
            'total_posts': len(df),
            'posting_frequency': weekly_counts.mean(),
            'posting_consistency': weekly_counts.std(),
            'active_weeks': len(weekly_counts),
            'last_post_date': df['timestamp'].max().strftime('%Y-%m-%d')
        }
    
    def _calculate_engagement_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate view-based engagement metrics"""
        views = df['views'].replace(0, np.nan)  # Handle 0 views
        
        return {
            'avg_views': views.mean(),
            'median_views': views.median(),
            'max_views': views.max(),
            'min_views': views.min(),
            'view_consistency': views.std(),
            'top_post_id': df.loc[df['views'].idxmax()]['post_id']
        }
    
    def _calculate_business_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate business metrics from NER extracted data"""
        prices = []
        products = []
        
        for _, row in df.iterrows():
            if 'entities' in row and row['entities']:
                for entity in row['entities']:
                    if entity['entity_group'] == 'B-PRICE':
                        try:
                            prices.append(float(entity['word'].replace(',', '')))
                        except (ValueError, AttributeError):
                            continue
                    elif entity['entity_group'] == 'B-PRODUCT':
                        products.append(entity['word'])
        
        avg_price = np.mean(prices) if prices else 0
        price_std = np.std(prices) if prices else 0
        
        return {
            'avg_price': avg_price,
            'price_std': price_std,
            'product_variety': len(set(products)),
            'total_products': len(products),
            'price_range': (min(prices), max(prices)) if prices else (0, 0)
        }
    
    def _calculate_temporal_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate time-based patterns"""
        df = df.sort_values('timestamp')
        
        # Engagement trends
        monthly_engagement = df.groupby(['year', 'month'])['views'].mean()
        
        return {
            'first_post_date': df['timestamp'].min().strftime('%Y-%m-%d'),
            'monthly_engagement_trend': monthly_engagement.to_dict(),
            'best_performing_month': monthly_engagement.idxmax(),
            'worst_performing_month': monthly_engagement.idxmin()
        }
    
    def _calculate_lending_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate weighted lending score"""
        score = 0
        weights = config.METRIC_WEIGHTS
        
        # Normalize and weight each metric
        score += weights['avg_views'] * self._normalize(metrics['engagement_metrics']['avg_views'], 5000)
        score += weights['posting_frequency'] * self._normalize(metrics['activity_metrics']['posting_frequency'], 5)
        score += weights['avg_price'] * self._normalize(metrics['business_metrics']['avg_price'], 10000)
        score += weights['product_variety'] * self._normalize(metrics['business_metrics']['product_variety'], 50)
        
        return min(100, score * 100)  # Cap at 100
    
    @staticmethod
    def _normalize(value: float, max_value: float) -> float:
        """Normalize value to 0-1 range based on expected max"""
        return min(1.0, max(0.0, value / max_value))