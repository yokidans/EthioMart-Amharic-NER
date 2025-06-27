import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from collections import defaultdict
from .config import config
import logging

logger = logging.getLogger(__name__)

class VendorDataLoader:
    def __init__(self):
        config.validate()
        self.vendor_data = self._load_all_vendor_data()
        
    def _load_all_vendor_data(self) -> Dict[str, List[Dict]]:
        """Load all vendor data from processed JSON files"""
        vendor_data = defaultdict(list)
        
        for vendor_file in config.DATA_DIR.glob("vendor_*.json"):
            try:
                with open(vendor_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    vendor_name = data.get('vendor_info', {}).get('username', 'unknown')
                    vendor_data[vendor_name].extend(data['posts'])
            except Exception as e:
                logger.error(f"Error loading {vendor_file}: {str(e)}")
                continue
                
        return dict(vendor_data)
    
    def get_vendor_names(self) -> List[str]:
        """Get list of all available vendor names"""
        return list(self.vendor_data.keys())
    
    def get_vendor_posts(self, vendor_name: str) -> List[Dict]:
        """Get all posts for a specific vendor"""
        return self.vendor_data.get(vendor_name, [])
    
    def get_processed_dataframe(self, vendor_name: str) -> Optional[pd.DataFrame]:
        """Get vendor data as processed DataFrame"""
        posts = self.get_vendor_posts(vendor_name)
        if not posts:
            return None
            
        df = pd.DataFrame(posts)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract year, month, week for temporal analysis
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['week'] = df['timestamp'].dt.isocalendar().week
        
        return df