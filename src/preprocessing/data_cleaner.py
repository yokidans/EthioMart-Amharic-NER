# src/preprocessing/data_cleaner.py
import pandas as pd
import re
import pytesseract
from PIL import Image
from io import BytesIO
import logging
from pathlib import Path
import zipfile
import json
from typing import Dict, List
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_cleaner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EthioMartDataCleaner:
    def __init__(self):
        self.amharic_regex = re.compile(r'[\u1200-\u137F]+')
        self.price_regex = re.compile(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(ብር|birr|br|ዶላር|dollar)', re.I)
        self.phone_regex = re.compile(r'(09\d{8}|9\d{8})')
        self.product_regex = re.compile(r'(ሽያጭ|ለመግዛት|ይግዙ|ይሸጥ|ለሽያጭ|ገዝተውልኝ)\s*([^\n]+)', re.I)
        
        # Configure paths based on scraper output
        self.raw_data_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.media_dir = Path("data/media")
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_combined_data(self) -> pd.DataFrame:
        """Load the combined data from scraper output"""
        combined_path = self.raw_data_dir / "all_messages_combined.csv"
        if not combined_path.exists():
            raise FileNotFoundError(f"Combined data not found at {combined_path}")
        
        logger.info(f"Loading combined data from {combined_path}")
        return pd.read_csv(combined_path)

    def clean_text(self, text: str) -> str:
        """Clean and normalize Amharic text"""
        if pd.isna(text) or text == '[no text]':
            return ""
            
        # Normalize prices
        text = self.price_regex.sub(r'\1 ብር', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters but keep Amharic and basic punctuation
        text = re.sub(r'[^\w\s\u1200-\u137F.,!?]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()

    def extract_entities(self, text: str) -> Dict:
        """Extract entities from cleaned text"""
        if not text:
            return {}
            
        return {
            "prices": [match[0] for match in self.price_regex.findall(text)],
            "phones": self.phone_regex.findall(text),
            "products": [match[1].strip() for match in self.product_regex.findall(text)]
        }

    def process_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process message data with cleaning and entity extraction"""
        logger.info("Processing message data...")
        
        # Clean text
        df['clean_text'] = df['text'].apply(self.clean_text)
        
        # Extract entities
        df['entities'] = df['clean_text'].apply(self.extract_entities)
        
        # Explode entities into columns
        entity_cols = pd.json_normalize(df['entities'])
        df = pd.concat([df, entity_cols], axis=1)
        
        # Add metadata
        df['has_product'] = df['products'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        df['has_price'] = df['prices'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        df['has_phone'] = df['phones'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
        
        return df

    def process_image_text(self, image_path: Path) -> Dict:
        """Extract text from a single image"""
        try:
            with Image.open(image_path) as img:
                text = pytesseract.image_to_string(img, lang='amh+eng')
                clean_text = self.clean_text(text)
                entities = self.extract_entities(clean_text)
                
                return {
                    "image_path": str(image_path),
                    "original_text": text,
                    "clean_text": clean_text,
                    "entities": entities
                }
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
            return None

    def process_all_images(self) -> pd.DataFrame:
        """Process all images in the media directory"""
        logger.info("Processing images from media directory...")
        
        image_data = []
        allowed_extensions = ['.jpg', '.jpeg', '.png']
        
        for image_path in self.media_dir.glob('*'):
            if image_path.suffix.lower() in allowed_extensions:
                result = self.process_image_text(image_path)
                if result:
                    image_data.append(result)
        
        if image_data:
            df = pd.DataFrame(image_data)
            save_path = self.processed_dir / "image_text.parquet"
            df.to_parquet(save_path)
            logger.info(f"Saved image text data to {save_path}")
            return df
        else:
            logger.warning("No valid images found to process")
            return pd.DataFrame()

    def process_zip_images(self) -> pd.DataFrame:
        """Process images from channel ZIP archives"""
        logger.info("Processing images from ZIP archives...")
        
        image_data = []
        zip_files = list(self.media_dir.glob('*_images.zip'))
        
        if not zip_files:
            logger.warning("No image ZIP files found in media directory")
            return pd.DataFrame()
        
        for zip_path in zip_files:
            try:
                with zipfile.ZipFile(zip_path) as z:
                    for img_name in z.namelist():
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            with z.open(img_name) as f:
                                img = Image.open(BytesIO(f.read()))
                                text = pytesseract.image_to_string(img, lang='amh+eng')
                                clean_text = self.clean_text(text)
                                entities = self.extract_entities(clean_text)
                                
                                image_data.append({
                                    "zip_file": zip_path.name,
                                    "image_name": img_name,
                                    "clean_text": clean_text,
                                    "entities": entities
                                })
            except Exception as e:
                logger.error(f"Error processing {zip_path.name}: {str(e)}")
        
        if image_data:
            df = pd.DataFrame(image_data)
            save_path = self.processed_dir / "zipped_image_text.parquet"
            df.to_parquet(save_path)
            logger.info(f"Saved zipped image text data to {save_path}")
            return df
        else:
            logger.warning("No valid images found in ZIP files")
            return pd.DataFrame()

    def save_metadata(self, df: pd.DataFrame) -> None:
        """Save processing metadata with proper type conversion"""
        metadata = {
            "total_messages": int(len(df)),
            "messages_with_text": int(df['clean_text'].str.len().gt(0).sum()),
            "messages_with_prices": int(df['has_price'].sum()),
            "messages_with_phones": int(df['has_phone'].sum()),
            "messages_with_products": int(df['has_product'].sum()),
            "processed_at": pd.Timestamp.now().isoformat()
        }
        
        metadata_path = self.processed_dir / "processing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved processing metadata to {metadata_path}")

    def run_clean_pipeline(self) -> None:
        """Run the complete data cleaning pipeline"""
        try:
            # Process message data
            df = self.load_combined_data()
            cleaned_df = self.process_messages(df)
            
            # Save cleaned data
            messages_path = self.processed_dir / "cleaned_messages.parquet"
            cleaned_df.to_parquet(messages_path)
            logger.info(f"Saved cleaned messages to {messages_path}")
            
            # Save metadata
            self.save_metadata(cleaned_df)
            
            # Process images (both loose and zipped)
            self.process_all_images()
            self.process_zip_images()
            
            logger.info("Data cleaning pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in cleaning pipeline: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    try:
        cleaner = EthioMartDataCleaner()
        cleaner.run_clean_pipeline()
    except Exception as e:
        logger.error(f"Fatal error in data cleaner: {str(e)}", exc_info=True)