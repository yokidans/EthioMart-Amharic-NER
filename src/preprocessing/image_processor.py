import pytesseract
from PIL import Image
import pandas as pd
from pathlib import Path
import logging
import io
import os
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class ImageToTextConverter:
    def __init__(self, tesseract_cmd: Optional[str] = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        # Configure Tesseract for Amharic (if you have the trained data)
        self.tesseract_config = '--psm 6 -l amh+eng'
        
    def extract_text(self, image_path: Path) -> Optional[str]:
        """Extract text from an image file"""
        try:
            with Image.open(image_path) as img:
                text = pytesseract.image_to_string(img, config=self.tesseract_config)
            return text.strip() if text else None
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None
            
    def extract_text_from_bytes(self, image_bytes: bytes) -> Optional[str]:
        """Extract text from image bytes"""
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                text = pytesseract.image_to_string(img, config=self.tesseract_config)
            return text.strip() if text else None
        except Exception as e:
            logger.error(f"Error processing image bytes: {str(e)}")
            return None

class TelegramImageProcessor:
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.converter = ImageToTextConverter()
        
    def process_images_in_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process images referenced in a dataframe"""
        # This is a placeholder - implement based on your actual Telegram data structure
        # You would need to download the images first from Telegram
        df['extracted_image_text'] = None
        return df
        
    def save_image_text(self, df: pd.DataFrame, output_file: str = "image_text.csv"):
        """Save extracted image text"""
        output_path = self.output_dir / output_file
        df.to_csv(output_path, index=False)
        logger.info(f"Saved image text data to {output_path}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage (would need actual Telegram image data)
    processor = TelegramImageProcessor()
    sample_df = pd.DataFrame({'image_path': []})  # Would contain paths to downloaded images
    processed_df = processor.process_images_in_dataframe(sample_df)
    processor.save_image_text(processed_df)