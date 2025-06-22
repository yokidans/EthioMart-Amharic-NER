# run_pipeline.py
from src.data_ingestion.telegram_scraper import EthioMartScraper
from src.preprocessing.data_cleaner import EthioMartPreprocessor
from src.modeling.train_ner import train

def full_pipeline():
    # Step 1: Data collection
    scraper = EthioMartScraper()
    scraper.run()
    
    # Step 2: Preprocessing
    processor = EthioMartPreprocessor()
    processor.process_existing_data()
    processor.process_images()
    
    # Step 3: Model training
    train()
    
    print("Pipeline executed successfully")

if __name__ == "__main__":
    full_pipeline()