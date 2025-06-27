#!/usr/bin/env python3
import argparse
from pathlib import Path
from vendor_analytics.scorecard_engine import VendorScorecardEngine
from vendor_analytics.visualization import ScorecardVisualizer
from vendor_analytics.config import config
import logging

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate vendor scorecards for micro-lending assessment")
    parser.add_argument('--all', action='store_true', help="Process all vendors")
    parser.add_argument('--vendor', type=str, help="Process a specific vendor")
    parser.add_argument('--visualize', action='store_true', help="Generate visualizations")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.OUTPUT_DIR / "scorecard_generation.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        engine = VendorScorecardEngine()
        visualizer = ScorecardVisualizer()
        
        if args.all:
            logger.info("Generating scorecards for all vendors...")
            scorecards = engine.generate_all_scorecards()
            
            if args.visualize:
                # Generate visualizations for all vendors
                for vendor, metrics in scorecards.items():
                    img_path = config.OUTPUT_DIR / f"{vendor}_scorecard.png"
                    visualizer.visualize_vendor_scorecard(metrics, img_path)
                
                # Generate comparison
                cmp_path = config.OUTPUT_DIR / "vendor_comparison.png"
                visualizer.visualize_vendor_comparison(
                    {v: m for v, m in scorecards.items() if 'lending_score' in m},
                    cmp_path
                )
                
        elif args.vendor:
            logger.info(f"Generating scorecard for vendor: {args.vendor}")
            scorecard = engine.generate_vendor_scorecard(args.vendor)
            
            if args.visualize:
                img_path = config.OUTPUT_DIR / f"{args.vendor}_scorecard.png"
                visualizer.visualize_vendor_scorecard(scorecard, img_path)
                logger.info(f"Visualization saved to {img_path}")
            
        else:
            logger.warning("Please specify either --all or --vendor")
        
    except Exception as e:
        logger.error(f"Error generating scorecards: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()