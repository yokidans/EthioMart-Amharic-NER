import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
import pandas as pd
from .config import config
import logging

logger = logging.getLogger(__name__)

class ScorecardVisualizer:
    @staticmethod
    def visualize_vendor_scorecard(metrics: Dict[str, Any], save_path: Path = None):
        """Generate visualization for a single vendor's scorecard"""
        plt.figure(figsize=(15, 10))
        
        # Main score visualization
        plt.subplot(2, 2, 1)
        ScorecardVisualizer._plot_gauge(
            metrics['lending_score'], 
            "Lending Score", 
            (0, 100),
            ['Poor', 'Fair', 'Good', 'Excellent']
        )
        
        # Key metrics
        plt.subplot(2, 2, 2)
        key_metrics = {
            'Avg Views': metrics['engagement_metrics']['avg_views'],
            'Posts/Week': metrics['activity_metrics']['posting_frequency'],
            'Avg Price': metrics['business_metrics']['avg_price'],
            'Products': metrics['business_metrics']['product_variety']
        }
        sns.barplot(
            x=list(key_metrics.values()),
            y=list(key_metrics.keys()),
            palette="Blues_d"
        )
        plt.title("Key Performance Metrics")
        plt.xlabel("Value")
        
        # Temporal trends
        plt.subplot(2, 2, 3)
        monthly_engagement = metrics['temporal_metrics']['monthly_engagement_trend']
        months = [f"{k[0]}-{k[1]}" for k in monthly_engagement.keys()]
        values = list(monthly_engagement.values())
        sns.lineplot(x=months, y=values)
        plt.title("Monthly Engagement Trend")
        plt.xticks(rotation=45)
        plt.ylabel("Average Views")
        
        # Top post details
        plt.subplot(2, 2, 4)
        top_post = metrics['top_post_details']
        plt.text(
            0.1, 0.5,
            f"Top Post ({top_post['views']} views)\n"
            f"Date: {top_post['date']}\n"
            f"Products: {', '.join(top_post['products'][:3])}\n"
            f"Prices: {', '.join(top_post['prices'][:3])}",
            fontsize=10
        )
        plt.axis('off')
        plt.title("Top Performing Post")
        
        plt.suptitle(f"Vendor Scorecard: {metrics['vendor_name']}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def visualize_vendor_comparison(summary_data: Dict[str, Dict], save_path: Path = None):
        """Visualize comparison of multiple vendors"""
        df = pd.DataFrame.from_dict(summary_data, orient='index')
        df = df.sort_values('lending_score', ascending=False)
        
        plt.figure(figsize=(12, 8))
        
        # Score comparison
        plt.subplot(2, 2, 1)
        sns.barplot(
            y=df.index,
            x=df['lending_score'],
            palette="viridis"
        )
        plt.title("Lending Score Comparison")
        plt.xlabel("Score")
        plt.ylabel("Vendor")
        
        # Metrics heatmap
        plt.subplot(2, 2, 2)
        metrics_df = df[['avg_views', 'posting_frequency', 'avg_price']]
        metrics_df = metrics_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        sns.heatmap(
            metrics_df.T,
            annot=True,
            cmap="YlGnBu",
            fmt=".2f"
        )
        plt.title("Normalized Metrics Comparison")
        
        # Correlation
        plt.subplot(2, 2, 3)
        sns.scatterplot(
            data=df,
            x='avg_views',
            y='posting_frequency',
            size='lending_score',
            hue='lending_score',
            sizes=(50, 200),
            palette="coolwarm"
        )
        plt.title("Engagement vs Activity")
        plt.xlabel("Average Views")
        plt.ylabel("Posts per Week")
        
        # Price distribution
        plt.subplot(2, 2, 4)
        sns.boxplot(
            y=df['avg_price'],
            showfliers=False
        )
        plt.title("Price Distribution Across Vendors")
        plt.ylabel("Average Price (ETB)")
        
        plt.suptitle("Vendor Performance Comparison")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def _plot_gauge(value: float, title: str, range: tuple, segments: list):
        """Helper function to create gauge chart"""
        min_val, max_val = range
        angle = 180 * (value - min_val) / (max_val - min_val)
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        
        # Background
        ax.bar(
            x=[0, 0.5, 1, 1.5],
            width=[0.5, 0.5, 0.5, 0.5],
            height=0.5,
            bottom=2,
            color=['#ff5a5a', '#ffd35a', '#c9ff5a', '#5aff7f'],
            align='edge'
        )
        
        # Needle
        ax.plot([angle * 3.14159 / 180, angle * 3.14159 / 180], [1.5, 2.5], color='black', linewidth=3)
        ax.plot([0, 0], [1.5, 2.5], color='black', linewidth=1, alpha=0.5)
        
        # Labels
        ax.text(0, 3.5, title, ha='center', va='center', fontsize=12)
        ax.text(0, 1, f"{value:.1f}", ha='center', va='center', fontsize=16)
        
        for i, segment in enumerate(segments):
            angle = 180 * (i + 0.5) / len(segments)
            ax.text(
                angle * 3.14159 / 180, 2.2, 
                segment, 
                ha='center', 
                va='center',
                fontsize=10
            )
        
        ax.set_ylim(0, 4)
        ax.axis('off')