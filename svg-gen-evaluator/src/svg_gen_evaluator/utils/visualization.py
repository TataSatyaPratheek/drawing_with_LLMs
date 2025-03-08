"""
Visualization utilities for evaluation results.
"""
import logging
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def create_visualizations(
    results_df: pd.DataFrame, 
    output_dir: Union[str, Path]
) -> None:
    """
    Create visualizations of evaluation results.
    
    Args:
        results_df: DataFrame containing evaluation results
        output_dir: Directory to save visualization files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Apply plot style
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        # Fallback for older matplotlib versions
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass
    
    _create_similarity_histogram(results_df, output_dir)
    _create_similarity_vs_size_scatter(results_df, output_dir)
    _create_similarity_vs_time_scatter(results_df, output_dir)
    _create_summary_plot(results_df, output_dir)


def _create_similarity_histogram(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create histogram of similarity scores."""
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['similarity'], bins=20, alpha=0.7, color='#1f77b4')
    
    plt.title('Distribution of CLIP Similarity Scores', fontsize=16)
    plt.xlabel('Similarity Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    mean_similarity = results_df['similarity'].mean()
    plt.axvline(x=mean_similarity, color='red', linestyle='--', 
                label=f'Mean: {mean_similarity:.4f}')
    
    plt.legend()
    plt.tight_layout()
    
    output_path = output_dir / "similarity_distribution.png"
    plt.savefig(output_path, dpi=100)
    plt.close()


def _create_similarity_vs_size_scatter(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create scatter plot of similarity vs SVG size."""
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['svg_size'], results_df['similarity'], 
                alpha=0.7, s=50, c='#2ca02c')
    
    plt.title('Similarity Score vs. SVG Size', fontsize=16)
    plt.xlabel('SVG Size (bytes)', fontsize=14)
    plt.ylabel('Similarity Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line if there are enough points
    if len(results_df) > 1:
        try:
            import numpy as np
            from scipy import stats
            
            slope, intercept, r_value, _, _ = stats.linregress(
                results_df['svg_size'], results_df['similarity']
            )
            
            x = np.array([results_df['svg_size'].min(), results_df['svg_size'].max()])
            y = slope * x + intercept
            
            plt.plot(x, y, color='red', linestyle='--', 
                     label=f'R²: {r_value**2:.4f}')
            plt.legend()
        except:
            # Skip trend line if scipy is not available
            pass
    
    plt.tight_layout()
    
    output_path = output_dir / "similarity_vs_size.png"
    plt.savefig(output_path, dpi=100)
    plt.close()


def _create_similarity_vs_time_scatter(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create scatter plot of similarity vs generation time."""
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['generation_time'], results_df['similarity'], 
                alpha=0.7, s=50, c='#ff7f0e')
    
    plt.title('Similarity Score vs. Generation Time', fontsize=16)
    plt.xlabel('Generation Time (seconds)', fontsize=14)
    plt.ylabel('Similarity Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_path = output_dir / "similarity_vs_time.png"
    plt.savefig(output_path, dpi=100)
    plt.close()


def _create_summary_plot(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create summary visualization with key metrics."""
    valid_count = results_df['valid'].sum()
    total_count = len(results_df)
    valid_percentage = (valid_count / total_count) * 100 if total_count > 0 else 0
    
    avg_similarity = results_df['similarity'].mean()
    avg_generation_time = results_df['generation_time'].mean()
    
    # Create a figure with metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Remove axis
    ax.axis('off')
    
    # Add header
    ax.text(0.5, 0.9, 'SVG Generation Evaluation Summary', 
            fontsize=20, ha='center', weight='bold')
    
    # Add metrics
    metrics_text = (
        f"Total examples: {total_count}\n"
        f"Valid SVGs: {valid_count} ({valid_percentage:.2f}%)\n"
        f"Average CLIP similarity: {avg_similarity:.4f}\n"
        f"Average generation time: {avg_generation_time:.2f} seconds"
    )
    
    ax.text(0.5, 0.5, metrics_text, fontsize=16, ha='center', va='center',
            bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=1'))
    
    plt.tight_layout()
    
    output_path = output_dir / "evaluation_summary.png"
    plt.savefig(output_path, dpi=100)
    plt.close()