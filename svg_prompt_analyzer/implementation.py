"""
Example usage of SVG Prompt Analyzer.
"""

import os
import time
import pandas as pd
from svg_prompt_analyzer import SVGPromptAnalyzerApp

def main():
    """Run example SVG generation."""
    print("SVG Prompt Analyzer Example")
    
    # Create output directories
    os.makedirs("output/train", exist_ok=True)
    os.makedirs("output/test", exist_ok=True)
    
    # Check if files exist, if not create sample files
    if not os.path.exists("train.csv"):
        create_sample_csv("train.csv", 5)
        
    if not os.path.exists("test.csv"):
        create_sample_csv("test.csv", 3)
    
    # Process training data
    print("\nProcessing training data...")
    start_time = time.time()
    app1 = SVGPromptAnalyzerApp(
        "train.csv", 
        "output/train", 
        use_optimization=True,
        optimization_level=1
    )
    train_results = app1.run()
    print(f"Processed {len(train_results)} training prompts in {time.time() - start_time:.2f}s")
    
    # Process test data with higher optimization level
    print("\nProcessing test data...")
    start_time = time.time()
    app2 = SVGPromptAnalyzerApp(
        "test.csv", 
        "output/test", 
        use_optimization=True,
        optimization_level=2  # Higher optimization for test set
    )
    test_results = app2.run()
    print(f"Processed {len(test_results)} test prompts in {time.time() - start_time:.2f}s")
    
    # Print summary
    print("\nResults Summary:")
    print(f"Train: {sum(1 for r in train_results if r.get('success', False))}/{len(train_results)} successful")
    print(f"Test: {sum(1 for r in test_results if r.get('success', False))}/{len(test_results)} successful")
    
    # Calculate average CLIP scores
    train_avg_score = sum(r.get('clip_score', 0) for r in train_results) / len(train_results) if train_results else 0
    test_avg_score = sum(r.get('clip_score', 0) for r in test_results) / len(test_results) if test_results else 0
    
    print(f"Train average CLIP score: {train_avg_score:.4f}")
    print(f"Test average CLIP score: {test_avg_score:.4f}")
    
    print("\nOutput files are in the output/train and output/test directories")


def create_sample_csv(filename, num_samples):
    """Create a sample CSV file with prompts."""
    sample_prompts = [
        "A red apple on a wooden table",
        "A blue bird perched on a branch",
        "A mountain landscape at sunset",
        "A cup of coffee with steam rising",
        "A tall lighthouse on a rocky shore",
        "A bicycle leaning against a brick wall",
        "A cat sleeping on a windowsill",
        "A sailboat on a calm lake"
    ]
    
    prompts = sample_prompts[:num_samples]
    
    df = pd.DataFrame({
        'prompt': prompts,
        'width': [800] * len(prompts),
        'height': [600] * len(prompts)
    })
    
    df.to_csv(filename, index=False)
    print(f"Created sample file: {filename}")


if __name__ == "__main__":
    main()