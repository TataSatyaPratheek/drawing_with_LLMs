#!/usr/bin/env python
"""
Basic example script for evaluating a simple SVG generation model.

This script demonstrates how to use the SVG Generation Evaluator to
evaluate the built-in SimpleModel on a test dataset.

Usage:
    python basic_evaluation.py --test-data path/to/test.csv
"""
import argparse
import logging
import sys
from pathlib import Path

from svg_gen_evaluator.core.evaluator import SVGEvaluator
from svg_gen_evaluator.models.simple import SimpleModel


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a simple SVG generation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--test-data", "-t",
        type=str,
        required=True,
        help="Path to test data CSV with 'id' and 'description' columns",
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="simple_model_results",
        help="Directory to save evaluation results",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point for the example script."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    
    try:
        # Create evaluator with SimpleModel
        evaluator = SVGEvaluator(
            model_class=SimpleModel,
            output_dir=args.output_dir,
        )
        
        # Run evaluation
        results_df, avg_similarity = evaluator.evaluate(
            test_csv_path=args.test_data,
            save_images=True,
        )
        
        # Print summary
        print("\n=== Evaluation Summary ===")
        print(f"Total examples: {len(results_df)}")
        print(f"Valid SVGs: {results_df['valid'].sum()}")
        print(f"Average CLIP similarity: {avg_similarity:.4f}")
        print(f"Results saved to: {args.output_dir}")
        
        return 0
    
    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())