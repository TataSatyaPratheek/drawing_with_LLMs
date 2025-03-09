"""
Main Entry Point for SVG Prompt Analyzer
=====================================
This module provides a unified CLI entry point for SVG generation with LLM enhancement.
"""

import os
import argparse
import logging
from typing import Optional

from svg_prompt_analyzer.utils.logger import setup_logger
from svg_prompt_analyzer.llm_integration.llm_enhanced_app import LLMEnhancedSVGApp
from svg_prompt_analyzer.main import SVGPromptAnalyzerApp  # Original app

def main(args: Optional[argparse.Namespace] = None):
    """
    Main entry point for the application.
    
    Args:
        args: Optional parsed command line arguments
    """
    if args is None:
        args = parse_args()
    
    # Setup logging
    setup_logger(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Check if LLM mode is enabled
    if args.use_llm:
        logger.info("Using LLM-enhanced mode")
        app = LLMEnhancedSVGApp(
            input_file=args.input_file,
            output_dir=args.output_dir,
            config_file=args.config,
            batch_size=args.batch_size,
            use_llm=True,
            use_optimization=args.optimize,
            optimization_level=args.optimization_level,
            parallel_processing=args.parallel,
            num_workers=args.workers
        )
    else:
        logger.info("Using standard mode (no LLM)")
        # Use the original app
        app = SVGPromptAnalyzerApp(
            args.input_file, 
            args.output_dir,
            args.batch_size
        )
    
    # Run the application
    results = app.run()
    
    # Print summary
    successful_results = [r for r in results if "error" not in r]
    failed_results = [r for r in results if "error" in r]
    
    print(f"\nProcessed {len(results)} prompts:")
    print(f"- {len(successful_results)} successful")
    print(f"- {len(failed_results)} failed")
    
    # Show failed prompts if any
    if failed_results:
        print("\nFailed prompts:")
        for result in failed_results:
            print(f"  {result['id']}: {result.get('prompt', '?')} -> {result.get('error', 'Unknown error')}")
            
    # Show output location
    print(f"\nOutput saved to: {os.path.abspath(args.output_dir)}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SVG Prompt Analyzer")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--output-dir", "-o", default="output", help="Directory for output SVG files")
    
    # Basic arguments (compatible with original app)
    parser.add_argument("--batch-size", "-b", type=int, default=5, help="Number of prompts to process in each batch")
    parser.add_argument("--log-level", "-l", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="Logging level")
    
    # LLM-related arguments
    llm_group = parser.add_argument_group("LLM Enhancement")
    llm_group.add_argument("--use-llm", action="store_true", help="Use LLM enhancement")
    llm_group.add_argument("--config", "-c", help="Path to configuration file")
    llm_group.add_argument("--optimize", "-p", action="store_true", help="Use optimization (requires LLM)")
    llm_group.add_argument("--optimization-level", "-O", type=int, default=1, choices=[1, 2, 3], 
                      help="Optimization level (1-3, higher is more thorough)")
    
    # Performance arguments
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument("--parallel", "-P", action="store_true", help="Process prompts in parallel")
    perf_group.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers")
    perf_group.add_argument("--memory-efficient", "-m", action="store_true", help="Use memory-efficient processing")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()