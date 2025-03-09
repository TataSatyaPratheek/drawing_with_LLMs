"""
SVG Prompt Analyzer - Main Module (Memory-Optimized)
=================================
Entry point for the SVG Prompt Analyzer application.

This module provides the main application class and CLI interface
with memory optimizations for processing large datasets.
"""

import csv
import os
import logging
import argparse
import gc
from typing import List, Dict, Generator

from svg_prompt_analyzer.analysis.prompt_analyzer import PromptAnalyzer
from svg_prompt_analyzer.generation.svg_generator import SVGGenerator
from svg_prompt_analyzer.utils.logger import setup_logger


class SVGPromptAnalyzerApp:
    """Main application class for analyzing prompts and generating SVGs."""
    
    def __init__(self, input_file: str, output_dir: str = "output", batch_size: int = 5):
        """
        Initialize the application.
        
        Args:
            input_file: Path to input CSV file
            output_dir: Directory for output SVG files
            batch_size: Number of prompts to process in each batch
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.generator = SVGGenerator(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self) -> List[Dict[str, str]]:
        """
        Run the application on all prompts in the input file.
        
        Returns:
            List of dictionaries containing prompt ID, prompt text, and SVG file path
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting SVG Prompt Analyzer with input file: {self.input_file}")
        
        # Read prompts from CSV in batches
        all_results = []
        for batch in self.read_prompts_in_batches():
            # Create a fresh analyzer for each batch to prevent memory buildup
            analyzer = PromptAnalyzer()
            
            # Process each prompt in the batch
            batch_results = []
            for prompt in batch:
                result = self.process_prompt(prompt["id"], prompt["description"], analyzer)
                batch_results.append(result)
            
            all_results.extend(batch_results)
            
            # Explicitly run garbage collection between batches
            gc.collect()
            
            logger.info(f"Processed a batch of {len(batch_results)} prompts")
        
        logger.info(f"Processed a total of {len(all_results)} prompts")
        
        return all_results
    
    def read_prompts_in_batches(self) -> Generator[List[Dict[str, str]], None, None]:
        """
        Read prompts from CSV file in batches.
        
        Yields:
            Batches of prompts as dictionaries containing ID and description
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Reading prompts from {self.input_file} in batches of {self.batch_size}")
        
        current_batch = []
        with open(self.input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                current_batch.append({
                    "id": row["id"],
                    "description": row["description"]
                })
                
                if len(current_batch) >= self.batch_size:
                    yield current_batch
                    current_batch = []
            
            # Yield any remaining prompts
            if current_batch:
                yield current_batch
    
    def process_prompt(self, prompt_id: str, prompt_text: str, analyzer: PromptAnalyzer) -> Dict[str, str]:
        """
        Process a single prompt.
        
        Args:
            prompt_id: ID of the prompt
            prompt_text: Text of the prompt
            analyzer: PromptAnalyzer instance to use
            
        Returns:
            Dictionary containing prompt ID, prompt text, and SVG file path
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Processing prompt {prompt_id}: {prompt_text}")
        
        try:
            # Analyze prompt with memory optimization flags
            scene = analyzer.analyze_prompt(prompt_id, prompt_text)
            
            # Generate and save SVG
            svg_path = self.generator.save_svg(scene)
            
            # Return result
            return {
                "id": prompt_id,
                "prompt": prompt_text,
                "svg_path": svg_path
            }
        except Exception as e:
            logger.error(f"Error processing prompt {prompt_id}: {e}")
            return {
                "id": prompt_id,
                "prompt": prompt_text,
                "svg_path": None,
                "error": str(e)
            }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SVG Prompt Analyzer and Generator")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--output-dir", "-o", default="output", help="Directory for output SVG files")
    parser.add_argument("--log-level", "-l", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    parser.add_argument("--batch-size", "-b", type=int, default=5, help="Number of prompts to process in each batch")
    parser.add_argument("--memory-efficient", "-m", action="store_true", help="Use memory-efficient processing")
    return parser.parse_args()


def run_cli():
    """CLI entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logger(args.log_level)
    
    # Create and run application
    app = SVGPromptAnalyzerApp(args.input_file, args.output_dir, args.batch_size)
    results = app.run()
    
    # Filter out failed results
    successful_results = [r for r in results if "error" not in r]
    failed_results = [r for r in results if "error" in r]
    
    # Print summary
    print(f"\nSuccessfully processed {len(successful_results)} out of {len(results)} prompts:")
    for result in successful_results:
        print(f"  {result['id']}: {result['prompt']} -> {result['svg_path']}")
    
    if failed_results:
        print(f"\nFailed to process {len(failed_results)} prompts:")
        for result in failed_results:
            print(f"  {result['id']}: {result['prompt']} -> {result['error']}")


if __name__ == "__main__":
    run_cli()