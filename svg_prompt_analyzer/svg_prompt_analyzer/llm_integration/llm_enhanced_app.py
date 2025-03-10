"""
LLM Enhanced SVG Application Module
=============================
This module integrates all components of the LLM-enhanced SVG generation system.
It serves as the main application entry point and orchestrates the entire process
from prompt analysis to SVG generation and optimization.
"""

import os
import csv
import time
import json
import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

# Import core optimizations
from svg_prompt_analyzer.core import CONFIG, memoize, Profiler, get_thread_pool
from svg_prompt_analyzer.core.memory_manager import MemoryManager
from svg_prompt_analyzer.core.batch_processor import BatchProcessor
from svg_prompt_analyzer.core.core_module_integration import get_core_manager

from svg_prompt_analyzer.main import SVGPromptAnalyzerApp
from svg_prompt_analyzer.llm_integration.llm_manager import LLMManager
from svg_prompt_analyzer.llm_integration.llm_prompt_analyzer import LLMPromptAnalyzer
from svg_prompt_analyzer.llm_integration.llm_svg_generator import LLMSVGGenerator
from svg_prompt_analyzer.llm_integration.clip_evaluator import CLIPEvaluator
from svg_prompt_analyzer.utils.logger import setup_logger

logger = logging.getLogger(__name__)

# Get memory manager singleton
memory_manager = MemoryManager()


class LLMEnhancedSVGApp:
    """
    Main application class integrating all LLM-enhanced components.
    Orchestrates the entire process from prompt analysis to optimization.
    """
    
    def __init__(self,
                 input_file: str,
                 output_dir: str = "output",
                 config_file: Optional[str] = None,
                 batch_size: int = 5,
                 use_llm: bool = True,
                 use_optimization: bool = False,
                 optimization_level: int = 1,
                 parallel_processing: bool = False,
                 num_workers: int = 1,
                 memory_efficient: bool = False,
                 enable_fallback: bool = True):
        """
        Initialize the LLM-enhanced SVG generation application.
        
        Args:
            input_file: Path to CSV input file with prompts
            output_dir: Directory to save output files
            config_file: Optional path to configuration file
            batch_size: Number of prompts to process in each batch
            use_llm: Whether to use LLM enhancement
            use_optimization: Whether to use optimization
            optimization_level: Optimization level (1-3)
            parallel_processing: Whether to process prompts in parallel
            num_workers: Number of parallel workers (if parallel_processing is True)
            memory_efficient: Whether to use memory-efficient mode
            enable_fallback: Whether to enable fallback to traditional methods
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.use_llm = use_llm
        self.use_optimization = use_optimization
        self.optimization_level = optimization_level
        self.parallel_processing = parallel_processing
        self.num_workers = max(1, min(num_workers, os.cpu_count() or 4))
        self.memory_efficient = memory_efficient
        self.enable_fallback = enable_fallback
        
        # Initialize core manager
        self.core_manager = get_core_manager(config_file)
        
        # Load configuration from core manager
        self.config = self.core_manager.config
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize standard app (used as fallback if needed)
        self.standard_app = SVGPromptAnalyzerApp(input_file, output_dir, batch_size) if enable_fallback else None
        
        # Get optimized model config from hardware manager
        llm_config = self.config.get("llm", {})
        model_config = self.core_manager.optimize_model_config(llm_config)
        
        # Configure LLM Manager with optimized settings
        self.llm_manager = LLMManager(
            models_config={
                "prompt_analyzer": model_config.get("prompt_analyzer_model", "mistralai/Mistral-7B-Instruct-v0.2"),
                "svg_generator": model_config.get("svg_generator_model", "deepseek-ai/deepseek-coder-6.7b-instruct")
            },
            cache_dir=model_config.get("cache_dir", ".cache/models"),
            device=model_config.get("device", "auto"),
            use_8bit=model_config.get("use_8bit", True),
            use_4bit=model_config.get("use_4bit", False)
        )
        
        # Configure CLIP evaluator (only if optimization is enabled)
        clip_config = self.config.get("clip", {})
        self.clip_evaluator = CLIPEvaluator(
            model_name=clip_config.get("model_name", "SigLIP-SoViT-400m"),
            device=clip_config.get("device", "auto"),
            cache_dir=clip_config.get("cache_dir", ".cache/clip")
        ) if self.use_optimization else None
        
        # Configure LLM Prompt Analyzer
        self.prompt_analyzer = LLMPromptAnalyzer(
            llm_manager=self.llm_manager,
            use_fallback=self.enable_fallback,
            fallback_threshold=self.config.get("fallback", {}).get("analyzer_threshold", 0.7),
            use_caching=not self.memory_efficient
        )
        
        # Configure LLM SVG Generator
        self.svg_generator = LLMSVGGenerator(
            llm_manager=self.llm_manager,
            output_dir=self.output_dir,
            use_fallback=self.enable_fallback,
            fallback_threshold=self.config.get("fallback", {}).get("generator_threshold", 0.6),
            use_caching=not self.memory_efficient,
            max_svg_size=self.config.get("system", {}).get("max_svg_size", 9500)
        )
        
        # Configure RL Optimizer (only if optimization is enabled)
        if self.use_optimization:
            opt_config = self.config.get("optimization", {})
            self.optimizer = RLOptimizer(
                llm_manager=self.llm_manager,
                clip_evaluator=self.clip_evaluator,
                max_iterations=opt_config.get("max_iterations", [1, 2, 3])[min(2, self.optimization_level - 1)],
                population_size=opt_config.get("population_size", [3, 5, 8])[min(2, self.optimization_level - 1)],
                exploration_rate=opt_config.get("exploration_rate", 0.3),
                hall_of_fame_dir=opt_config.get("hall_of_fame_dir", "hall_of_fame"),
                use_caching=not self.memory_efficient
            )
        else:
            self.optimizer = None
            
        # Create batch processor for parallel task processing
        self.batch_processor = self.core_manager.create_batch_processor(
            name="svg_generation",
            process_func=self._process_prompt_batch,
            batch_config={
                "optimal_batch_size": batch_size,
                "max_batch_size": batch_size * 2,
                "adaptive_batching": True,
                "prefetch_next_batch": True
            }
        )
            
        # Summary statistics
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None,
            "optimized": 0,
            "clip_scores": []
        }
        
    @memory_manager.memory_efficient_function
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the SVG generation process for all prompts in the input file.
        
        Returns:
            List of result dictionaries for each processed prompt
        """
        with Profiler("llm_enhanced_svg_generation"):
            logger.info(f"Starting LLM-Enhanced SVG generation with input file: {self.input_file}")
            
            # Track timing
            self.stats["start_time"] = datetime.now()
            start_time = time.time()
            
            # Read prompts from input file
            prompt_data = self._read_input_file()
            
            if not prompt_data:
                logger.error(f"No valid prompts found in input file: {self.input_file}")
                return []
                
            logger.info(f"Found {len(prompt_data)} prompts to process")
            
            # Initialize results list
            results = []
            
            # Process prompts based on configuration
            if self.parallel_processing and self.num_workers > 1:
                # Use batch processor for parallel processing
                for prompt_item in prompt_data:
                    self.batch_processor.add_item(prompt_item["id"], prompt_item)
                
                # Wait for all results with a reasonable timeout
                total_timeout = max(300, len(prompt_data) * 30)  # At least 30s per prompt
                batch_results = self.batch_processor.get_results(timeout=total_timeout)
                
                # Convert to list format
                results = list(batch_results.values())
            else:
                # Sequential processing with batches
                for i in range(0, len(prompt_data), self.batch_size):
                    batch = prompt_data[i:i + self.batch_size]
                    
                    logger.info(f"Processing batch {i // self.batch_size + 1}/{(len(prompt_data) + self.batch_size - 1) // self.batch_size} "
                              f"({len(batch)} prompts)")
                    
                    batch_results = self._process_prompt_batch(batch)
                    results.extend(batch_results)
                    
                    # Force garbage collection after each batch
                    memory_manager.force_garbage_collection()
                    
            # Update end time
            self.stats["end_time"] = datetime.now()
            total_time = time.time() - start_time
            
            # Generate summary
            summary = self._generate_summary(total_time)
            
            # Save summary
            self._save_summary(summary)
            
            logger.info(f"Completed SVG generation for {len(prompt_data)} prompts "
                      f"in {total_time:.2f}s ({self.stats['successful']} successful, "
                      f"{self.stats['failed']} failed)")
                      
            return results
        
    def _process_prompt_batch(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process a batch of prompts.
        
        Args:
            batch: List of prompt dictionaries
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for prompt_item in batch:
            result = self._process_prompt(prompt_item)
            results.append(result)
            
        return results
        
    @memory_manager.memory_efficient_function
    def _process_prompt(self, prompt_item: Dict[str, str]) -> Dict[str, Any]:
        """
        Process a single prompt through the full pipeline.
        
        Args:
            prompt_item: Dictionary with prompt information
            
        Returns:
            Result dictionary
        """
        with Profiler(f"process_prompt_{prompt_item.get('id', 'unknown')}"):
            prompt_id = prompt_item.get("id", "unknown")
            prompt_text = prompt_item.get("description", "")
            
            logger.info(f"Processing prompt: {prompt_id} - '{prompt_text}'")
            
            try:
                # Update statistics
                self.stats["processed"] += 1
                
                # 1. Use LLM to analyze the prompt and create a scene
                logger.info(f"Analyzing prompt: {prompt_id}")
                scene = self.prompt_analyzer.analyze_prompt(prompt_id, prompt_text)
                
                # 2. Generate SVG from the scene
                logger.info(f"Generating SVG for prompt: {prompt_id}")
                svg_path = self.svg_generator.save_svg(scene)
                
                # Read the generated SVG
                with open(svg_path, 'r', encoding='utf-8') as f:
                    svg_code = f.read()
                    
                # 3. Optimize the SVG if requested
                clip_score = None
                if self.use_optimization:
                    logger.info(f"Optimizing SVG for prompt: {prompt_id}")
                    
                    # Load CLIP evaluator if not already loaded
                    if not self.clip_evaluator.model_loaded:
                        self.clip_evaluator.load_model()
                        
                    # Optimize SVG
                    optimized_svg, clip_score = self.optimizer.optimize(
                        scene=scene,
                        base_svg=svg_code,
                        optimization_level=self.optimization_level
                    )
                    
                    # Save optimized SVG if score improved
                    if clip_score > 0:
                        optimized_path = os.path.join(self.output_dir, f"{prompt_id}_optimized.svg")
                        with open(optimized_path, 'w', encoding='utf-8') as f:
                            f.write(optimized_svg)
                            
                        # Update the main SVG with the optimized version
                        with open(svg_path, 'w', encoding='utf-8') as f:
                            f.write(optimized_svg)
                            
                        # Update statistics
                        self.stats["optimized"] += 1
                        self.stats["clip_scores"].append(clip_score)
                        
                        # Update SVG code for result
                        svg_code = optimized_svg
                
                # 4. Evaluate final SVG similarity if CLIP is available
                if clip_score is None and self.clip_evaluator and self.clip_evaluator.model_loaded:
                    clip_score = self.clip_evaluator.compute_similarity(svg_code, prompt_text)
                    if clip_score > 0:
                        self.stats["clip_scores"].append(clip_score)
                
                # Update statistics
                self.stats["successful"] += 1
                
                # Create result
                result = {
                    "id": prompt_id,
                    "prompt": prompt_text,
                    "svg_path": svg_path,
                    "success": True
                }
                
                # Add CLIP score if available
                if clip_score is not None:
                    result["clip_score"] = clip_score
                    
                return result
                
            except Exception as e:
                logger.error(f"Error processing prompt {prompt_id}: {str(e)}")
                
                # Update statistics
                self.stats["failed"] += 1
                
                # Create error result
                return {
                    "id": prompt_id,
                    "prompt": prompt_text,
                    "error": str(e),
                    "success": False
                }
            
    @memoize
    def _read_input_file(self) -> List[Dict[str, str]]:
        """
        Read prompts from the input CSV file.
        
        Returns:
            List of prompt dictionaries
        """
        prompts = []
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Ensure required fields are present
                    if "id" not in row or "description" not in row:
                        logger.warning(f"Skipping row: missing required fields. Fields: {list(row.keys())}")
                        continue
                        
                    prompts.append({
                        "id": row["id"],
                        "description": row["description"]
                    })
                    
        except Exception as e:
            logger.error(f"Error reading input file: {str(e)}")
            
        return prompts
        
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """
        Generate a summary of the generation process.
        
        Args:
            total_time: Total processing time in seconds
            
        Returns:
            Summary dictionary
        """
        # Calculate statistics
        processed = self.stats["processed"]
        successful = self.stats["successful"]
        failed = self.stats["failed"]
        optimized = self.stats["optimized"]
        success_rate = successful / processed if processed > 0 else 0
        
        # Calculate CLIP score statistics
        clip_scores = self.stats["clip_scores"]
        avg_clip_score = sum(clip_scores) / len(clip_scores) if clip_scores else None
        max_clip_score = max(clip_scores) if clip_scores else None
        min_clip_score = min(clip_scores) if clip_scores else None
        
        # Get resource statistics
        resource_stats = self.core_manager.get_resource_stats()
        
        # Create summary
        summary = {
            "total_prompts": processed,
            "successful": successful,
            "failed": failed,
            "success_rate": success_rate,
            "total_time_seconds": total_time,
            "avg_time_per_prompt": total_time / processed if processed > 0 else 0,
            "start_time": self.stats["start_time"].isoformat() if self.stats["start_time"] else None,
            "end_time": self.stats["end_time"].isoformat() if self.stats["end_time"] else None,
            "configuration": {
                "use_llm": self.use_llm,
                "use_optimization": self.use_optimization,
                "optimization_level": self.optimization_level if self.use_optimization else None,
                "parallel_processing": self.parallel_processing,
                "num_workers": self.num_workers if self.parallel_processing else 1,
                "memory_efficient": self.memory_efficient,
                "enable_fallback": self.enable_fallback,
                "device": self.core_manager.get_optimal_device()
            },
            "resources": {
                "memory_usage": resource_stats.get("memory", {}),
                "cpu_usage": resource_stats.get("cpu", {}),
                "gpu_usage": resource_stats.get("gpu", {})
            },
            "recommendations": self.core_manager.get_resource_recommendations()
        }
        
        # Add optimization statistics if applicable
        if self.use_optimization:
            summary["optimization"] = {
                "optimized_count": optimized,
                "optimization_rate": optimized / successful if successful > 0 else 0,
                "clip_scores": {
                    "average": avg_clip_score,
                    "max": max_clip_score,
                    "min": min_clip_score
                }
            }
            
        return summary
        
    def _save_summary(self, summary: Dict[str, Any]) -> None:
        """
        Save summary to a JSON file.
        
        Args:
            summary: Summary dictionary
        """
        try:
            summary_path = os.path.join(self.output_dir, "summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"Saved summary to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
    
    def shutdown(self) -> None:
        """Clean up resources when done."""
        logger.info("Shutting down LLM Enhanced SVG App")
        
        # Stop batch processor
        self.batch_processor.stop(wait_complete=True)
        
        # Shutdown core manager
        self.core_manager.shutdown()
        
        # Release model resources
        if self.clip_evaluator:
            self.clip_evaluator.close()


# CLI entry point
def run_cli() -> None:
    """Command-line interface entry point for the LLM-enhanced app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-Enhanced SVG Generator")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--output-dir", "-o", default="output", help="Directory for output SVG files")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--batch-size", "-b", type=int, default=5, help="Number of prompts to process in each batch")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM enhancement")
    parser.add_argument("--optimize", "-p", action="store_true", help="Enable optimization")
    parser.add_argument("--optimization-level", "-O", type=int, default=1, choices=[1, 2, 3], 
                        help="Optimization level (1-3, higher is more thorough)")
    parser.add_argument("--parallel", "-P", action="store_true", help="Process prompts in parallel")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--memory-efficient", "-m", action="store_true", help="Enable memory-efficient mode")
    parser.add_argument("--disable-fallback", "-D", action="store_true", help="Disable fallback to traditional methods")
    parser.add_argument("--log-level", "-l", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    setup_logger(args.log_level)
    
    # Create and run the application
    app = LLMEnhancedSVGApp(
        input_file=args.input_file,
        output_dir=args.output_dir,
        config_file=args.config,
        batch_size=args.batch_size,
        use_llm=not args.no_llm,
        use_optimization=args.optimize,
        optimization_level=args.optimization_level,
        parallel_processing=args.parallel,
        num_workers=args.workers,
        memory_efficient=args.memory_efficient,
        enable_fallback=not args.disable_fallback
    )
    
    # Run the application
    try:
        results = app.run()
        
        # Print summary
        successful = sum(1 for r in results if r.get("success", False))
        failed = sum(1 for r in results if not r.get("success", False))
        
        print(f"\nProcessed {len(results)} prompts:")
        print(f"- {successful} successful")
        print(f"- {failed} failed")
        
        if app.use_optimization and app.stats["clip_scores"]:
            avg_score = sum(app.stats["clip_scores"]) / len(app.stats["clip_scores"])
            print(f"- Average CLIP score: {avg_score:.4f}")
            
        print(f"\nOutput saved to: {os.path.abspath(app.output_dir)}")
        
        # Clean up
        app.shutdown()
        
    except KeyboardInterrupt:
        print("\nOperation canceled by user")
        app.shutdown()
    except Exception as e:
        logger.error(f"Error running application: {str(e)}")
        app.shutdown()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    run_cli()