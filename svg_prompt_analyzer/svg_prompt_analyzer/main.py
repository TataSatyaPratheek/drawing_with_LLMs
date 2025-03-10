"""
Optimized Main Entry Point for SVG Prompt Analyzer
=====================================
This module provides a unified CLI entry point for SVG generation with LLM enhancement
and aggressive optimization for both memory and performance.
"""

import os
import sys
import argparse
import logging
import json
import time
import gc
import traceback
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# Import core modules
from svg_prompt_analyzer.core import CONFIG, memoize, Profiler
from svg_prompt_analyzer.core.memory_manager import MemoryManager
from svg_prompt_analyzer.core.hardware_manager import HardwareManager
from svg_prompt_analyzer.core.resource_monitor import ResourceMonitor
from svg_prompt_analyzer.core.core_module_integration import get_core_manager

# Import utils for logging
from svg_prompt_analyzer.utils.logger import setup_logger, get_logger

# Configure logger
logger = get_logger(__name__)

# Core component instances
memory_manager = MemoryManager()
hardware_manager = HardwareManager()
resource_monitor = ResourceMonitor()


class SVGPromptAnalyzerApp:
    """
    LLM-enhanced SVG generation application with RL optimization support.
    """
    
    def __init__(
        self,
        input_file: str,
        output_dir: str = "output",
        config_file: Optional[str] = None,
        batch_size: int = 5,
        use_llm: bool = True,
        use_optimization: bool = False,
        optimization_level: int = 1,
        parallel_processing: bool = False,
        num_workers: int = 0,
        memory_efficient: bool = True
    ):
        """
        Initialize application.
        
        Args:
            input_file: Path to input CSV file with prompts
            output_dir: Directory for output SVG files
            config_file: Path to configuration file
            batch_size: Number of prompts to process in each batch
            use_llm: Whether to use LLM enhancement
            use_optimization: Whether to use RL optimization
            optimization_level: Optimization level (1-3)
            parallel_processing: Whether to process prompts in parallel
            num_workers: Number of parallel workers (0 = auto)
            memory_efficient: Whether to use memory-efficient processing
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.config_file = config_file
        self.batch_size = batch_size
        self.use_llm = use_llm
        self.use_optimization = use_optimization
        self.optimization_level = min(3, max(1, optimization_level))
        self.parallel_processing = parallel_processing
        self.memory_efficient = memory_efficient
        
        # Determine number of workers
        if num_workers <= 0:
            # Auto-determine based on available cores and memory
            self.num_workers = min(os.cpu_count() or 4, 
                                   max(1, int((os.cpu_count() or 4) * 0.75)))
        else:
            self.num_workers = num_workers
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize managers lazily
        self._llm_manager = None
        self._clip_evaluator = None
        self._ppo_optimizer = None
        
        # Initialize core manager
        self.core_manager = get_core_manager(config_file)
        
        logger.info(f"LLM-enhanced SVG application initialized with optimization level {optimization_level}")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration with fallback to default locations.
        
        Args:
            config_path: Optional explicit configuration path
            
        Returns:
            Configuration dictionary
        """
        # Default configuration
        default_config = {
            "system": {
                "memory_efficient": self.memory_efficient,
                "log_level": "INFO",
                "memory_management": {
                    "gc_threshold": 0.8,
                    "gc_collect_frequency": 5,
                    "cuda_empty_cache_frequency": 3,
                    "memory_warning_threshold": 0.9,
                    "check_interval": 5
                },
                "resource_monitoring": {
                    "check_interval": 5.0,
                    "memory_threshold": 0.8,
                    "cpu_threshold": 0.9,
                    "gpu_threshold": 0.85
                },
                "batch_processing": {
                    "optimal_batch_size": self.batch_size,
                    "max_batch_size": self.batch_size * 2,
                    "min_batch_size": 1,
                    "batch_timeout": 0.1,
                    "adaptive_batching": True,
                    "prefetch_next_batch": True,
                    "monitor_memory": True
                }
            },
            "llm": {
                "provider": "local",
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "device": "auto",
                "cache_dir": ".cache/models",
                "use_cache": True,
                "batch_size": min(8, self.batch_size),
            },
            "optimization": {
                "enabled": self.use_optimization,
                "level": self.optimization_level,
                "iterations": 5 * self.optimization_level,  # Scale iterations by level
                "population_size": 4 * self.optimization_level,  # Scale population by level
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "epsilon": 0.2,
                "batch_size": 64,
                "save_models": True,
                "models_dir": ".cache/ppo_models"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    
                # Merge configs
                self._merge_configs(default_config, loaded_config)
                logger.info(f"Loaded configuration from {config_path}")
                
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {str(e)}")
                
        # Look for config.json in default locations
        if not config_path:
            default_locations = [
                "./config.json",
                "./svg_prompt_analyzer/config.json",
                os.path.join(os.path.dirname(__file__), "../config.json")
            ]
            
            for location in default_locations:
                if os.path.exists(location):
                    try:
                        with open(location, 'r') as f:
                            loaded_config = json.load(f)
                            
                        # Merge configs
                        self._merge_configs(default_config, loaded_config)
                        logger.info(f"Loaded configuration from {location}")
                        break
                        
                    except Exception:
                        continue
        
        return default_config
    
    def _merge_configs(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Merge source config into target config.
        
        Args:
            target: Target configuration to update
            source: Source configuration to merge from
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # Recursively update nested dictionaries
                self._merge_configs(target[key], value)
            else:
                # Update or add value
                target[key] = value
    
    def _get_llm_manager(self):
        """Get LLM manager (lazy initialization)."""
        if self._llm_manager is None:
            from svg_prompt_analyzer.llm_integration.llm_manager import LLMManager
            
            # Get LLM configuration
            llm_config = self.config.get("llm", {})
            
            self._llm_manager = LLMManager(
                models_config={
                    "default": llm_config.get("model", "mistralai/Mistral-7B-Instruct-v0.2"),
                    "svg_generator": llm_config.get("svg_model", llm_config.get("model", "mistralai/Mistral-7B-Instruct-v0.2")),
                    "prompt_analyzer": llm_config.get("prompt_model", llm_config.get("model", "mistralai/Mistral-7B-Instruct-v0.2"))
                },
                cache_dir=llm_config.get("cache_dir", ".cache/models"),
                device=llm_config.get("device", "auto"),
                use_8bit=llm_config.get("use_8bit", False),
                use_4bit=llm_config.get("use_4bit", True),
                cache_responses=llm_config.get("use_cache", True)
            )
            
        return self._llm_manager
    
    def _get_clip_evaluator(self):
        """Get CLIP evaluator (lazy initialization)."""
        if self._clip_evaluator is None:
            from svg_prompt_analyzer.llm_integration.clip_evaluator import CLIPEvaluator
            
            # Get CLIP configuration
            clip_config = self.config.get("clip", {})
            
            self._clip_evaluator = CLIPEvaluator(
                model_name=clip_config.get("model", "openai/clip-vit-base-patch32"),
                device=clip_config.get("device", "auto"),
                cache_dir=clip_config.get("cache_dir", ".cache/clip"),
                batch_size=clip_config.get("batch_size", min(32, self.batch_size * 2)),
                cache_size=clip_config.get("cache_size", 2048),
                use_fp16=clip_config.get("use_fp16", None)
            )
            
        return self._clip_evaluator
    
    def _get_ppo_optimizer(self):
        """Get PPO optimizer (lazy initialization)."""
        if self._ppo_optimizer is None and self.use_optimization:
            from svg_prompt_analyzer.llm_integration.ppo_optimizer import PPOOptimizer
            
            # Get optimization configuration
            opt_config = self.config.get("optimization", {})
            
            self._ppo_optimizer = PPOOptimizer(
                llm_manager=self._get_llm_manager(),
                clip_evaluator=self._get_clip_evaluator(),
                state_dim=opt_config.get("state_dim", 64),
                action_dim=opt_config.get("action_dim", 16),
                hidden_dim=opt_config.get("hidden_dim", 128),
                lr=opt_config.get("learning_rate", 3e-4),
                gamma=opt_config.get("gamma", 0.99),
                epsilon=opt_config.get("epsilon", 0.2),
                value_coef=opt_config.get("value_coef", 0.5),
                entropy_coef=opt_config.get("entropy_coef", 0.01),
                max_grad_norm=opt_config.get("max_grad_norm", 0.5),
                update_epochs=opt_config.get("update_epochs", 4),
                batch_size=opt_config.get("batch_size", 64),
                device=opt_config.get("device", "auto"),
                cache_dir=opt_config.get("models_dir", ".cache/ppo_models")
            )
            
            # Try to load existing models
            self._ppo_optimizer.load_models()
            
        return self._ppo_optimizer
    
    def _load_prompts(self) -> List[Dict[str, Any]]:
        """
        Load prompts from input file.
        
        Returns:
            List of prompt data dictionaries
        """
        import csv
        
        prompts = []
        
        try:
            # Determine file type based on extension
            ext = os.path.splitext(self.input_file)[1].lower()
            
            if ext == '.csv':
                # Load from CSV
                with open(self.input_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Ensure required fields
                        if 'prompt' not in row:
                            # Try alternative column names
                            for alt_name in ['text', 'description', 'query']:
                                if alt_name in row:
                                    row['prompt'] = row[alt_name]
                                    break
                                    
                        if 'prompt' in row:
                            prompts.append(row)
                            
            elif ext in ['.json', '.jsonl']:
                # Load from JSON or JSONL
                with open(self.input_file, 'r', encoding='utf-8') as f:
                    if ext == '.jsonl':
                        # JSONL format (one JSON object per line)
                        for line in f:
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    if isinstance(data, dict) and 'prompt' in data:
                                        prompts.append(data)
                                except json.JSONDecodeError:
                                    continue
                    else:
                        # Regular JSON format
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'prompt' in item:
                                    prompts.append(item)
                                    
            elif ext in ['.txt']:
                # Load from text file (one prompt per line)
                with open(self.input_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            prompts.append({'prompt': line.strip()})
                            
            else:
                logger.error(f"Unsupported file type: {ext}")
                prompts = []
                
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            prompts = []
            
        logger.info(f"Loaded {len(prompts)} prompts from {self.input_file}")
        return prompts
    
    @memory_manager.memory_efficient_function
    def process_prompt(self, prompt_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Process a single prompt.
        
        Args:
            prompt_data: Prompt data dictionary
            index: Prompt index
            
        Returns:
            Result dictionary
        """
        with Profiler(f"process_prompt_{index}"):
            prompt_text = prompt_data['prompt']
            
            try:
                # Create output filename
                filename = f"svg_{index:04d}.svg"
                output_path = os.path.join(self.output_dir, filename)
                
                # Set default dimensions
                width = int(prompt_data.get('width', 800))
                height = int(prompt_data.get('height', 600))
                
                # Create scene
                from svg_prompt_analyzer.models.scene import Scene
                scene = Scene(
                    width=width,
                    height=height,
                    metadata={
                        'prompt': prompt_text,
                        'index': index
                    }
                )
                
                # Add prompt as a scene attribute for easier access
                scene.prompt = prompt_text
                
                logger.info(f"[{index}] Processing prompt: {prompt_text}")
                
                # Generate initial SVG
                llm = self._get_llm_manager()
                
                with Profiler(f"generate_svg_{index}"):
                    generation_prompt = f"""You are an expert SVG illustrator. Create a high-quality SVG illustration for the following prompt.

Prompt: "{prompt_text}"

The SVG should be semantically meaningful and directly represent the elements in the prompt.

Create complete SVG code:
```xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"
    xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>SVG illustration of {prompt_text}</title>
    <desc>Generated from prompt: {prompt_text}</desc>
    
    <!-- Definitions for patterns, gradients, etc. -->
    <defs>
"""
                    
                    svg_response = llm.generate(
                        role="svg_generator",
                        prompt=generation_prompt,
                        max_tokens=10000,
                        temperature=0.7,
                        stop_sequences=["```"]
                    )
                    
                    # Extract SVG code from response
                    from svg_prompt_analyzer.llm_integration.llm_manager import extract_svg_from_text
                    svg_code = extract_svg_from_text(svg_response)
                    
                    if not svg_code:
                        logger.warning(f"[{index}] Failed to extract SVG code from response")
                        svg_code = f"""
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"
    xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#F5F5F5" />
    <text x="{width/2}" y="{height/2}" text-anchor="middle" font-size="24" fill="#333">
        Failed to generate SVG for: {prompt_text}
    </text>
</svg>
"""
                
                # If optimization is enabled, optimize the SVG
                if self.use_optimization:
                    optimizer = self._get_ppo_optimizer()
                    opt_config = self.config.get("optimization", {})
                    
                    with Profiler(f"optimize_svg_{index}"):
                        logger.info(f"[{index}] Optimizing SVG with {opt_config.get('iterations', 5)} iterations")
                        
                        # Define callback for optimization progress
                        def optimization_callback(iteration, score, svg):
                            logger.info(f"[{index}] Optimization iteration {iteration}: score={score:.4f}")
                            
                            # Save intermediate results for higher optimization levels
                            if self.optimization_level >= 2:
                                interim_path = os.path.join(
                                    self.output_dir, 
                                    f"svg_{index:04d}_iter_{iteration:02d}.svg"
                                )
                                with open(interim_path, 'w', encoding='utf-8') as f:
                                    f.write(svg)
                        
                        # Run optimization
                        optimized_svg, final_score = optimizer.optimize(
                            scene=scene,
                            base_svg=svg_code,
                            max_iterations=opt_config.get('iterations', 5),
                            population_size=opt_config.get('population_size', 4),
                            callback=optimization_callback if self.optimization_level >= 2 else None
                        )
                        
                        svg_code = optimized_svg
                        
                        logger.info(f"[{index}] Optimization complete: final_score={final_score:.4f}")
                        
                        # Save models if enabled
                        if opt_config.get("save_models", True):
                            optimizer.save_models()
                
                # Evaluate with CLIP
                clip = self._get_clip_evaluator()
                clip_score = clip.compute_similarity(svg_code, prompt_text)
                
                # Save SVG
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(svg_code)
                    
                logger.info(f"[{index}] Saved SVG to {output_path} with CLIP score: {clip_score:.4f}")
                
                # Clean up
                memory_manager.operation_checkpoint()
                
                return {
                    'index': index,
                    'prompt': prompt_text,
                    'output_path': output_path,
                    'clip_score': clip_score,
                    'success': True,
                    'error': None
                }
                
            except Exception as e:
                logger.error(f"[{index}] Error processing prompt: {str(e)}")
                return {
                    'index': index,
                    'prompt': prompt_text,
                    'output_path': None,
                    'clip_score': 0.0,
                    'success': False,
                    'error': str(e)
                }
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the application.
        
        Returns:
            List of result dictionaries
        """
        # Start resource monitoring
        resource_monitor.start_monitoring()
        
        # Load prompts
        prompts = self._load_prompts()
        
        if not prompts:
            logger.error("No prompts to process")
            return []
            
        # Process prompts
        results = []
        
        try:
            if self.parallel_processing and self.num_workers > 1:
                # Process in parallel
                logger.info(f"Processing {len(prompts)} prompts in parallel with {self.num_workers} workers")
                
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = []
                    
                    # Submit tasks
                    for i, prompt_data in enumerate(prompts):
                        futures.append(executor.submit(self.process_prompt, prompt_data, i))
                        
                    # Collect results
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                            
                            # Log progress
                            logger.info(f"Progress: {len(results)}/{len(prompts)} ({len(results)/len(prompts)*100:.1f}%)")
                            
                            # Periodic memory cleanup
                            if len(results) % 5 == 0:
                                memory_manager.force_garbage_collection()
                                
                        except Exception as e:
                            logger.error(f"Error in parallel processing: {str(e)}")
                            
            else:
                # Process sequentially
                logger.info(f"Processing {len(prompts)} prompts sequentially")
                
                for i, prompt_data in enumerate(prompts):
                    result = self.process_prompt(prompt_data, i)
                    results.append(result)
                    
                    # Log progress
                    logger.info(f"Progress: {i+1}/{len(prompts)} ({(i+1)/len(prompts)*100:.1f}%)")
                    
                    # Periodic memory cleanup
                    if (i+1) % 5 == 0:
                        memory_manager.force_garbage_collection()
                        
        finally:
            # Clean up resources
            if self._ppo_optimizer:
                self._ppo_optimizer.cleanup()
                
            if self._clip_evaluator:
                self._clip_evaluator.close()
                
            # Force final garbage collection
            memory_manager.force_garbage_collection()
            
            # Get resource recommendations
            recommendations = resource_monitor.get_recommendations()
            if recommendations:
                logger.info("Resource optimization recommendations:")
                for rec in recommendations:
                    logger.info(f"- {rec}")
                    
            # Stop resource monitoring
            resource_monitor.stop_monitoring()
            
        return results


@memory_manager.memory_efficient_function
def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Main entry point with optimized memory management.
    
    Args:
        args: Optional parsed command line arguments
    """
    if args is None:
        args = parse_args()
    
    # Get start time
    start_time = time.time()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        log_level = args.log_level or config.get("system", {}).get("log_level", "INFO")
        setup_logger(log_level)
        
        # Initialize core manager
        core_manager = get_core_manager(args.config)
        
        logger.info(f"Starting SVG Prompt Analyzer with optimization level {args.optimization_level}")
        
        # Create application
        app = LLMEnhancedSVGApp(
            input_file=args.input_file,
            output_dir=args.output_dir,
            config_file=args.config,
            batch_size=args.batch_size,
            use_llm=args.use_llm,
            use_optimization=args.optimize,
            optimization_level=args.optimization_level,
            parallel_processing=args.parallel,
            num_workers=args.workers,
            memory_efficient=args.memory_efficient
        )
        
        # Run application
        results = app.run()
        
        # Process results
        summarize_results(results, args.output_dir, time.time() - start_time)
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Always clean up
        memory_manager.force_garbage_collection()
        memory_manager.empty_cuda_cache()
        resource_monitor.stop_monitoring()
        core_manager.shutdown()


def summarize_results(results, output_dir, elapsed_time):
    """Summarize results with memory-efficient processing."""
    if not results:
        logger.info("No results to summarize")
        return
        
    # Count success/failure
    successful = 0
    failed = 0
    total_score = 0.0
    
    for result in results:
        if result.get("success", False):
            successful += 1
            total_score += result.get("clip_score", 0.0)
        else:
            failed += 1
    
    # Calculate average score
    avg_score = total_score / successful if successful > 0 else 0.0
    
    logger.info(f"\nProcessed {len(results)} prompts in {elapsed_time:.2f}s:")
    logger.info(f"- {successful} successful")
    logger.info(f"- {failed} failed")
    logger.info(f"- Average CLIP score: {avg_score:.4f}")
    logger.info(f"\nOutput saved to: {os.path.abspath(output_dir)}")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, "summary.json")
    try:
        summary = {
            "total_prompts": len(results),
            "successful": successful,
            "failed": failed,
            "average_score": avg_score,
            "processing_time": elapsed_time,
            "timestamp": time.time(),
            "results": results
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Summary saved to: {summary_path}")
    except Exception as e:
        logger.error(f"Error saving summary: {str(e)}")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration with fallback to default."""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading config from {config_path}: {str(e)}")
    
    # Look for config.json in default locations
    default_locations = [
        "./config.json",
        "./svg_prompt_analyzer/config.json",
        os.path.join(os.path.dirname(__file__), "config.json")
    ]
    
    for location in default_locations:
        if os.path.exists(location):
            try:
                with open(location, 'r') as f:
                    return json.load(f)
            except Exception:
                continue
    
    # Return minimal default config
    return {
        "llm": {"device": "auto"},
        "clip": {"device": "auto"},
        "system": {"memory_efficient": True}
    }


def parse_args():
    """Parse command line arguments with optimized defaults."""
    parser = argparse.ArgumentParser(description="Memory-Optimized SVG Prompt Analyzer with RL")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--output-dir", "-o", default="output", help="Directory for output SVG files")
    
    # Basic arguments
    parser.add_argument("--batch-size", "-b", type=int, default=5, help="Number of prompts to process in each batch")
    parser.add_argument("--log-level", "-l", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                      help="Logging level (defaults to config setting)")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    
    # LLM-related arguments
    llm_group = parser.add_argument_group("LLM Enhancement")
    llm_group.add_argument("--use-llm", action="store_true", help="Use LLM enhancement")
    llm_group.add_argument("--optimize", "-p", action="store_true", help="Use RL optimization (requires LLM)")
    llm_group.add_argument("--optimization-level", "-O", type=int, default=1, choices=[1, 2, 3], 
                      help="Optimization level: 1=basic, 2=standard, 3=aggressive")
    
    # Performance arguments
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument("--parallel", "-P", action="store_true", help="Process prompts in parallel")
    perf_group.add_argument("--workers", "-w", type=int, default=0, 
                         help="Number of parallel workers (0 = auto)")
    perf_group.add_argument("--memory-efficient", "-m", action="store_true", help="Use memory-efficient processing")
    
    return parser.parse_args()


if __name__ == "__main__":
    main()