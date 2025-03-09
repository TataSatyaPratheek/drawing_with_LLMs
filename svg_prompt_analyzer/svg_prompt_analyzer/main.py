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
import weakref
import threading
import tracemalloc
from typing import Optional, Dict, Any, List, Tuple
from functools import wraps
from pathlib import Path

# Import utils first to set up logging
from svg_prompt_analyzer.utils.logger import setup_logger

# Lazy imports to reduce startup time
_llm_enhanced_app = None
_svg_prompt_analyzer_app = None

# Global memory monitoring
_memory_monitor = None
_last_gc_collection = 0
_last_cuda_emptied = 0
_gc_collection_frequency = 5  # Operations between collections
_cuda_empty_frequency = 3     # Operations between CUDA cache clearing
_op_counter = 0
_cuda_available = False

# Memory Monitor Thread
class MemoryMonitor(threading.Thread):
    """Thread for monitoring memory usage and triggering GC when needed."""
    
    def __init__(self, threshold: float = 0.8, check_interval: int = 5):
        """Initialize memory monitor.
        
        Args:
            threshold: Memory threshold (0.0-1.0) to trigger GC
            check_interval: Check interval in seconds
        """
        super().__init__(daemon=True)
        self.threshold = threshold
        self.check_interval = check_interval
        self.running = True
        self.logger = logging.getLogger(__name__).getChild("memory")
        
        # Start tracemalloc for memory tracking
        tracemalloc.start()
        
        # Check for CUDA
        try:
            import torch
            global _cuda_available
            _cuda_available = torch.cuda.is_available()
        except ImportError:
            _cuda_available = False
    
    def run(self):
        """Run memory monitoring loop."""
        import psutil
        process = psutil.Process(os.getpid())
        
        while self.running:
            try:
                # Get current memory usage
                mem_info = process.memory_info()
                mem_usage = mem_info.rss / psutil.virtual_memory().total
                
                # Get tracemalloc statistics
                current, peak = tracemalloc.get_traced_memory()
                
                # Log memory usage
                self.logger.debug(f"Memory: {mem_usage:.2%} (Current: {current/1e6:.1f}MB, Peak: {peak/1e6:.1f}MB)")
                
                # Check memory threshold
                if mem_usage > self.threshold:
                    self.logger.warning(f"Memory usage {mem_usage:.2%} exceeded threshold {self.threshold:.2%}")
                    force_gc_collection()
                
                # Check CUDA memory if available
                if _cuda_available:
                    try:
                        import torch
                        allocated = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                        if allocated > self.threshold:
                            self.logger.warning(f"CUDA memory usage {allocated:.2%} exceeded threshold")
                            force_cuda_empty_cache()
                    except Exception as e:
                        self.logger.debug(f"Failed to check CUDA memory: {str(e)}")
                
                # Sleep for check interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in memory monitor: {str(e)}")
                time.sleep(self.check_interval * 2)  # Longer sleep on error
                
    def stop(self):
        """Stop memory monitoring."""
        self.running = False
        tracemalloc.stop()


def memory_tracked(func):
    """Decorator to track memory usage before and after function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _op_counter, _last_gc_collection, _last_cuda_emptied
        
        # Increment operation counter
        _op_counter += 1
        
        # Check if we should run GC
        if _op_counter - _last_gc_collection >= _gc_collection_frequency:
            force_gc_collection()
            _last_gc_collection = _op_counter
        
        # Check if we should empty CUDA cache
        if _cuda_available and _op_counter - _last_cuda_emptied >= _cuda_empty_frequency:
            force_cuda_empty_cache()
            _last_cuda_emptied = _op_counter
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # Log error and perform emergency GC
            logger = logging.getLogger(__name__)
            logger.error(f"Error in {func.__name__}: {str(e)}")
            force_gc_collection()
            raise
    
    return wrapper


def force_gc_collection():
    """Force aggressive garbage collection."""
    logger = logging.getLogger(__name__).getChild("gc")
    start_time = time.time()
    
    # Get current memory usage
    try:
        import psutil
        process = psutil.Process(os.getpid())
        before_mem = process.memory_info().rss / 1024 / 1024  # MB
    except Exception:
        before_mem = 0
    
    # Run garbage collection
    gc.collect(0)  # Young generation
    gc.collect(1)  # Middle generation 
    gc.collect(2)  # Old generation
    
    # Check memory after collection
    try:
        import psutil
        process = psutil.Process(os.getpid())
        after_mem = process.memory_info().rss / 1024 / 1024  # MB
        logger.debug(f"GC freed {before_mem - after_mem:.1f}MB in {time.time() - start_time:.3f}s")
    except Exception:
        pass


def force_cuda_empty_cache():
    """Force CUDA cache emptying."""
    if not _cuda_available:
        return
        
    logger = logging.getLogger(__name__).getChild("cuda")
    start_time = time.time()
    
    try:
        import torch
        if torch.cuda.is_available():
            # Get before memory
            before_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            # Empty cache
            torch.cuda.empty_cache()
            
            # Synchronize to ensure completion
            torch.cuda.synchronize()
            
            # Get after memory
            after_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            
            logger.debug(f"CUDA cache emptied: {before_mem - after_mem:.1f}MB in {time.time() - start_time:.3f}s")
    except Exception as e:
        logger.debug(f"Failed to empty CUDA cache: {str(e)}")


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


def start_memory_monitor(config: Dict[str, Any]) -> None:
    """Start memory monitoring based on config."""
    global _memory_monitor, _gc_collection_frequency, _cuda_empty_frequency
    
    if not _memory_monitor:
        # Get memory monitoring settings
        system_config = config.get("system", {})
        threshold = system_config.get("gc_threshold", 0.8)
        
        # Get memory management settings if available
        memory_config = system_config.get("memory_management", {})
        _gc_collection_frequency = memory_config.get("gc_collect_frequency", 5)
        _cuda_empty_frequency = memory_config.get("cuda_empty_cache_frequency", 3)
        
        # Start memory monitor
        _memory_monitor = MemoryMonitor(threshold=threshold)
        _memory_monitor.start()


def stop_memory_monitor() -> None:
    """Stop memory monitoring."""
    global _memory_monitor
    
    if _memory_monitor:
        _memory_monitor.stop()
        _memory_monitor.join(timeout=1.0)
        _memory_monitor = None


@memory_tracked
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
        logger = logging.getLogger(__name__)
        
        # Start memory monitoring
        start_memory_monitor(config)
        
        logger.info(f"Starting SVG Prompt Analyzer with memory optimization")
        
        # Choose application based on LLM setting
        if args.use_llm or config.get("llm", {}).get("enabled", True):
            logger.info("Using LLM-enhanced mode")
            # Lazy import to reduce startup memory
            global _llm_enhanced_app
            if _llm_enhanced_app is None:
                from svg_prompt_analyzer.llm_integration.llm_enhanced_app import LLMEnhancedSVGApp
                _llm_enhanced_app = LLMEnhancedSVGApp
            
            app = _llm_enhanced_app(
                input_file=args.input_file,
                output_dir=args.output_dir,
                config_file=args.config,
                batch_size=args.batch_size,
                use_llm=True,
                use_optimization=args.optimize,
                optimization_level=args.optimization_level,
                parallel_processing=args.parallel,
                num_workers=args.workers,
                memory_efficient=args.memory_efficient or config.get("system", {}).get("memory_efficient", True)
            )
        else:
            logger.info("Using standard mode (no LLM)")
            # Lazy import to reduce startup memory
            global _svg_prompt_analyzer_app
            if _svg_prompt_analyzer_app is None:
                from svg_prompt_analyzer.deprecated.app import SVGPromptAnalyzerApp
                _svg_prompt_analyzer_app = SVGPromptAnalyzerApp
            
            app = _svg_prompt_analyzer_app(
                args.input_file, 
                args.output_dir,
                args.batch_size
            )
        
        # Run with memory tracking
        with WeakMethod(app, 'run') as run_method:
            results = run_method()
        
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
        force_gc_collection()
        force_cuda_empty_cache()
        stop_memory_monitor()


class WeakMethod:
    """Weak reference to a method to avoid reference cycles."""
    
    def __init__(self, obj, method_name):
        self.wref = weakref.ref(obj)
        self.method_name = method_name
        
    def __enter__(self):
        obj = self.wref()
        if obj is None:
            raise RuntimeError("Object has been garbage collected")
        return getattr(obj, self.method_name)
        
    def __exit__(self, *args):
        # Help GC by clearing reference
        self.wref = None


def summarize_results(results, output_dir, elapsed_time):
    """Summarize results with memory-efficient processing."""
    if not results:
        print("No results to summarize")
        return
        
    # Count success/failure without storing full results
    successful = 0
    failed = 0
    
    for result in results:
        if result.get("success", False):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessed {len(results)} prompts in {elapsed_time:.2f}s:")
    print(f"- {successful} successful")
    print(f"- {failed} failed")
    
    # Release results from memory
    results.clear()
    
    print(f"\nOutput saved to: {os.path.abspath(output_dir)}")


def parse_args():
    """Parse command line arguments with optimized defaults."""
    parser = argparse.ArgumentParser(description="Memory-Optimized SVG Prompt Analyzer")
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
    llm_group.add_argument("--optimize", "-p", action="store_true", help="Use optimization (requires LLM)")
    llm_group.add_argument("--optimization-level", "-O", type=int, default=1, choices=[1, 2, 3], 
                      help="Optimization level (1-3, higher is more thorough)")
    
    # Performance arguments
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument("--parallel", "-P", action="store_true", help="Process prompts in parallel")
    perf_group.add_argument("--workers", "-w", type=int, default=0, 
                         help="Number of parallel workers (0 = auto)")
    perf_group.add_argument("--memory-efficient", "-m", action="store_true", help="Use memory-efficient processing")
    
    return parser.parse_args()


if __name__ == "__main__":
    main()