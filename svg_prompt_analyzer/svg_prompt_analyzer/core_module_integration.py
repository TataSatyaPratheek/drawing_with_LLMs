"""
Core Module Integration
====================
This module integrates the core performance optimization components 
(memory management, hardware adaptation, batch processing, resource monitoring)
and provides a unified interface for the rest of the application.
"""

import os
import time
import logging
import json
from typing import Dict, Any, Optional, List, Tuple, Callable, Union

# Import core components
from svg_prompt_analyzer.core.memory_manager import MemoryManager
from svg_prompt_analyzer.core.hardware_manager import HardwareManager
from svg_prompt_analyzer.core.batch_processor import BatchProcessor
from svg_prompt_analyzer.core.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


class CoreManager:
    """
    Central manager for core performance optimizations.
    Provides unified access to memory management, hardware adaptation,
    batch processing, and resource monitoring.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the core manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.memory_manager = MemoryManager(config=self.config)
        self.hardware_manager = HardwareManager(config=self.config)
        self.resource_monitor = ResourceMonitor(config=self.config)
        
        # Track batch processors
        self.batch_processors: Dict[str, BatchProcessor] = {}
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        logger.info("Core manager initialized")
        
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
                "memory_efficient": True,
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
                    "optimal_batch_size": 8,
                    "max_batch_size": 16,
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
                "quantization": "4bit",
                "device": "auto",
                "cache_dir": ".cache/models",
                "use_cache": True,
                "max_batch_size": 8,
                "gradient_checkpointing": True,
                "performance": {
                    "prefetch_model_weights": True,
                    "num_workers": 4,
                    "context_length": 4096,
                    "kv_cache_enabled": True,
                    "half_precision_enabled": True
                }
            }
        }
        
        # Try to load from specified path
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return loaded_config
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {str(e)}")
                
        # Try default locations
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
                    logger.info(f"Loaded configuration from {location}")
                    return loaded_config
                except Exception:
                    continue
        
        logger.warning("No configuration file found, using defaults")
        return default_config
        
    def create_batch_processor(self, 
                              name: str,
                              process_func: Callable,
                              batch_config: Optional[Dict[str, Any]] = None) -> BatchProcessor:
        """
        Create a new batch processor with optimal configuration.
        
        Args:
            name: Name for the batch processor
            process_func: Function to process batches
            batch_config: Optional specific configuration
            
        Returns:
            Configured BatchProcessor instance
        """
        # Merge specific config with defaults
        config = self.config.copy()
        if batch_config:
            # Update only batch processing section
            if "system" in config and "batch_processing" in config["system"]:
                config["system"]["batch_processing"].update(batch_config)
                
        # Create batch processor
        processor = BatchProcessor(
            process_func=process_func,
            memory_manager=self.memory_manager,
            config=config
        )
        
        # Store for tracking
        self.batch_processors[name] = processor
        
        # Start processor
        processor.start()
        
        logger.info(f"Created batch processor '{name}' with optimal configuration")
        
        return processor
        
    def stop_batch_processors(self, wait_complete: bool = True) -> None:
        """
        Stop all batch processors.
        
        Args:
            wait_complete: Whether to wait for all items to finish processing
        """
        for name, processor in self.batch_processors.items():
            logger.info(f"Stopping batch processor '{name}'")
            processor.stop(wait_complete=wait_complete)
            
    def optimize_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize model configuration based on hardware.
        
        Args:
            model_config: Original model configuration
            
        Returns:
            Optimized model configuration
        """
        return self.hardware_manager.optimize_model_config(model_config)
        
    def memory_track(self, obj: Any, name: Optional[str] = None) -> str:
        """
        Track an object for memory management.
        
        Args:
            obj: Object to track
            name: Optional name for the object
            
        Returns:
            Tracking ID
        """
        return self.memory_manager.track_object(obj, name)
        
    def force_gc(self) -> Dict[str, Any]:
        """
        Force garbage collection.
        
        Returns:
            Dictionary with memory statistics
        """
        return self.memory_manager.force_garbage_collection()
        
    def memory_efficient(self, func: Callable) -> Callable:
        """
        Decorator for memory-efficient functions.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        return self.memory_manager.memory_efficient_function(func)
        
    def get_resource_stats(self) -> Dict[str, Any]:
        """
        Get current resource statistics.
        
        Returns:
            Dictionary of resource statistics
        """
        return self.resource_monitor.get_resource_stats()
        
    def get_resource_recommendations(self) -> List[str]:
        """
        Get performance optimization recommendations.
        
        Returns:
            List of recommendation strings
        """
        return self.resource_monitor.get_recommendations()
        
    def get_optimal_device(self) -> str:
        """
        Get optimal device for computation.
        
        Returns:
            Device name ('cuda', 'mps', 'cpu', etc.)
        """
        return self.hardware_manager.get_optimal_device()
        
    def calculate_optimal_batch_size(self, 
                                   item_size_estimate: int, 
                                   model_size_estimate: int = 0) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            item_size_estimate: Estimated memory per item in bytes
            model_size_estimate: Estimated model size in bytes
            
        Returns:
            Optimal batch size
        """
        return self.memory_manager.calculate_optimal_batch_size(
            item_size_estimate=item_size_estimate,
            model_size_estimate=model_size_estimate,
            target_device=self.get_optimal_device()
        )
        
    def shutdown(self) -> None:
        """Clean up resources on shutdown."""
        logger.info("Shutting down core manager")
        
        # Stop batch processors
        self.stop_batch_processors(wait_complete=True)
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        # Clean up memory manager
        self.memory_manager.shutdown()
        
        logger.info("Core manager shutdown complete")


# Singleton instance for easy access throughout the application
_core_manager = None

def get_core_manager(config_path: Optional[str] = None) -> CoreManager:
    """
    Get the global CoreManager instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        CoreManager instance
    """
    global _core_manager
    
    if _core_manager is None:
        _core_manager = CoreManager(config_path=config_path)
        
    return _core_manager