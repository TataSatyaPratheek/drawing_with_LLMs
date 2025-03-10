"""
Core module for high-performance production operations.
This module implements memory, CPU, and I/O optimizations for production environments.
"""

import os
import sys
import logging
import warnings
import functools
from typing import Dict, Any, List, Optional, Union, Callable
import importlib.util
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from svg_prompt_analyzer.core.batch_processor import BatchProcessor
from svg_prompt_analyzer.core.core_module_integration import CoreManager, get_core_manager
from svg_prompt_analyzer.core.hardware_manager import HardwareManager
from svg_prompt_analyzer.core.memory_manager import MemoryManager
from svg_prompt_analyzer.core.resource_monitor import ResourceMonitor


# Configure logging with reasonable defaults for production
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings in production
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Global configuration settings with production defaults
CONFIG: Dict[str, Any] = {
    # Resource management
    "max_workers": os.cpu_count() or 4,
    "thread_pool_size": min(32, (os.cpu_count() or 4) * 2),
    "process_pool_size": os.cpu_count() or 4,
    "io_bound_thread_multiplier": 4,
    
    # Memory optimization
    "use_memory_mapping": True,
    "chunk_size": 1024 * 1024,  # 1MB chunks for streaming operations
    "cache_size_mb": 128,
    "gc_threshold": 1000,
    
    # Performance settings
    "enable_jit": True,
    "vectorized_operations": True,
    "use_numpy_if_available": True,
    "enable_profiling": False,
    
    # Distributed processing
    "distributed_mode": False,
    "coordinator_address": None,
}

# Initialize resource pools - lazily created on first use
_thread_pool = None
_process_pool = None

# Import utility for lazy loading modules to reduce startup time
def lazy_import(name: str) -> Any:
    """Lazily import a module or component when first accessed."""
    try:
        spec = importlib.util.find_spec(name)
        loader = importlib.util.LazyLoader(spec.loader)
        spec.loader = loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        return module
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to lazily import {name}: {e}")
        return None

# Memory efficiency utilities
def get_thread_pool() -> ThreadPoolExecutor:
    """Get or create the thread pool executor with optimal settings."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(
            max_workers=CONFIG["thread_pool_size"],
            thread_name_prefix="core_thread_"
        )
    return _thread_pool

def get_process_pool() -> ProcessPoolExecutor:
    """Get or create the process pool executor with optimal settings."""
    global _process_pool
    if _process_pool is None:
        _process_pool = ProcessPoolExecutor(
            max_workers=CONFIG["process_pool_size"]
        )
    return _process_pool

# Performance optimization decorators
def memoize(func: Callable) -> Callable:
    """Memoization decorator for caching expensive function results."""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a hashable key from the arguments
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
            
            # Simple cache size management
            if len(cache) > CONFIG["gc_threshold"]:
                # Remove oldest 25% of entries when threshold is reached
                remove_count = len(cache) // 4
                for _ in range(remove_count):
                    if cache:
                        cache.pop(next(iter(cache)))
                
        return cache[key]
    
    return wrapper

# Numpy-accelerated operations if available
try:
    if CONFIG["use_numpy_if_available"]:
        import numpy as np
        NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.info("NumPy not available, falling back to pure Python implementations")

# Optional JIT compilation if supported
try:
    if CONFIG["enable_jit"]:
        import numba
        jit = numba.jit
        NUMBA_AVAILABLE = True
except ImportError:
    # Fallback no-op decorator when numba is not available
    jit = lambda *args, **kwargs: lambda func: func
    NUMBA_AVAILABLE = False
    logger.info("Numba not available, JIT compilation disabled")

# Configure automatic garbage collection optimization
try:
    import gc
    gc.set_threshold(CONFIG["gc_threshold"], CONFIG["gc_threshold"] * 5, CONFIG["gc_threshold"] * 10)
except ImportError:
    logger.warning("Failed to configure garbage collection optimization")

# Context manager for performance profiling
class Profiler:
    """Simple context manager for code profiling in production."""
    def __init__(self, name: str, enabled: bool = None):
        self.name = name
        self.enabled = CONFIG["enable_profiling"] if enabled is None else enabled
        self.start_time = None
    
    def __enter__(self):
        if not self.enabled:
            return self
        
        import time
        self.start_time = time.time()
        logger.debug(f"Profiling started: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled or self.start_time is None:
            return
        
        import time
        duration = time.time() - self.start_time
        logger.debug(f"Profiling completed: {self.name} - {duration:.6f}s")

# Expose the main components for direct imports from the core module
__all__ = [
    "CONFIG",
    "get_thread_pool",
    "get_process_pool",
    "memoize",
    "jit",
    "Profiler",
    "lazy_import",
    "BatchProcessor",
    "CoreManager",
    "get_core_manager",
    "HardwareManager",
    "MemoryManager",
    "ResourceMonitor"
]

# Initialization message with platform info
logger.debug(f"Core module initialized on Python {sys.version} - "
             f"CPU count: {os.cpu_count()}, "
             f"NumPy: {NUMPY_AVAILABLE}, "
             f"JIT: {NUMBA_AVAILABLE}")

def configure(settings: Dict[str, Any]) -> None:
    """
    Update the core module configuration with custom settings.
    
    Args:
        settings: Dictionary of configuration settings to update
    """
    global CONFIG
    CONFIG.update(settings)
    logger.info(f"Core configuration updated: {', '.join(settings.keys())}")
    
    # Reset pools if their sizes were changed
    if "thread_pool_size" in settings or "process_pool_size" in settings:
        global _thread_pool, _process_pool
        
        if _thread_pool is not None:
            _thread_pool.shutdown(wait=False)
            _thread_pool = None
            
        if _process_pool is not None:
            _process_pool.shutdown(wait=False)
            _process_pool = None
            
        logger.info("Thread and process pools reset due to configuration change")

# Add configure to public API
__all__.append("configure")