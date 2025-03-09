"""
Enhanced Memory Manager
====================
This module provides advanced memory management capabilities for the SVG Prompt Analyzer.
It implements adaptive memory tracking, intelligent garbage collection, and hardware-aware
resource allocation.
"""

import os
import gc
import sys
import time
import logging
import threading
import weakref
import psutil
from typing import Dict, Any, Optional, List, Union, Callable, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False


class MemoryManager:
    """
    Advanced memory manager with adaptive resource allocation and garbage collection.
    Implements singleton pattern for system-wide memory management.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for resource efficiency."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MemoryManager, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, 
                 gc_threshold: float = 0.75,
                 gc_collect_frequency: int = 5,
                 cuda_empty_frequency: int = 3,
                 memory_warning_threshold: float = 0.9,
                 enable_tracemalloc: bool = True,
                 check_interval: int = 5,
                 adaptive_batch_sizing: bool = True,
                 log_level: str = "INFO"):
        """
        Initialize the memory manager.
        
        Args:
            gc_threshold: Memory threshold (0.0-1.0) to trigger GC
            gc_collect_frequency: Operations between GC collections
            cuda_empty_frequency: Operations between CUDA cache clearing
            memory_warning_threshold: Threshold for memory warnings
            enable_tracemalloc: Whether to enable tracemalloc for detailed tracking
            check_interval: Seconds between memory checks
            adaptive_batch_sizing: Whether to adapt batch sizes to available memory
            log_level: Logging level for memory manager
        """
        # Initialize only once (singleton pattern)
        if hasattr(self, 'initialized'):
            return
            
        self.gc_threshold = gc_threshold
        self.gc_collect_frequency = gc_collect_frequency
        self.cuda_empty_frequency = cuda_empty_frequency
        self.memory_warning_threshold = memory_warning_threshold
        self.check_interval = check_interval
        self.adaptive_batch_sizing = adaptive_batch_sizing
        
        # Operation counter
        self.op_counter = 0
        self.last_gc_collection = 0
        self.last_cuda_emptied = 0
        
        # Hardware detection
        self.hardware_info = self._detect_hardware()
        
        # Tracked objects (using weak references to avoid memory leaks)
        self.tracked_objects = weakref.WeakValueDictionary()
        
        # Memory tracking
        self.enable_tracemalloc = enable_tracemalloc and TRACEMALLOC_AVAILABLE
        if self.enable_tracemalloc:
            tracemalloc.start()
            
        # Memory monitoring thread
        self.monitor_thread = None
        self.monitor_running = False
        
        # Set up logging
        self.logger = logging.getLogger(__name__ + ".memory")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(getattr(logging, log_level))
        
        # Start memory monitoring
        self.start_monitoring()
        
        # Print hardware information
        self._log_hardware_info()
        
        # Flag initialization complete
        self.initialized = True
        self.logger.info("Memory Manager initialized")
        
    def _detect_hardware(self) -> Dict[str, Any]:
        """
        Detect hardware capabilities of the system.
        
        Returns:
            Dictionary of hardware information
        """
        info = {
            "platform": sys.platform,
            "processor_count": os.cpu_count() or 1,
            "total_ram": psutil.virtual_memory().total,
            "total_ram_gb": psutil.virtual_memory().total / (1024**3),
            "has_cuda": False,
            "has_mps": False,
            "has_rocm": False,
            "has_gpu": False,
            "gpu_info": []
        }
        
        # Check for PyTorch and CUDA
        if TORCH_AVAILABLE:
            import torch
            
            # Check for CUDA (NVIDIA)
            if torch.cuda.is_available():
                info["has_cuda"] = True
                info["has_gpu"] = True
                info["cuda_version"] = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
                
                # Get GPU information
                gpu_count = torch.cuda.device_count()
                info["gpu_count"] = gpu_count
                
                for i in range(gpu_count):
                    gpu_props = torch.cuda.get_device_properties(i)
                    info["gpu_info"].append({
                        "name": gpu_props.name,
                        "total_memory": gpu_props.total_memory,
                        "total_memory_gb": gpu_props.total_memory / (1024**3),
                        "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
                    })
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available'):
                if torch.mps.is_available():
                    info["has_mps"] = True
                    info["has_gpu"] = True
                    info["gpu_info"].append({
                        "name": "Apple Silicon",
                        "total_memory": "Shared with system"
                    })
            
            # Check for ROCm (AMD)
            if hasattr(torch, 'xpu') and hasattr(torch.xpu, 'is_available'):
                if torch.xpu.is_available():
                    info["has_rocm"] = True
                    info["has_gpu"] = True
                    
                    # Get ROCm devices
                    xpu_count = torch.xpu.device_count()
                    info["gpu_count"] = xpu_count
                    
                    for i in range(xpu_count):
                        try:
                            device_name = torch.xpu.get_device_name(i)
                            info["gpu_info"].append({
                                "name": device_name,
                                "total_memory": "Unknown"
                            })
                        except:
                            info["gpu_info"].append({
                                "name": f"AMD GPU {i}",
                                "total_memory": "Unknown"
                            })
        
        return info
        
    def _log_hardware_info(self) -> None:
        """Log detected hardware information."""
        self.logger.info(f"System: {self.hardware_info['platform']}, "
                        f"CPUs: {self.hardware_info['processor_count']}, "
                        f"RAM: {self.hardware_info['total_ram_gb']:.2f} GB")
                        
        if self.hardware_info["has_gpu"]:
            gpu_info = []
            for i, gpu in enumerate(self.hardware_info["gpu_info"]):
                if "total_memory_gb" in gpu:
                    gpu_info.append(f"{gpu['name']} ({gpu['total_memory_gb']:.2f} GB)")
                else:
                    gpu_info.append(f"{gpu['name']}")
                    
            self.logger.info(f"GPUs: {', '.join(gpu_info)}")
            
            if self.hardware_info["has_cuda"]:
                self.logger.info(f"CUDA version: {self.hardware_info['cuda_version']}")
            elif self.hardware_info["has_mps"]:
                self.logger.info("Apple Silicon MPS available")
            elif self.hardware_info["has_rocm"]:
                self.logger.info("AMD ROCm available")
        else:
            self.logger.info("No GPU acceleration available")
            
    def start_monitoring(self) -> None:
        """Start background memory monitoring thread."""
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            return
            
        self.monitor_running = True
        self.monitor_thread = threading.Thread(
            target=self._memory_monitor_loop,
            daemon=True,
            name="MemoryMonitorThread"
        )
        self.monitor_thread.start()
        self.logger.debug("Memory monitoring thread started")
        
    def stop_monitoring(self) -> None:
        """Stop background memory monitoring thread."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            return
            
        self.monitor_running = False
        self.monitor_thread.join(timeout=self.check_interval * 2)
        
        if self.monitor_thread.is_alive():
            self.logger.warning("Memory monitoring thread failed to terminate gracefully")
        else:
            self.logger.debug("Memory monitoring thread stopped")
            
        self.monitor_thread = None
        
    def _memory_monitor_loop(self) -> None:
        """Memory monitoring thread main loop."""
        process = psutil.Process(os.getpid())
        
        while self.monitor_running:
            try:
                # Get system memory info
                sys_mem = psutil.virtual_memory()
                sys_percent = sys_mem.percent / 100
                
                # Get process memory info
                proc_mem = process.memory_info()
                proc_percent = proc_mem.rss / sys_mem.total
                
                # Check if we need to force garbage collection
                if proc_percent > self.gc_threshold or sys_percent > self.gc_threshold:
                    self.logger.info(f"Memory threshold exceeded: Process {proc_percent:.1%}, System {sys_percent:.1%}")
                    self.force_garbage_collection()
                
                # Log memory usage at debug level
                self.logger.debug(f"Memory usage: Process {proc_percent:.1%} ({proc_mem.rss / 1024**2:.1f} MB), "
                               f"System {sys_percent:.1%} ({sys_mem.used / 1024**3:.1f} GB / {sys_mem.total / 1024**3:.1f} GB)")
                
                # Check GPU memory if available
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i)
                        total = torch.cuda.get_device_properties(i).total_memory if i < len(self.hardware_info["gpu_info"]) else 0
                        if total > 0:
                            gpu_percent = allocated / total
                            self.logger.debug(f"GPU {i} memory: {gpu_percent:.1%} ({allocated / 1024**3:.1f} GB)")
                            
                            if gpu_percent > self.gc_threshold:
                                self.logger.info(f"GPU {i} memory threshold exceeded: {gpu_percent:.1%}")
                                self.empty_cuda_cache()
                
                # Get tracemalloc statistics if enabled
                if self.enable_tracemalloc:
                    current, peak = tracemalloc.get_traced_memory()
                    self.logger.debug(f"Tracemalloc: Current {current / 1024**2:.1f} MB, Peak {peak / 1024**2:.1f} MB")
                    
                    # Log top allocations periodically (every 5 iterations)
                    if self.op_counter % 5 == 0:
                        top_stats = tracemalloc.take_snapshot().statistics('lineno')
                        for stat in top_stats[:3]:  # Show top 3 allocations
                            self.logger.debug(f"Memory block: {stat.size / 1024:.1f} KB from {stat.traceback.format()[-1]}")
                
                # Sleep for check interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in memory monitor: {str(e)}")
                time.sleep(self.check_interval * 2)  # Sleep longer after error
                
    def operation_checkpoint(self) -> None:
        """
        Check memory after an operation and trigger GC if needed.
        Call this periodically in long-running operations.
        """
        self.op_counter += 1
        
        # Check if we should run GC
        if self.op_counter - self.last_gc_collection >= self.gc_collect_frequency:
            self.force_garbage_collection()
            self.last_gc_collection = self.op_counter
        
        # Check if we should empty CUDA cache
        if (TORCH_AVAILABLE and torch.cuda.is_available() and 
            self.op_counter - self.last_cuda_emptied >= self.cuda_empty_frequency):
            self.empty_cuda_cache()
            self.last_cuda_emptied = self.op_counter
            
    def force_garbage_collection(self) -> Dict[str, Any]:
        """
        Force aggressive garbage collection and return memory stats.
        
        Returns:
            Dictionary with memory statistics
        """
        start_time = time.time()
        self.logger.debug("Forcing garbage collection")
        
        # Get memory before collection
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        # Run GC collection for all generations
        gc.collect(0)  # Young generation
        gc.collect(1)  # Middle generation
        gc.collect(2)  # Old generation (most thorough)
        
        # Get memory after collection
        mem_after = process.memory_info().rss
        mem_freed = mem_before - mem_after
        
        # Get collection stats
        collection_time = time.time() - start_time
        stats = {
            "mem_before": mem_before,
            "mem_after": mem_after,
            "mem_freed": mem_freed,
            "mem_freed_mb": mem_freed / (1024 * 1024),
            "collection_time": collection_time
        }
        
        if mem_freed > 0:
            self.logger.info(f"GC freed {stats['mem_freed_mb']:.2f} MB in {collection_time:.3f}s")
        else:
            self.logger.debug(f"GC completed in {collection_time:.3f}s (no memory freed)")
            
        return stats
        
    def empty_cuda_cache(self) -> Dict[str, Any]:
        """
        Empty CUDA cache if available and return stats.
        
        Returns:
            Dictionary with CUDA memory statistics
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"status": "CUDA not available"}
            
        start_time = time.time()
        self.logger.debug("Emptying CUDA cache")
        
        # Get memory before clearing
        mem_before = {i: torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())}
        
        # Empty cache
        torch.cuda.empty_cache()
        
        # Synchronize to ensure completion
        torch.cuda.synchronize()
        
        # Get memory after clearing
        mem_after = {i: torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())}
        
        # Calculate freed memory
        mem_freed = {i: mem_before[i] - mem_after[i] for i in range(torch.cuda.device_count())}
        total_freed = sum(mem_freed.values())
        
        # Get stats
        clear_time = time.time() - start_time
        stats = {
            "mem_before": mem_before,
            "mem_after": mem_after,
            "mem_freed": mem_freed,
            "total_freed_mb": total_freed / (1024 * 1024),
            "clear_time": clear_time
        }
        
        if total_freed > 0:
            self.logger.info(f"CUDA cache cleared: {stats['total_freed_mb']:.2f} MB in {clear_time:.3f}s")
        else:
            self.logger.debug(f"CUDA cache cleared in {clear_time:.3f}s (no memory freed)")
            
        return stats
        
    def track_object(self, obj: Any, name: str = None) -> str:
        """
        Track an object for memory management.
        
        Args:
            obj: Object to track
            name: Optional name for the object
            
        Returns:
            Tracking ID
        """
        obj_id = id(obj)
        name = name or f"object_{obj_id}"
        
        self.tracked_objects[obj_id] = obj
        self.logger.debug(f"Tracking object {name} (id: {obj_id})")
        
        return str(obj_id)
        
    def untrack_object(self, obj_id: Union[str, int]) -> bool:
        """
        Stop tracking an object.
        
        Args:
            obj_id: ID of object to untrack
            
        Returns:
            Whether the object was successfully untracked
        """
        obj_id = int(obj_id) if isinstance(obj_id, str) else obj_id
        
        if obj_id in self.tracked_objects:
            del self.tracked_objects[obj_id]
            self.logger.debug(f"Untracked object (id: {obj_id})")
            return True
            
        return False
        
    def calculate_optimal_batch_size(self, 
                                   item_size_estimate: int, 
                                   model_size_estimate: int = 0,
                                   target_device: str = "auto") -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            item_size_estimate: Estimated memory per item in bytes
            model_size_estimate: Estimated model size in bytes
            target_device: Target device ('cpu', 'cuda', 'auto')
            
        Returns:
            Optimal batch size
        """
        if not self.adaptive_batch_sizing:
            return 8  # Default batch size
            
        # Determine target device
        if target_device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                target_device = "cuda"
            else:
                target_device = "cpu"
                
        # Calculate available memory
        available_memory = 0
        if target_device == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
            # GPU memory calculation
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            free_memory = total_memory - allocated_memory
            
            # Reserve some memory to prevent OOM errors (25%)
            available_memory = free_memory * 0.75
        else:
            # CPU memory calculation
            system_mem = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            process_mem = process.memory_info().rss
            
            # Calculate available memory (50% of available system memory)
            available_memory = (system_mem.available * 0.5)
            
        # Calculate batch size
        # We need memory for: model + batch_size * item_size
        memory_for_batch = available_memory - model_size_estimate
        
        # Ensure positive memory for batch
        if memory_for_batch <= 0:
            self.logger.warning("Insufficient memory for processing even a single item")
            return 1
            
        # Calculate batch size
        batch_size = int(memory_for_batch / item_size_estimate)
        
        # Ensure reasonable range
        batch_size = max(1, min(64, batch_size))
        
        self.logger.info(f"Calculated optimal batch size: {batch_size} "
                       f"(Available: {available_memory / 1024**3:.2f} GB, "
                       f"Item: {item_size_estimate / 1024**2:.2f} MB, "
                       f"Model: {model_size_estimate / 1024**3:.2f} GB)")
                       
        return batch_size
        
    def memory_efficient_function(self, func: Callable) -> Callable:
        """
        Decorator for memory-efficient functions.
        Automatically performs GC before and after function execution.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        def wrapper(*args, **kwargs):
            # Increment operation counter
            self.operation_checkpoint()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                self.logger.error(f"Error in {func.__name__}: {str(e)}")
                # Force GC on error
                self.force_garbage_collection()
                raise
            finally:
                # Clean up any temporary objects
                self.operation_checkpoint()
                
        return wrapper
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        stats = {
            "system": {},
            "process": {},
            "cuda": {},
            "tracked_objects": len(self.tracked_objects),
            "gc_stats": {}
        }
        
        # System memory
        sys_mem = psutil.virtual_memory()
        stats["system"] = {
            "total": sys_mem.total,
            "available": sys_mem.available,
            "used": sys_mem.used,
            "percent": sys_mem.percent
        }
        
        # Process memory
        process = psutil.Process(os.getpid())
        proc_mem = process.memory_info()
        stats["process"] = {
            "rss": proc_mem.rss,
            "vms": proc_mem.vms,
            "percent": proc_mem.rss / sys_mem.total * 100
        }
        
        # CUDA memory
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                stats["cuda"][i] = {
                    "allocated": torch.cuda.memory_allocated(i),
                    "cached": torch.cuda.memory_reserved(i),
                    "device_name": torch.cuda.get_device_name(i)
                }
                
        # GC stats
        stats["gc_stats"] = {
            "garbage": len(gc.garbage),
            "tracked_objects": len(self.tracked_objects)
        }
        
        # Tracemalloc stats
        if self.enable_tracemalloc:
            current, peak = tracemalloc.get_traced_memory()
            stats["tracemalloc"] = {
                "current": current,
                "peak": peak
            }
            
        return stats
        
    def shutdown(self) -> None:
        """Clean up resources on shutdown."""
        self.logger.info("Shutting down memory manager")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Clear caches
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.empty_cuda_cache()
            
        # Clear tracked objects
        self.tracked_objects.clear()
        
        # Final GC
        self.force_garbage_collection()
        
        # Stop tracemalloc
        if self.enable_tracemalloc and TRACEMALLOC_AVAILABLE:
            tracemalloc.stop()