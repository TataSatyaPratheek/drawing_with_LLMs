"""
Resource Monitor
===============
This module provides real-time monitoring of system resources (CPU, memory, GPU)
to help optimize performance and prevent resource exhaustion.
"""

import os
import time
import logging
import threading
import json
from typing import Dict, Any, Optional, List, Tuple, Callable, Set

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("psutil not available, resource monitoring will be limited")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.debug("PyTorch not available, GPU monitoring will be disabled")


class ResourceMonitor:
    """
    Monitors system resources and provides alerts and metrics for optimization.
    Implements singleton pattern for system-wide monitoring.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for resource efficiency."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceMonitor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, 
                 check_interval: float = 5.0,
                 memory_threshold: float = 0.8,
                 cpu_threshold: float = 0.9,
                 gpu_threshold: float = 0.85,
                 callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the resource monitor.
        
        Args:
            check_interval: Time between resource checks (seconds)
            memory_threshold: Memory usage threshold for warnings (0.0-1.0)
            cpu_threshold: CPU usage threshold for warnings (0.0-1.0)
            gpu_threshold: GPU memory threshold for warnings (0.0-1.0)
            callback: Optional callback function for resource alerts
            config: Optional configuration dictionary to override defaults
        """
        # Initialize only once (singleton pattern)
        if self._initialized:
            return
            
        # Apply configuration overrides if provided
        if config:
            monitor_config = config.get("system", {}).get("resource_monitoring", {})
            check_interval = monitor_config.get("check_interval", check_interval)
            memory_threshold = monitor_config.get("memory_threshold", memory_threshold)
            cpu_threshold = monitor_config.get("cpu_threshold", cpu_threshold)
            gpu_threshold = monitor_config.get("gpu_threshold", gpu_threshold)
            
        self.check_interval = check_interval
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.gpu_threshold = gpu_threshold
        self.callback = callback
        
        # Resource history
        self.max_history_points = 60  # Keep last 60 measurements
        self.memory_history: List[Tuple[float, float]] = []  # (timestamp, usage)
        self.cpu_history: List[Tuple[float, float]] = []
        self.gpu_memory_history: Dict[int, List[Tuple[float, float]]] = {}  # {gpu_id: [(timestamp, usage)]}
        
        # Monitoring thread
        self.monitor_thread = None
        self.monitoring = False
        
        # Resource statistics
        self.peak_memory_usage = 0.0
        self.peak_cpu_usage = 0.0
        self.peak_gpu_memory: Dict[int, float] = {}
        
        # Alert counts
        self.memory_alerts = 0
        self.cpu_alerts = 0
        self.gpu_alerts = 0
        
        # Flag initialization complete
        self._initialized = True
        
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ResourceMonitorThread"
        )
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.check_interval * 2)
            
        logger.info("Resource monitoring stopped")
        
    def _monitoring_loop(self) -> None:
        """Monitor resources in a background thread."""
        while self.monitoring:
            try:
                # Check resources
                stats = self.get_resource_stats()
                current_time = time.time()
                
                # Update history
                if PSUTIL_AVAILABLE:
                    # Memory
                    memory_usage = stats["memory"]["percent"] / 100.0
                    self.memory_history.append((current_time, memory_usage))
                    self.memory_history = self.memory_history[-self.max_history_points:]
                    
                    # Update peak memory
                    self.peak_memory_usage = max(self.peak_memory_usage, memory_usage)
                    
                    # CPU
                    cpu_usage = stats["cpu"]["percent"] / 100.0
                    self.cpu_history.append((current_time, cpu_usage))
                    self.cpu_history = self.cpu_history[-self.max_history_points:]
                    
                    # Update peak CPU
                    self.peak_cpu_usage = max(self.peak_cpu_usage, cpu_usage)
                    
                    # Check for alerts
                    if memory_usage > self.memory_threshold:
                        self.memory_alerts += 1
                        logger.warning(f"Memory usage alert: {memory_usage:.1%} (threshold: {self.memory_threshold:.1%})")
                        self._handle_memory_alert(memory_usage)
                        
                    if cpu_usage > self.cpu_threshold:
                        self.cpu_alerts += 1
                        logger.warning(f"CPU usage alert: {cpu_usage:.1%} (threshold: {self.cpu_threshold:.1%})")
                
                # GPU monitoring if available
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        gpu_memory = stats["gpu"].get(i, {}).get("memory_percent", 0) / 100.0
                        
                        # Initialize history for GPU if not exists
                        if i not in self.gpu_memory_history:
                            self.gpu_memory_history[i] = []
                            
                        # Update history
                        self.gpu_memory_history[i].append((current_time, gpu_memory))
                        self.gpu_memory_history[i] = self.gpu_memory_history[i][-self.max_history_points:]
                        
                        # Update peak GPU memory
                        self.peak_gpu_memory[i] = max(self.peak_gpu_memory.get(i, 0), gpu_memory)
                        
                        # Check for alerts
                        if gpu_memory > self.gpu_threshold:
                            self.gpu_alerts += 1
                            logger.warning(f"GPU {i} memory alert: {gpu_memory:.1%} (threshold: {self.gpu_threshold:.1%})")
                            self._handle_gpu_alert(i, gpu_memory)
                
                # Invoke callback if provided
                if self.callback:
                    self.callback(stats)
                    
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
                time.sleep(self.check_interval)  # Continue monitoring after error
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """
        Get current resource statistics.
        
        Returns:
            Dictionary of resource statistics
        """
        stats = {
            "timestamp": time.time(),
            "memory": {},
            "cpu": {},
            "gpu": {},
            "process": {}
        }
        
        try:
            # System memory
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                stats["memory"] = {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "free": memory.available,  # Using available is more accurate than free
                    "percent": memory.percent,
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3)
                }
                
                # CPU
                stats["cpu"] = {
                    "percent": psutil.cpu_percent(interval=None),
                    "count": psutil.cpu_count(logical=True),
                    "physical_count": psutil.cpu_count(logical=False)
                }
                
                # Current process
                process = psutil.Process(os.getpid())
                proc_info = process.memory_info()
                
                stats["process"] = {
                    "memory_rss": proc_info.rss,
                    "memory_vms": proc_info.vms,
                    "memory_rss_gb": proc_info.rss / (1024**3),
                    "cpu_percent": process.cpu_percent(interval=None),
                    "threads": process.num_threads(),
                    "uptime": time.time() - process.create_time()
                }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            
        try:
            # GPU stats if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_stats = {
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated": torch.cuda.memory_allocated(i),
                        "memory_reserved": torch.cuda.memory_reserved(i),
                        "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                        "memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024**3)
                    }
                    
                    # Get total memory
                    try:
                        total_memory = torch.cuda.get_device_properties(i).total_memory
                        gpu_stats["memory_total"] = total_memory
                        gpu_stats["memory_total_gb"] = total_memory / (1024**3)
                        
                        # Calculate percentage
                        if total_memory > 0:
                            gpu_stats["memory_percent"] = (torch.cuda.memory_allocated(i) / total_memory) * 100
                    except:
                        # Some devices might not report total memory
                        pass
                        
                    stats["gpu"][i] = gpu_stats
        except Exception as e:
            logger.error(f"Error getting GPU stats: {str(e)}")
            
        return stats
        
    def _handle_memory_alert(self, memory_usage: float) -> None:
        """
        Handle memory usage alert.
        
        Args:
            memory_usage: Current memory usage (0.0-1.0)
        """
        import gc
        
        logger.warning("Memory usage threshold exceeded, triggering garbage collection")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _handle_gpu_alert(self, gpu_id: int, memory_usage: float) -> None:
        """
        Handle GPU memory alert.
        
        Args:
            gpu_id: GPU identifier
            memory_usage: Current memory usage (0.0-1.0)
        """
        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.warning(f"GPU {gpu_id} memory usage threshold exceeded, clearing CUDA cache")
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Synchronize to ensure completion
            torch.cuda.synchronize()
            
    def get_resource_summary(self) -> Dict[str, Any]:
        """
        Get summary of resource usage.
        
        Returns:
            Dictionary with resource usage summary
        """
        summary = {
            "memory": {
                "current": self.memory_history[-1][1] if self.memory_history else 0,
                "peak": self.peak_memory_usage,
                "alerts": self.memory_alerts
            },
            "cpu": {
                "current": self.cpu_history[-1][1] if self.cpu_history else 0,
                "peak": self.peak_cpu_usage,
                "alerts": self.cpu_alerts
            },
            "gpu": {
                "devices": len(self.gpu_memory_history),
                "alerts": self.gpu_alerts
            }
        }
        
        # Add GPU details if available
        if self.gpu_memory_history:
            summary["gpu"]["details"] = {}
            for gpu_id, history in self.gpu_memory_history.items():
                summary["gpu"]["details"][gpu_id] = {
                    "current": history[-1][1] if history else 0,
                    "peak": self.peak_gpu_memory.get(gpu_id, 0)
                }
                
        return summary
        
    def get_recommendations(self) -> List[str]:
        """
        Get performance optimization recommendations based on resource usage.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Memory recommendations
        if self.peak_memory_usage > 0.9:
            recommendations.append("Consider enabling memory-efficient mode for lower memory usage")
            recommendations.append("Try reducing batch sizes or enabling gradient checkpointing")
            
        elif self.peak_memory_usage > 0.7:
            recommendations.append("Memory usage is high, consider increasing GC frequency")
            
        # CPU recommendations
        if self.peak_cpu_usage > 0.9:
            recommendations.append("CPU usage is very high, consider reducing concurrency")
            
        elif self.peak_cpu_usage > 0.7:
            recommendations.append("CPU is under significant load, consider optimizing CPU-intensive operations")
            
        # GPU recommendations
        high_gpu_usage = False
        for gpu_id, peak in self.peak_gpu_memory.items():
            if peak > 0.9:
                high_gpu_usage = True
                recommendations.append(f"GPU {gpu_id} memory usage is very high, consider model quantization")
                
            elif peak > 0.7:
                recommendations.append(f"GPU {gpu_id} memory usage is high, consider optimizing tensor operations")
                
        # General recommendations
        if high_gpu_usage or self.peak_memory_usage > 0.8:
            recommendations.append("Consider model pruning to reduce memory requirements")
            
        if self.memory_alerts > 10 or self.gpu_alerts > 10:
            recommendations.append("Frequent memory alerts indicate potential memory leaks or inadequate resources")
            
        return recommendations
        
    def report_to_file(self, filename: str) -> None:
        """
        Export resource report to file.
        
        Args:
            filename: Output filename
        """
        # Get current stats and summary
        stats = self.get_resource_stats()
        summary = self.get_resource_summary()
        recommendations = self.get_recommendations()
        
        # Create report
        report = {
            "timestamp": time.time(),
            "current_stats": stats,
            "summary": summary,
            "recommendations": recommendations,
            "history": {
                "memory": self.memory_history,
                "cpu": self.cpu_history,
                "gpu": self.gpu_memory_history
            }
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Resource report saved to {filename}")