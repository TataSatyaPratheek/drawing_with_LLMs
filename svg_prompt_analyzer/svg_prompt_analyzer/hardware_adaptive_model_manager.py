"""
Hardware-Adaptive Model Manager
=============================
This module provides hardware-specific optimizations for LLM models,
dynamically adapting to available computation resources.
"""

import os
import gc
import logging
import platform
import threading
import json
import time
from typing import Dict, Any, Optional, List, Union, Tuple

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class HardwareManager:
    """
    Hardware manager for device-specific optimizations and model loading strategies.
    Implements singleton pattern for system-wide resource management.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for resource efficiency."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(HardwareManager, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, 
                 auto_optimize: bool = True,
                 quantization_threshold_memory: float = 8.0,  # GB
                 max_cpu_threads: Optional[int] = None,
                 enable_mkldnn: bool = True,
                 enable_metal: bool = True,
                 enable_tensorrt: bool = True,
                 enable_cuda_graphs: bool = True):
        """
        Initialize the hardware manager.
        
        Args:
            auto_optimize: Whether to automatically optimize for hardware
            quantization_threshold_memory: Threshold in GB to enable quantization
            max_cpu_threads: Maximum CPU threads to use (None = auto)
            enable_mkldnn: Whether to enable MKL-DNN on CPU
            enable_metal: Whether to enable Metal on macOS
            enable_tensorrt: Whether to enable TensorRT on NVIDIA
            enable_cuda_graphs: Whether to enable CUDA Graphs for inference
        """
        # Initialize only once (singleton pattern)
        if hasattr(self, 'initialized'):
            return
            
        self.auto_optimize = auto_optimize
        self.quantization_threshold_memory = quantization_threshold_memory
        self.max_cpu_threads = max_cpu_threads
        self.enable_mkldnn = enable_mkldnn
        self.enable_metal = enable_metal
        self.enable_tensorrt = enable_tensorrt
        self.enable_cuda_graphs = enable_cuda_graphs
        
        # Hardware detection
        self.hardware_info = self._detect_hardware()
        
        # Device optimization status
        self.optimizations_applied = {
            "cpu": False,
            "cuda": False,
            "mps": False,
            "rocm": False
        }
        
        # Available optimization levels
        self.optimization_levels = {
            "balanced": {
                "description": "Balance between speed and memory usage",
                "quantization": "int8",
                "precision": "mixed",
                "threads": 0.8,  # 80% of available cores
                "batch_size": "auto"
            },
            "speed": {
                "description": "Optimize for maximum speed",
                "quantization": "none",
                "precision": "full",
                "threads": 1.0,  # 100% of available cores 
                "batch_size": "auto"
            },
            "memory": {
                "description": "Optimize for minimum memory usage",
                "quantization": "int4",
                "precision": "low",
                "threads": 0.5,  # 50% of available cores
                "batch_size": "minimum"
            },
            "extreme_memory": {
                "description": "Extreme memory optimization for constrained environments",
                "quantization": "int4",
                "precision": "lowest",
                "threads": 0.3,  # 30% of available cores
                "batch_size": 1
            }
        }
        
        # Apply optimizations if auto-optimize is enabled
        if self.auto_optimize:
            self._apply_hardware_optimizations()
        
        # Flag initialization complete
        self.initialized = True
        logger.info(f"Hardware Manager initialized for {self.get_optimal_device()}")
        
    def _detect_hardware(self) -> Dict[str, Any]:
        """
        Detect hardware capabilities of the system.
        
        Returns:
            Dictionary of hardware information
        """
        info = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count() or 1,
            "cpu_info": {},
            "memory_gb": 0,
            "gpu_available": False,
            "gpu_info": [],
            "cuda_available": False,
            "mps_available": False,
            "rocm_available": False,
            "low_memory_device": False,
            "optimization_target": "balanced"
        }
        
        # Get memory
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                info["memory_gb"] = mem.total / (1024**3)
                # Consider low memory if less than 8GB
                info["low_memory_device"] = info["memory_gb"] < 8.0
            except Exception as e:
                logger.warning(f"Error getting memory info: {e}")
        
        # Get CPU info
        try:
            if PSUTIL_AVAILABLE:
                info["cpu_info"] = {
                    "physical_cores": psutil.cpu_count(logical=False) or 1,
                    "logical_cores": psutil.cpu_count(logical=True) or 1,
                    "freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown"
                }
                
                # Check for low-power device
                if info["cpu_info"]["physical_cores"] <= 2:
                    info["low_memory_device"] = True  # Likely a constrained device
        except Exception as e:
            logger.warning(f"Error getting CPU info: {e}")
        
        # Get GPU info
        if TORCH_AVAILABLE:
            # Check for CUDA (NVIDIA)
            if torch.cuda.is_available():
                info["cuda_available"] = True
                info["gpu_available"] = True
                
                gpu_count = torch.cuda.device_count()
                info["cuda_version"] = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
                
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info = {
                        "name": props.name,
                        "memory_gb": props.total_memory / (1024**3),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multi_processor_count": props.multi_processor_count
                    }
                    info["gpu_info"].append(gpu_info)
                    
                    # Determine if this is a low-end GPU
                    if props.total_memory < 4 * (1024**3):  # Less than 4GB VRAM
                        info["low_memory_device"] = True
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available'):
                if torch.mps.is_available():
                    info["mps_available"] = True
                    info["gpu_available"] = True
                    info["gpu_info"].append({
                        "name": "Apple Silicon MPS",
                        "memory_gb": "Shared with system",
                        "compute_capability": "Unknown",
                        "multi_processor_count": "Unknown"
                    })
            
            # Check for ROCm (AMD)
            if hasattr(torch, 'xpu') and hasattr(torch.xpu, 'is_available'):
                if torch.xpu.is_available():
                    info["rocm_available"] = True
                    info["gpu_available"] = True
                    
                    # Get ROCm devices
                    xpu_count = torch.xpu.device_count()
                    
                    for i in range(xpu_count):
                        try:
                            device_name = torch.xpu.get_device_name(i)
                            info["gpu_info"].append({
                                "name": device_name,
                                "memory_gb": "Unknown",
                                "compute_capability": "Unknown",
                                "multi_processor_count": "Unknown"
                            })
                        except:
                            info["gpu_info"].append({
                                "name": f"AMD GPU {i}",
                                "memory_gb": "Unknown",
                                "compute_capability": "Unknown",
                                "multi_processor_count": "Unknown"
                            })
        
        # Determine optimal optimization target based on hardware
        if info["low_memory_device"]:
            if info["memory_gb"] < 4.0:
                info["optimization_target"] = "extreme_memory"
            else:
                info["optimization_target"] = "memory"
        elif info["gpu_available"]:
            # Check if high-end GPU
            high_end_gpu = False
            for gpu in info["gpu_info"]:
                if isinstance(gpu.get("memory_gb"), (int, float)) and gpu["memory_gb"] > 16.0:
                    high_end_gpu = True
                    break
            
            if high_end_gpu and info["memory_gb"] > 16.0:
                info["optimization_target"] = "speed"
            else:
                info["optimization_target"] = "balanced"
        else:
            # CPU only system
            if info["memory_gb"] > 16.0:
                info["optimization_target"] = "balanced"
            else:
                info["optimization_target"] = "memory"
        
        return info
        
    def get_optimal_device(self) -> str:
        """
        Determine the optimal device for computation based on hardware.
        
        Returns:
            Device type ('cuda', 'mps', 'xpu', 'cpu')
        """
        if not TORCH_AVAILABLE:
            return "cpu"
            
        # Check for CUDA (NVIDIA)
        if self.hardware_info["cuda_available"]:
            return "cuda"
            
        # Check for MPS (Apple Silicon)
        if self.hardware_info["mps_available"]:
            return "mps"
            
        # Check for ROCm (AMD)
        if self.hardware_info["rocm_available"]:
            return "xpu"
            
        # Fall back to CPU
        return "cpu"
        
    def get_optimization_level(self) -> Dict[str, Any]:
        """
        Get current optimization level configuration.
        
        Returns:
            Dictionary of optimization settings
        """
        target = self.hardware_info["optimization_target"]
        return self.optimization_levels[target]
        
    def _apply_hardware_optimizations(self) -> None:
        """Apply hardware-specific optimizations for PyTorch."""
        if not TORCH_AVAILABLE:
            return
            
        # Get optimal device
        device = self.get_optimal_device()
        
        # Apply device-specific optimizations
        if device == "cuda" and not self.optimizations_applied["cuda"]:
            self._optimize_for_cuda()
            self.optimizations_applied["cuda"] = True
        elif device == "mps" and not self.optimizations_applied["mps"]:
            self._optimize_for_mps()
            self.optimizations_applied["mps"] = True
        elif device == "xpu" and not self.optimizations_applied["rocm"]:
            self._optimize_for_rocm()
            self.optimizations_applied["rocm"] = True
        elif device == "cpu" and not self.optimizations_applied["cpu"]:
            self._optimize_for_cpu()
            self.optimizations_applied["cpu"] = True
            
    def _optimize_for_cuda(self) -> None:
        """Apply CUDA-specific optimizations."""
        try:
            # Enable cuDNN benchmarking
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark mode")
                
                # Use deterministic algorithms if needed
                # torch.backends.cudnn.deterministic = True
                
            # Enable TF32 precision on Ampere or newer GPUs
            if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
                logger.info("Enabled TF32 precision for matrix multiplications")
                
                if hasattr(torch.backends.cuda, 'allow_tf32'):
                    torch.backends.cuda.allow_tf32 = True
                    logger.info("Enabled TF32 precision for CUDA operations")
                    
            # Set device
            torch.cuda.set_device(0)  # Use first GPU by default
            
            # Enable CUDA graphs if available and requested
            if self.enable_cuda_graphs and hasattr(torch.cuda, 'graphs') and callable(getattr(torch.cuda, 'is_current_stream_capturing', None)):
                logger.info("CUDA graphs support is available")
                
            # Enable TensorRT if available and requested
            if self.enable_tensorrt:
                try:
                    import torch_tensorrt
                    logger.info("TensorRT support is available")
                except ImportError:
                    logger.debug("TensorRT not available")
                
            logger.info("Applied CUDA-specific optimizations")
        except Exception as e:
            logger.warning(f"Error optimizing for CUDA: {e}")
            
    def _optimize_for_mps(self) -> None:
        """Apply MPS (Apple Silicon) specific optimizations."""
        try:
            # Set device
            if hasattr(torch.mps, 'current_device'):
                # Not much to optimize for MPS yet
                device = torch.device("mps")
                logger.info("Using MPS device for computation")
            
            logger.info("Applied MPS-specific optimizations")
        except Exception as e:
            logger.warning(f"Error optimizing for MPS: {e}")
            
    def _optimize_for_rocm(self) -> None:
        """Apply ROCm (AMD) specific optimizations."""
        try:
            # Not much specific to ROCm yet
            logger.info("Applied ROCm-specific optimizations")
        except Exception as e:
            logger.warning(f"Error optimizing for ROCm: {e}")
            
    def _optimize_for_cpu(self) -> None:
        """Apply CPU-specific optimizations."""
        try:
            # Set number of threads
            if self.max_cpu_threads is None:
                # Auto-determine based on system
                physical_cores = self.hardware_info["cpu_info"].get("physical_cores", os.cpu_count() or 1)
                optimization_level = self.get_optimization_level()
                thread_factor = optimization_level.get("threads", 0.8)
                
                # Calculate optimal threads
                optimal_threads = max(1, int(physical_cores * thread_factor))
                
                # Reserve at least one core for system operations
                if optimal_threads > 2:
                    optimal_threads -= 1
            else:
                optimal_threads = self.max_cpu_threads
                
            # Set threads for PyTorch
            if hasattr(torch, 'set_num_threads'):
                torch.set_num_threads(optimal_threads)
                logger.info(f"Set PyTorch to use {optimal_threads} threads")
                
            # Set interop threads
            if hasattr(torch, 'set_num_interop_threads'):
                interop_threads = max(1, optimal_threads // 2)
                torch.set_num_interop_threads(interop_threads)
                logger.info(f"Set PyTorch interop threads to {interop_threads}")
                
            # Enable MKL-DNN if available and requested
            if self.enable_mkldnn and hasattr(torch.backends, 'mkldnn'):
                torch.backends.mkldnn.enabled = True
                logger.info("Enabled MKL-DNN for CPU operations")
                
            # Check for AVX/AVX2 support
            # This is a bit tricky as PyTorch doesn't expose this directly
            # In a full implementation, we'd check the CPU flags
                
            logger.info("Applied CPU-specific optimizations")
        except Exception as e:
            logger.warning(f"Error optimizing for CPU: {e}")
            
    def optimize_model_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize model configuration based on hardware.
        
        Args:
            model_config: Original model configuration
            
        Returns:
            Optimized model configuration
        """
        device = self.get_optimal_device()
        optimization_level = self.get_optimization_level()
        
        # Start with copy of original config
        optimized_config = model_config.copy() if model_config else {}
        
        # Set device
        optimized_config["device"] = device
        
        # Set quantization based on optimization level and device
        if device in ["cuda", "xpu"]:
            # GPU optimization
            if optimization_level["quantization"] == "int8":
                optimized_config["use_8bit"] = True
                optimized_config["use_4bit"] = False
            elif optimization_level["quantization"] == "int4":
                optimized_config["use_8bit"] = False
                optimized_config["use_4bit"] = True
            else:
                optimized_config["use_8bit"] = False
                optimized_config["use_4bit"] = False
                
            # Set precision
            if optimization_level["precision"] == "mixed":
                optimized_config["precision"] = "fp16"
            elif optimization_level["precision"] == "low":
                optimized_config["precision"] = "bf16"
            else:
                optimized_config["precision"] = "fp32"
        elif device == "mps":
            # MPS optimization
            optimized_config["use_8bit"] = False
            optimized_config["use_4bit"] = False
            optimized_config["precision"] = "fp16"  # MPS works well with fp16
        else:
            # CPU optimization
            if optimization_level["quantization"] == "int8":
                optimized_config["use_8bit"] = True
                optimized_config["use_4bit"] = False
            elif optimization_level["quantization"] == "int4":
                optimized_config["use_8bit"] = False
                optimized_config["use_4bit"] = True
            else:
                optimized_config["use_8bit"] = False
                optimized_config["use_4bit"] = False
        
        # Set batch size
        if optimization_level["batch_size"] == "auto":
            # Will be calculated elsewhere based on input size
            pass
        elif optimization_level["batch_size"] == "minimum":
            optimized_config["batch_size"] = 1
        else:
            optimized_config["batch_size"] = optimization_level["batch_size"]
        
        # Set gradient checkpointing based on memory constraint
        if self.hardware_info["low_memory_device"]:
            optimized_config["gradient_checkpointing"] = True
        
        # Other hardware-specific configs
        if device == "cuda":
            optimized_config["cuda_graphs"] = self.enable_cuda_graphs
            
        # Model-specific optimizations (could be expanded based on model type)
        
        logger.info(f"Optimized model config for {device}: {optimization_level['description']}")
        return optimized_config
    
    def get_device_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics for the current device.
        
        Returns:
            Dictionary of memory statistics
        """
        stats = {}
        
        if not TORCH_AVAILABLE:
            return stats
            
        device = self.get_optimal_device()
        
        if device == "cuda":
            stats["total"] = {}
            stats["allocated"] = {}
            stats["cached"] = {}
            
            for i in range(torch.cuda.device_count()):
                stats["total"][i] = torch.cuda.get_device_properties(i).total_memory
                stats["allocated"][i] = torch.cuda.memory_allocated(i)
                stats["cached"][i] = torch.cuda.memory_reserved(i)
                
        elif device == "mps" and hasattr(torch.mps, 'current_allocated_memory'):
            # Apple Silicon stats
            stats["allocated"] = torch.mps.current_allocated_memory()
            stats["total"] = "Shared with system"
            
        elif device == "xpu" and hasattr(torch.xpu, 'memory_allocated'):
            # ROCm stats
            stats["allocated"] = torch.xpu.memory_allocated()
            
        # System memory stats
        if PSUTIL_AVAILABLE:
            sys_mem = psutil.virtual_memory()
            stats["system"] = {
                "total": sys_mem.total,
                "available": sys_mem.available,
                "percent": sys_mem.percent
            }
            
        return stats
        
    def can_run_model(self, 
                     model_size_bytes: int, 
                     input_size_bytes: int,
                     output_size_bytes: int, 
                     batch_size: int = 1) -> bool:
        """
        Check if a model can run with the given parameters.
        
        Args:
            model_size_bytes: Estimated model size in bytes
            input_size_bytes: Estimated input size in bytes
            output_size_bytes: Estimated output size in bytes
            batch_size: Batch size
            
        Returns:
            Whether the model can run
        """
        if not TORCH_AVAILABLE:
            # Can't determine accurately without PyTorch
            return True
            
        device = self.get_optimal_device()
        
        # Estimate total memory required
        total_required = model_size_bytes + (input_size_bytes + output_size_bytes) * batch_size
        
        # Add overhead factor (activation memory, gradients, etc.)
        overhead_factor = 1.5
        total_required = int(total_required * overhead_factor)
        
        # Check available memory
        if device == "cuda":
            device_idx = torch.cuda.current_device()
            total_mem = torch.cuda.get_device_properties(device_idx).total_memory
            allocated_mem = torch.cuda.memory_allocated(device_idx)
            free_mem = total_mem - allocated_mem
            
            can_run = free_mem >= total_required
            memory_utilization = (allocated_mem + total_required) / total_mem
            
            logger.info(f"CUDA memory check: Required={total_required/(1024**3):.2f}GB, "
                      f"Free={free_mem/(1024**3):.2f}GB, Can run={can_run}, "
                      f"Utilization would be {memory_utilization*100:.1f}%")
                      
            return can_run
            
        elif device == "mps":
            # Can't easily check MPS memory as it's shared with system
            # Use system memory as a proxy
            if PSUTIL_AVAILABLE:
                sys_mem = psutil.virtual_memory()
                free_mem = sys_mem.available
                
                can_run = free_mem >= total_required
                memory_utilization = (sys_mem.total - free_mem + total_required) / sys_mem.total
                
                logger.info(f"MPS memory check: Required={total_required/(1024**3):.2f}GB, "
                          f"Free={free_mem/(1024**3):.2f}GB, Can run={can_run}, "
                          f"Utilization would be {memory_utilization*100:.1f}%")
                          
                return can_run
                
        elif device == "xpu":
            # ROCm memory check not yet implemented
            pass
            
        # CPU check
        if PSUTIL_AVAILABLE:
            sys_mem = psutil.virtual_memory()
            free_mem = sys_mem.available
            
            can_run = free_mem >= total_required
            memory_utilization = (sys_mem.total - free_mem + total_required) / sys_mem.total
            
            logger.info(f"CPU memory check: Required={total_required/(1024**3):.2f}GB, "
                      f"Free={free_mem/(1024**3):.2f}GB, Can run={can_run}, "
                      f"Utilization would be {memory_utilization*100:.1f}%")
                      
            return can_run
            
        # Default if we can't determine
        return True
        
    def suggest_quantization(self, model_size_bytes: int) -> str:
        """
        Suggest quantization level based on model size and available memory.
        
        Args:
            model_size_bytes: Estimated model size in bytes
            
        Returns:
            Suggested quantization ('none', 'int8', 'int4')
        """
        device = self.get_optimal_device()
        
        # Convert model size to GB
        model_size_gb = model_size_bytes / (1024**3)
        
        # Check available memory
        if device == "cuda":
            device_idx = torch.cuda.current_device()
            total_mem = torch.cuda.get_device_properties(device_idx).total_memory
            free_mem_gb = (total_mem - torch.cuda.memory_allocated(device_idx)) / (1024**3)
            
            # Determine quantization based on available memory
            if free_mem_gb < model_size_gb * 0.5:  # Less than half the model size
                return "int4"
            elif free_mem_gb < model_size_gb:
                return "int8"
            else:
                return "none"
                
        elif device in ["mps", "xpu"]:
            # For MPS and ROCm, use system memory as a proxy
            if PSUTIL_AVAILABLE:
                sys_mem = psutil.virtual_memory()
                free_mem_gb = sys_mem.available / (1024**3)
                
                if free_mem_gb < model_size_gb * 0.75:
                    return "int4"
                elif free_mem_gb < model_size_gb * 1.5:
                    return "int8"
                else:
                    return "none"
        else:
            # CPU
            if PSUTIL_AVAILABLE:
                sys_mem = psutil.virtual_memory()
                free_mem_gb = sys_mem.available / (1024**3)
                
                if free_mem_gb < model_size_gb * 1.2:  # Need more memory on CPU
                    return "int4"
                elif free_mem_gb < model_size_gb * 2.0:
                    return "int8"
                else:
                    return "none"
                    
        # Default if we can't determine
        if model_size_gb > self.quantization_threshold_memory:
            return "int8"
        else:
            return "none"
            
    def print_hardware_summary(self) -> None:
        """Print a summary of detected hardware."""
        print("\n=== Hardware Summary ===")
        print(f"Platform: {self.hardware_info['platform']} ({self.hardware_info['architecture']})")
        print(f"CPU: {self.hardware_info['processor']}")
        print(f"  Cores: {self.hardware_info['cpu_info'].get('physical_cores', 'Unknown')} physical, "
              f"{self.hardware_info['cpu_info'].get('logical_cores', 'Unknown')} logical")
        print(f"Memory: {self.hardware_info['memory_gb']:.2f} GB")
        
        if self.hardware_info["gpu_available"]:
            print("\nGPU Information:")
            for i, gpu in enumerate(self.hardware_info["gpu_info"]):
                print(f"  GPU {i}: {gpu['name']}")
                if isinstance(gpu.get("memory_gb"), (int, float)):
                    print(f"    Memory: {gpu['memory_gb']:.2f} GB")
                else:
                    print(f"    Memory: {gpu.get('memory_gb', 'Unknown')}")
                    
                print(f"    Compute Capability: {gpu.get('compute_capability', 'Unknown')}")
                
        print(f"\nOptimal Device: {self.get_optimal_device()}")
        print(f"Optimization Target: {self.hardware_info['optimization_target']}")
        print(f"Low Memory Device: {'Yes' if self.hardware_info['low_memory_device'] else 'No'}")
        print("========================\n")