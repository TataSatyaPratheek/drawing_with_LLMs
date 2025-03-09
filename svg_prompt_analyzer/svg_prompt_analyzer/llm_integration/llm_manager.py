"""
LLM Manager Module - Optimized
================
This module provides functionality for managing and interacting with LLM models.
It handles model loading, caching, and inference with optimizations for performance
and memory efficiency across different hardware architectures.
"""

import os
import gc
import logging
import json
import threading
import time
import platform
import multiprocessing
import queue
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Global variables for device detection
DEVICE_INFO = {
    "platform": platform.system(),
    "processor": platform.processor(),
    "is_apple_silicon": False,
    "has_cuda": False,
    "has_mps": False,
    "has_rocm": False,
    "cpu_cores": multiprocessing.cpu_count(),
    "gpu_info": None
}

# Detection for Apple Silicon
if DEVICE_INFO["platform"] == "Darwin" and "arm" in platform.processor().lower():
    DEVICE_INFO["is_apple_silicon"] = True

# Delayed imports - will be loaded on demand
class LazyImporter:
    """Handles lazy importing of optional dependencies."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LazyImporter, cls).__new__(cls)
                cls._instance.imported = {}
            return cls._instance
    
    def import_torch(self):
        """Import PyTorch modules on demand."""
        if "torch" not in self.imported:
            try:
                import torch
                import torch.nn.functional as F
                self.imported["torch"] = torch
                self.imported["F"] = F
                
                # Check for CUDA
                if torch.cuda.is_available():
                    DEVICE_INFO["has_cuda"] = True
                    DEVICE_INFO["gpu_info"] = {
                        "count": torch.cuda.device_count(),
                        "name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                        "memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.device_count() > 0 else None
                    }
                
                # Check for MPS (Apple Silicon)
                if hasattr(torch, 'mps') and torch.mps.is_available():
                    DEVICE_INFO["has_mps"] = True
                
                # Check for ROCm (AMD)
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    DEVICE_INFO["has_rocm"] = True
                
                logger.debug(f"PyTorch {torch.__version__} loaded successfully")
                return True
            except ImportError:
                logger.warning("PyTorch not installed. Some functionality will be limited.")
                self.imported["torch"] = None
                return False
        return self.imported["torch"] is not None
    
    def import_transformers(self):
        """Import transformers modules on demand."""
        if "transformers" not in self.imported:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
                self.imported["AutoModelForCausalLM"] = AutoModelForCausalLM
                self.imported["AutoTokenizer"] = AutoTokenizer
                self.imported["StoppingCriteria"] = StoppingCriteria
                self.imported["StoppingCriteriaList"] = StoppingCriteriaList
                
                logger.debug("Transformers library loaded successfully")
                return True
            except ImportError:
                logger.warning("Transformers not installed. LLM functionality will be unavailable.")
                self.imported["transformers"] = None
                return False
        return "AutoModelForCausalLM" in self.imported
    
    def import_bitsandbytes(self):
        """Import bitsandbytes for quantization on demand."""
        if "bitsandbytes" not in self.imported:
            try:
                import bitsandbytes as bnb
                self.imported["bitsandbytes"] = bnb
                
                logger.debug("BitsAndBytes library loaded successfully")
                return True
            except ImportError:
                logger.warning("BitsAndBytes not installed. Quantization will be limited.")
                self.imported["bitsandbytes"] = None
                return False
        return self.imported["bitsandbytes"] is not None
    
    def get_module(self, name):
        """Get an imported module by name."""
        return self.imported.get(name)


# Define default model paths
DEFAULT_MODELS = {
    "prompt_analyzer": "mistralai/Mistral-7B-Instruct-v0.2",  # For prompt analysis
    "svg_generator": "deepseek-ai/deepseek-coder-6.7b-instruct",  # For SVG generation
    "scene_elaborator": "mistralai/Mistral-7B-Instruct-v0.2",  # For scene elaboration
}

class LLMManager:
    """
    Manager class for handling LLM models loading and inference.
    Optimized for performance across different hardware architectures.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for resource efficiency."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LLMManager, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, 
                 models_config: Optional[Dict[str, str]] = None,
                 cache_dir: str = ".cache/models",
                 device: str = "auto",
                 use_8bit: bool = True,
                 use_4bit: bool = False,
                 max_memory: Optional[Dict[str, str]] = None,
                 offload_folder: Optional[str] = None,
                 max_batch_size: int = 8,
                 enable_gradient_checkpointing: bool = True):
        """
        Initialize the LLM Manager with model configurations.
        
        Args:
            models_config: Dictionary mapping model roles to model names/paths
            cache_dir: Directory for model caching
            device: Device to run models on ('cpu', 'cuda', 'mps', 'xpu', 'auto')
            use_8bit: Whether to use 8-bit quantization
            use_4bit: Whether to use 4-bit quantization (takes precedence over 8-bit)
            max_memory: Memory constraints per device {"cuda:0": "8GiB", "cpu": "32GiB"}
            offload_folder: Directory for offloading model parts to disk
            max_batch_size: Maximum batch size for generation
            enable_gradient_checkpointing: Whether to use gradient checkpointing
        """
        # Initialize only once (singleton pattern)
        if hasattr(self, 'initialized'):
            return
        
        self.lazy_importer = LazyImporter()
        self.models_config = models_config or DEFAULT_MODELS
        self.cache_dir = cache_dir
        self.loaded_models = {}
        self.model_info = {}
        self.offload_folder = offload_folder
        self.max_batch_size = max_batch_size
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        
        # Create offload folder if specified
        if self.offload_folder:
            os.makedirs(self.offload_folder, exist_ok=True)
        
        # Set max memory constraints
        self.max_memory = max_memory
        
        # Determine device
        self.device = self._determine_optimal_device(device)
            
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Performance optimization settings
        self.batch_size = 1  # Default, can be dynamically adjusted
        self.inference_timeout = 60  # Seconds
        
        # Caching for generated responses
        self.response_cache = {}
        self.cache_size_limit = 100  # Number of responses to keep in cache
        
        # Batch processing
        self._batch_queue = queue.Queue()
        self._batch_results = {}
        self._batch_lock = threading.Lock()
        self._batch_thread = None
        self._batch_processing = False
        
        # Track initialization status
        self.initialized = True
        logger.info(f"LLM Manager initialized with device: {self.device}")
        logger.debug(f"Device info: {json.dumps(DEVICE_INFO, indent=2)}")
        
    def _determine_optimal_device(self, requested_device: str) -> str:
        """
        Determine the optimal device based on available hardware and request.
        
        Args:
            requested_device: Requested device ('cpu', 'cuda', 'mps', 'xpu', 'auto')
            
        Returns:
            Optimal device to use
        """
        if requested_device != "auto":
            return requested_device
            
        # Import torch to check device availability
        if self.lazy_importer.import_torch():
            torch = self.lazy_importer.get_module("torch")
            
            # Check CUDA (NVIDIA)
            if DEVICE_INFO["has_cuda"]:
                return "cuda"
                
            # Check MPS (Apple Silicon)
            elif DEVICE_INFO["has_mps"]:
                return "mps"
                
            # Check ROCm (AMD)
            elif DEVICE_INFO["has_rocm"]:
                return "xpu"
                
        # Fall back to CPU
        return "cpu"
        
    def load_model(self, role: str) -> bool:
        """
        Load a model for a specific role.
        
        Args:
            role: Role of the model (e.g., 'prompt_analyzer', 'svg_generator')
            
        Returns:
            Whether model loading was successful
        """
        if role not in self.models_config:
            logger.error(f"No model configured for role: {role}")
            return False
            
        model_name = self.models_config[role]
        
        if role in self.loaded_models:
            logger.info(f"Model for role {role} already loaded: {model_name}")
            return True
            
        logger.info(f"Loading model {model_name} for role {role}")
        
        # Ensure PyTorch and transformers are available
        if not self.lazy_importer.import_torch() or not self.lazy_importer.import_transformers():
            logger.error("Required dependencies not available")
            return False
        
        try:
            start_time = time.time()
            
            # Dynamically import and load based on model type
            if "mistral" in model_name.lower():
                success = self._load_mistral_model(role, model_name)
            elif "llama" in model_name.lower():
                success = self._load_llama_model(role, model_name)
            elif "deepseek" in model_name.lower():
                success = self._load_deepseek_model(role, model_name)
            else:
                success = self._load_generic_model(role, model_name)
                
            if success:
                load_time = time.time() - start_time
                logger.info(f"Successfully loaded model {model_name} for role {role} in {load_time:.2f} seconds")
                return True
            else:
                logger.error(f"Failed to load model {model_name} for role {role}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name} for role {role}: {str(e)}")
            return False
            
    def _load_mistral_model(self, role: str, model_name: str) -> bool:
        """
        Load a Mistral model with optimizations.
        
        Args:
            role: Role of the model
            model_name: Model name or path
            
        Returns:
            Whether loading was successful
        """
        try:
            # Get modules
            AutoTokenizer = self.lazy_importer.get_module("AutoTokenizer")
            AutoModelForCausalLM = self.lazy_importer.get_module("AutoModelForCausalLM")
            torch = self.lazy_importer.get_module("torch")
            
            # Load tokenizer with caching
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=self.cache_dir,
                use_fast=True  # Use faster tokenizer implementation
            )
            
            # Determine quantization and optimization config
            model_kwargs = self._get_model_loading_config()
            
            # Load model with optimizations
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if "cuda" in self.device else None,
                **model_kwargs
            )
            
            # Move model to device and optimize
            self._optimize_model(model)
            
            self.loaded_models[role] = {
                "model": model,
                "tokenizer": tokenizer,
                "type": "mistral"
            }
            
            self.model_info[role] = {
                "name": model_name,
                "type": "mistral",
                "parameters": self._get_model_size(model),
                "quantization": "4bit" if self.use_4bit else "8bit" if self.use_8bit else "none",
                "device": self.device,
                "loaded_at": time.time()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading Mistral model: {str(e)}")
            return False
            
    def _load_llama_model(self, role: str, model_name: str) -> bool:
        """
        Load a Llama model with optimizations.
        
        Args:
            role: Role of the model
            model_name: Model name or path
            
        Returns:
            Whether loading was successful
        """
        try:
            # Get modules
            AutoTokenizer = self.lazy_importer.get_module("AutoTokenizer")
            AutoModelForCausalLM = self.lazy_importer.get_module("AutoModelForCausalLM")
            torch = self.lazy_importer.get_module("torch")
            
            # Load tokenizer with caching
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=self.cache_dir,
                use_fast=True
            )
            
            # Determine quantization and optimization config
            model_kwargs = self._get_model_loading_config()
            
            # Load model with optimizations
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if "cuda" in self.device else None,
                **model_kwargs
            )
            
            # Move model to device and optimize
            self._optimize_model(model)
            
            self.loaded_models[role] = {
                "model": model,
                "tokenizer": tokenizer,
                "type": "llama"
            }
            
            self.model_info[role] = {
                "name": model_name,
                "type": "llama",
                "parameters": self._get_model_size(model),
                "quantization": "4bit" if self.use_4bit else "8bit" if self.use_8bit else "none",
                "device": self.device,
                "loaded_at": time.time()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading Llama model: {str(e)}")
            return False
    
    def _load_deepseek_model(self, role: str, model_name: str) -> bool:
        """
        Load a DeepSeek model with optimizations.
        
        Args:
            role: Role of the model
            model_name: Model name or path
            
        Returns:
            Whether loading was successful
        """
        try:
            # Get modules
            AutoTokenizer = self.lazy_importer.get_module("AutoTokenizer")
            AutoModelForCausalLM = self.lazy_importer.get_module("AutoModelForCausalLM")
            torch = self.lazy_importer.get_module("torch")
            
            # Load tokenizer with caching
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Determine quantization and optimization config
            model_kwargs = self._get_model_loading_config()
            
            # Load model with optimizations
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if "cuda" in self.device else None,
                trust_remote_code=True,
                **model_kwargs
            )
            
            # Move model to device and optimize
            self._optimize_model(model)
            
            self.loaded_models[role] = {
                "model": model,
                "tokenizer": tokenizer,
                "type": "deepseek"
            }
            
            self.model_info[role] = {
                "name": model_name,
                "type": "deepseek",
                "parameters": self._get_model_size(model),
                "quantization": "4bit" if self.use_4bit else "8bit" if self.use_8bit else "none",
                "device": self.device,
                "loaded_at": time.time()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading DeepSeek model: {str(e)}")
            return False
            
    def _load_generic_model(self, role: str, model_name: str) -> bool:
        """
        Load a generic model using Hugging Face Transformers.
        
        Args:
            role: Role of the model
            model_name: Model name or path
            
        Returns:
            Whether loading was successful
        """
        try:
            # Get modules
            AutoTokenizer = self.lazy_importer.get_module("AutoTokenizer")
            AutoModelForCausalLM = self.lazy_importer.get_module("AutoModelForCausalLM")
            torch = self.lazy_importer.get_module("torch")
            
            # Load tokenizer with caching
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=self.cache_dir,
                use_fast=True
            )
            
            # Determine quantization and optimization config
            model_kwargs = self._get_model_loading_config()
            
            # Load model with optimizations
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if "cuda" in self.device else None,
                **model_kwargs
            )
            
            # Move model to device and optimize
            self._optimize_model(model)
            
            self.loaded_models[role] = {
                "model": model,
                "tokenizer": tokenizer,
                "type": "generic"
            }
            
            self.model_info[role] = {
                "name": model_name,
                "type": "generic",
                "parameters": self._get_model_size(model),
                "quantization": "4bit" if self.use_4bit else "8bit" if self.use_8bit else "none",
                "device": self.device,
                "loaded_at": time.time()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading generic model: {str(e)}")
            return False
    
    def _get_model_loading_config(self) -> Dict[str, Any]:
        """
        Get optimized configuration for model loading based on hardware.
        
        Returns:
            Dictionary of model loading arguments
        """
        config = {}
        
        # Memory optimization
        if self.max_memory:
            config["max_memory"] = self.max_memory
            
        # Disk offloading
        if self.offload_folder:
            config["offload_folder"] = self.offload_folder
            
        # Device map
        if self.device != "cpu":
            config["device_map"] = self.device
        else:
            # For CPU, consider using "auto" device map for potential optimizations
            config["device_map"] = "auto"
            
        # Gradient checkpointing
        if self.enable_gradient_checkpointing:
            config["gradient_checkpointing"] = True
            
        # Quantization settings - 4-bit takes precedence
        if self.use_4bit and self.lazy_importer.import_bitsandbytes():
            torch = self.lazy_importer.get_module("torch")
            config.update({
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32
            })
        elif self.use_8bit and self.lazy_importer.import_bitsandbytes():
            config.update({
                "load_in_8bit": True
            })
            
        return config
            
    def _optimize_model(self, model) -> None:
        """
        Apply device-specific optimizations to the model.
        
        Args:
            model: Model to optimize
        """
        torch = self.lazy_importer.get_module("torch")
        
        if torch is None:
            return
            
        # Check if model should be on a specific device
        if self.device == "cuda" and torch.cuda.is_available():
            # Enable CUDA optimizations
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                
            # Enable TF32 precision on Ampere or newer GPUs
            if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
                
        elif self.device == "mps" and hasattr(torch, 'mps') and torch.mps.is_available():
            # MPS-specific optimizations for Apple Silicon
            pass  # No specific optimizations needed currently
            
        # Optimize for CPU if using it
        if self.device == "cpu":
            # Enable OpenMP parallel processing if available
            if hasattr(torch, 'set_num_threads'):
                # Use all available cores but leave one for system operations
                torch.set_num_threads(max(1, DEVICE_INFO["cpu_cores"] - 1))
                
            # Enable MKL optimizations if available
            if hasattr(torch, 'set_num_interop_threads'):
                torch.set_num_interop_threads(max(1, DEVICE_INFO["cpu_cores"] // 2))
        
        # Enable gradient checkpointing if requested
        if self.enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
                
    def _get_model_size(self, model: Any) -> str:
        """
        Get the approximate model size in billions of parameters.
        
        Args:
            model: Model to measure
            
        Returns:
            Size string (e.g., "7.12B")
        """
        try:
            model_parameters = sum(p.numel() for p in model.parameters())
            return f"{model_parameters / 1_000_000_000:.2f}B"
        except:
            return "Unknown"
    
    @lru_cache(maxsize=128)
    def _get_cache_key(self, role: str, prompt: str, max_tokens: int, temperature: float, seed: Optional[int] = None) -> str:
        """
        Generate a cache key for model outputs.
        
        Args:
            role: Model role
            prompt: Input prompt
            max_tokens: Max tokens to generate
            temperature: Generation temperature
            seed: Random seed if used
            
        Returns:
            Cache key string
        """
        import hashlib
        # Create a deterministic hash for the parameters
        params = f"{role}::{prompt}::{max_tokens}::{temperature}::{seed}"
        return hashlib.md5(params.encode()).hexdigest()
    
    def generate(self, 
                role: str, 
                prompt: str, 
                max_tokens: int = 1024, 
                temperature: float = 0.7, 
                num_samples: int = 1,
                stop_sequences: Optional[List[str]] = None,
                seed: Optional[int] = None,
                use_cache: bool = True) -> Union[str, List[str]]:
        """
        Generate text using a model for a specific role.
        
        Args:
            role: Role of the model to use
            prompt: Input prompt for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Generation temperature (higher = more random)
            num_samples: Number of samples to generate
            stop_sequences: Sequences that will stop generation
            seed: Optional seed for reproducibility
            use_cache: Whether to use response caching
            
        Returns:
            Generated text or list of generated texts if num_samples > 1
        """
        # Check cache if enabled
        if use_cache and num_samples == 1 and temperature < 0.1:
            cache_key = self._get_cache_key(role, prompt, max_tokens, temperature, seed)
            if cache_key in self.response_cache:
                logger.debug(f"Cache hit for {role} generation")
                return self.response_cache[cache_key]
        
        if role not in self.loaded_models:
            if not self.load_model(role):
                error_msg = f"Failed to load model for role: {role}"
                logger.error(error_msg)
                return error_msg
                
        model_data = self.loaded_models[role]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        model_type = model_data["type"]
        
        try:
            # Get modules
            torch = self.lazy_importer.get_module("torch")
            
            # Set seed for reproducibility if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            
            # Move inputs to the right device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            # Handle different model types and configurations
            stopping_criteria = self._get_stopping_criteria(tokenizer, stop_sequences, inputs["input_ids"].shape[1])
            
            # Improved generation config
            generation_config = {
                "max_length": max_tokens + inputs["input_ids"].shape[1],
                "do_sample": temperature > 0,
                "temperature": max(0.01, temperature),  # Avoid div by zero issues
                "num_return_sequences": 1,
                "pad_token_id": tokenizer.eos_token_id,
                "attention_mask": inputs["attention_mask"]
            }
            
            # Add stopping criteria if available
            if stopping_criteria:
                generation_config["stopping_criteria"] = stopping_criteria
            
            # Add top_k and top_p if temperature is high enough
            if temperature > 0.1:
                generation_config["top_k"] = 50
                generation_config["top_p"] = 0.95
            
            outputs = []
            
            # Use batch generation if multiple samples requested
            if num_samples > 1:
                # Duplicate inputs for batch generation
                batch_inputs = {
                    "input_ids": inputs["input_ids"].repeat(num_samples, 1),
                    "attention_mask": inputs["attention_mask"].repeat(num_samples, 1)
                }
                
                with torch.no_grad():
                    output_ids = model.generate(
                        **batch_inputs,
                        **generation_config
                    )
                
                # Process each sample
                for i in range(num_samples):
                    generated_text = tokenizer.decode(
                        output_ids[i][inputs["input_ids"].shape[1]:], 
                        skip_special_tokens=True
                    )
                    
                    # Apply manual stopping if criteria wasn't applied during generation
                    if stop_sequences and not stopping_criteria:
                        for stop_seq in stop_sequences:
                            if stop_seq in generated_text:
                                generated_text = generated_text[:generated_text.find(stop_seq)]
                                
                    outputs.append(generated_text)
            else:
                # Single sample generation
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        **generation_config
                    )
                
                # Decode the output
                generated_text = tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
                )
                
                # Apply manual stopping if criteria wasn't applied during generation
                if stop_sequences and not stopping_criteria:
                    for stop_seq in stop_sequences:
                        if stop_seq in generated_text:
                            generated_text = generated_text[:generated_text.find(stop_seq)]
                            
                outputs.append(generated_text)
                
                # Cache the result if applicable
                if use_cache and temperature < 0.1:
                    cache_key = self._get_cache_key(role, prompt, max_tokens, temperature, seed)
                    self.response_cache[cache_key] = generated_text
                    
                    # Limit cache size
                    if len(self.response_cache) > self.cache_size_limit:
                        # Remove oldest entry
                        oldest_key = next(iter(self.response_cache))
                        del self.response_cache[oldest_key]
            
            return outputs[0] if num_samples == 1 else outputs
            
        except Exception as e:
            error_msg = f"Error during generation with model {role}: {str(e)}"
            logger.error(error_msg)
            return error_msg
            
    def generate_batch(self, 
                     role: str, 
                     prompts: List[str], 
                     max_tokens: int = 1024, 
                     temperature: float = 0.7,
                     stop_sequences: Optional[List[str]] = None,
                     use_cache: bool = True,
                     batch_size: Optional[int] = None) -> List[str]:
        """
        Generate text for multiple prompts in an optimized batch.
        
        Args:
            role: Role of the model to use
            prompts: List of prompts to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Generation temperature
            stop_sequences: Sequences that will stop generation
            use_cache: Whether to use response caching
            batch_size: Maximum batch size (defaults to self.max_batch_size)
            
        Returns:
            List of generated texts in same order as prompts
        """
        if not prompts:
            return []
            
        # Use default batch size if not specified
        if batch_size is None:
            batch_size = self.max_batch_size
            
        # Limit batch size to max_batch_size
        batch_size = min(batch_size, self.max_batch_size)
            
        # For small number of prompts, just use individual generation
        if len(prompts) == 1:
            return [self.generate(role, prompts[0], max_tokens, temperature, 1, stop_sequences, None, use_cache)]
        
        # Check cache first for all prompts
        results = [None] * len(prompts)
        prompts_to_generate = []
        indices_to_generate = []
        
        if use_cache and temperature < 0.1:
            for i, prompt in enumerate(prompts):
                cache_key = self._get_cache_key(role, prompt, max_tokens, temperature, None)
                if cache_key in self.response_cache:
                    results[i] = self.response_cache[cache_key]
                else:
                    prompts_to_generate.append(prompt)
                    indices_to_generate.append(i)
        else:
            prompts_to_generate = prompts
            indices_to_generate = list(range(len(prompts)))
        
        # If all results are in cache, return them
        if not prompts_to_generate:
            return results
        
        # Ensure model is loaded
        if role not in self.loaded_models:
            if not self.load_model(role):
                error_msg = f"Failed to load model for role: {role}"
                logger.error(error_msg)
                return [error_msg] * len(prompts_to_generate)
                
        model_data = self.loaded_models[role]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        # Process in optimal batch sizes
        for batch_start in range(0, len(prompts_to_generate), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts_to_generate))
            batch_prompts = prompts_to_generate[batch_start:batch_end]
            batch_indices = indices_to_generate[batch_start:batch_end]
            
            try:
                # Get modules
                torch = self.lazy_importer.get_module("torch")
                
                # Tokenize batch with padding
                batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
                
                # Move to device
                if self.device != "cpu":
                    batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                
                # Create generation config
                generation_config = {
                    "max_length": max_tokens + batch_inputs["input_ids"].size(1),
                    "do_sample": temperature > 0,
                    "temperature": max(0.01, temperature),
                    "num_return_sequences": 1,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                
                # Add stopping criteria if available
                stopping_criteria = self._get_stopping_criteria(tokenizer, stop_sequences, batch_inputs["input_ids"].size(1))
                if stopping_criteria:
                    generation_config["stopping_criteria"] = stopping_criteria
                
                # Add top_k and top_p if temperature is high enough
                if temperature > 0.1:
                    generation_config["top_k"] = 50
                    generation_config["top_p"] = 0.95
                
                # Generate outputs
                with torch.no_grad():
                    output_ids = model.generate(
                        **batch_inputs,
                        **generation_config
                    )
                
                # Process and store results
                for i, output_id in enumerate(output_ids):
                    input_length = batch_inputs["input_ids"].size(1)
                    # Handle possible padding differences
                    if i < len(batch_inputs["input_ids"]):
                        # Find the actual input length for this particular item
                        actual_input_length = torch.sum(batch_inputs["attention_mask"][i]).item()
                        generated_text = tokenizer.decode(
                            output_id[actual_input_length:], 
                            skip_special_tokens=True
                        )
                    else:
                        # Fallback if indices don't match
                        generated_text = tokenizer.decode(
                            output_id[input_length:], 
                            skip_special_tokens=True
                        )
                    
                    # Apply manual stopping if needed
                    if stop_sequences and not stopping_criteria:
                        for stop_seq in stop_sequences:
                            if stop_seq in generated_text:
                                generated_text = generated_text[:generated_text.find(stop_seq)]
                    
                    # Store result
                    orig_index = batch_indices[i]
                    results[orig_index] = generated_text
                    
                    # Cache if needed
                    if use_cache and temperature < 0.1:
                        prompt = prompts[orig_index]
                        cache_key = self._get_cache_key(role, prompt, max_tokens, temperature, None)
                        self.response_cache[cache_key] = generated_text
                
                # Maintain cache size
                if use_cache and len(self.response_cache) > self.cache_size_limit:
                    # Remove excess keys
                    excess = len(self.response_cache) - self.cache_size_limit
                    for _ in range(excess):
                        if self.response_cache:
                            oldest_key = next(iter(self.response_cache))
                            del self.response_cache[oldest_key]
                
            except Exception as e:
                error_msg = f"Error in batch generation: {str(e)}"
                logger.error(error_msg)
                
                # Fill missing results with error message
                for i in batch_indices:
                    if results[i] is None:
                        results[i] = error_msg
        
        return results
            
    def _get_stopping_criteria(self, tokenizer, stop_sequences, input_length):
        """
        Create stopping criteria based on stop sequences.
        
        Args:
            tokenizer: Tokenizer to use
            stop_sequences: List of stop sequences
            input_length: Length of input sequence
            
        Returns:
            StoppingCriteriaList or None
        """
        if not stop_sequences:
            return None
            
        try:
            StoppingCriteria = self.lazy_importer.get_module("StoppingCriteria")
            StoppingCriteriaList = self.lazy_importer.get_module("StoppingCriteriaList")
            
            if not StoppingCriteria or not StoppingCriteriaList:
                return None
                
            class StopSequenceCriteria(StoppingCriteria):
                def __init__(self, tokenizer, stop_sequences, input_length):
                    self.tokenizer = tokenizer
                    self.stop_sequences = stop_sequences
                    self.input_length = input_length
                    
                def __call__(self, input_ids, scores, **kwargs):
                    decoded = self.tokenizer.decode(input_ids[0][self.input_length:])
                    return any(seq in decoded for seq in self.stop_sequences)
                    
            return StoppingCriteriaList([
                StopSequenceCriteria(tokenizer, stop_sequences, input_length)
            ])
            
        except Exception as e:
            logger.warning(f"StoppingCriteria not fully supported: {str(e)}")
            return None
            
    def get_model_info(self, role: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Args:
            role: Specific role to get info for, or None for all models
            
        Returns:
            Dictionary with model information
        """
        if role:
            return self.model_info.get(role, {"status": "not_loaded"})
        return self.model_info
        
    def unload_model(self, role: str) -> bool:
        """
        Unload a model to free up resources with proper cleanup.
        
        Args:
            role: Role of the model to unload
            
        Returns:
            Whether the model was successfully unloaded
        """
        if role not in self.loaded_models:
            return True  # Already unloaded
            
        try:
            torch = self.lazy_importer.get_module("torch")
            
            # Get model and check device
            model_data = self.loaded_models[role]
            model = model_data["model"]
            
            # Clear from loaded models dictionary
            del self.loaded_models[role]
            
            # Update info
            if role in self.model_info:
                self.model_info[role]["status"] = "unloaded"
                self.model_info[role]["unloaded_at"] = time.time()
            
            # Run garbage collection to free GPU memory
            if torch:
                # Execute model-specific cleanup
                model = model.to('cpu')  # Move to CPU first
                del model
                
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Extra optimization for CUDA
                    torch.cuda.synchronize()
                    
                # Clear MPS cache if on Apple Silicon
                if hasattr(torch, 'mps') and torch.mps.is_available():
                    torch.mps.empty_cache()
                    
            logger.info(f"Unloaded model for role: {role}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model for role {role}: {str(e)}")
            return False
    
    def adjust_for_available_memory(self) -> None:
        """
        Dynamically adjust model settings based on available memory.
        """
        torch = self.lazy_importer.get_module("torch")
        if not torch:
            return
            
        try:
            if torch.cuda.is_available():
                # Get total and available GPU memory
                total_memory = torch.cuda.get_device_properties(0).total_memory
                reserved_memory = torch.cuda.memory_reserved(0)
                allocated_memory = torch.cuda.memory_allocated(0)
                free_memory = total_memory - reserved_memory
                
                # Adjust quantization based on available memory
                if free_memory < 4 * 1024 * 1024 * 1024:  # Less than 4GB free
                    logger.info("Low GPU memory detected. Enabling 4-bit quantization.")
                    self.use_8bit = False
                    self.use_4bit = True
                elif free_memory < 8 * 1024 * 1024 * 1024:  # Less than 8GB free
                    logger.info("Limited GPU memory detected. Enabling 8-bit quantization.")
                    self.use_8bit = True
                    self.use_4bit = False
                    
                # Log memory state
                logger.debug(f"GPU Memory: Total={total_memory/1e9:.2f}GB, "
                           f"Reserved={reserved_memory/1e9:.2f}GB, "
                           f"Allocated={allocated_memory/1e9:.2f}GB, "
                           f"Free={free_memory/1e9:.2f}GB")
            
            # Check system memory
            import psutil
            system_memory = psutil.virtual_memory()
            if system_memory.available < 8 * 1024 * 1024 * 1024:  # Less than 8GB available
                logger.warning("Low system memory. Models may be unstable.")
                
        except Exception as e:
            logger.warning(f"Could not adjust for available memory: {str(e)}")
                
    def optimize_tokenizer(self, tokenizer) -> None:
        """
        Apply tokenizer optimizations.
        
        Args:
            tokenizer: Tokenizer to optimize
        """
        # Check if fast tokenizer is available
        if hasattr(tokenizer, 'is_fast') and not tokenizer.is_fast:
            logger.warning("Using slow tokenizer. Performance may be impacted.")
        
        # Set padding strategy for better batch processing
        tokenizer.padding_side = 'left'  # generally better for causal models
        
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token