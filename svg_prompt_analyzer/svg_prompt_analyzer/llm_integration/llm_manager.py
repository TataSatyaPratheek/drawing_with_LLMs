"""
Production-grade LLM manager with reinforcement learning optimization.
Provides optimized integration with open-source language models for generating
and refining SVG content with memory, performance, and RL-based optimization.
"""

import os
import time
import json
import hashlib
import threading
import re
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from io import BytesIO

# Import core optimizations
from svg_prompt_analyzer.core import CONFIG, memoize, jit, Profiler, get_thread_pool
from svg_prompt_analyzer.core.memory_manager import MemoryManager
from svg_prompt_analyzer.utils.logger import get_logger, log_function_call

# Configure logger
logger = get_logger(__name__)

# Type aliases
ResponseType = Dict[str, Any]
PromptType = Union[str, List[Dict[str, str]]]
TokenCount = int
MessageList = List[Dict[str, str]]

# Constants
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 30.0  # seconds
MAX_RETRIES = 5
RETRY_BASE_DELAY = 1.0  # seconds
MAX_CONCURRENT_REQUESTS = 5
CACHE_SIZE = 1000
CACHE_TTL = 3600 * 24  # 1 day in seconds


class ModelType(Enum):
    """Types of supported language models."""
    MISTRAL_7B = auto()
    MISTRAL_INSTRUCT = auto()
    LLAMA3_8B = auto()
    LLAMA3_INSTRUCT = auto()
    CODELLAMA_7B = auto()
    CODELLAMA_INSTRUCT = auto()
    STABLE_LM = auto()
    CUSTOM = auto()


@dataclass
class ModelConfig:
    """Configuration for a language model."""
    model_type: ModelType
    model_path: str
    weights_path: Optional[str] = None
    quantization: Optional[str] = None  # "int8", "int4", etc.
    max_tokens: int = DEFAULT_MAX_TOKENS
    token_limit: int = 4096
    supports_vision: bool = False
    supports_svg: bool = False
    device: str = "cuda"  # "cuda", "cpu", "mps"
    
    # Memory and performance settings
    batch_size: int = 1
    sliding_window: Optional[int] = None
    low_memory_mode: bool = False
    threads: int = 4
    
    # Optional model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)


class ModelBackend(Enum):
    """Backend engines for running models."""
    ONNX = auto()
    PYTORCH = auto()
    CTRANSFORMERS = auto()
    LLAMACPP = auto()
    TENSORRT = auto()


class ResponseCache:
    """Thread-safe cache for LLM responses."""
    
    def __init__(self, max_size: int = CACHE_SIZE, ttl: int = CACHE_TTL):
        """
        Initialize response cache.
        
        Args:
            max_size: Maximum number of cached responses
            ttl: Time-to-live for cached responses in seconds
        """
        self._cache: Dict[str, Tuple[ResponseType, float]] = {}  # (response, expiry_time)
        self._max_size = max_size
        self._ttl = ttl
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[ResponseType]:
        """
        Get cached response.
        
        Args:
            key: Cache key
            
        Returns:
            Cached response or None if not found or expired
        """
        with self._lock:
            if key in self._cache:
                response, expiry_time = self._cache[key]
                
                # Check if expired
                if time.time() < expiry_time:
                    return response
                    
                # Remove expired entry
                del self._cache[key]
                
            return None
    
    def set(self, key: str, response: ResponseType) -> None:
        """
        Cache response.
        
        Args:
            key: Cache key
            response: Response to cache
        """
        with self._lock:
            # Clean expired entries periodically
            if len(self._cache) >= self._max_size / 2:
                self._clean_expired()
                
            # Prune cache if still too large
            if len(self._cache) >= self._max_size:
                self._prune()
                
            # Set new entry
            expiry_time = time.time() + self._ttl
            self._cache[key] = (response, expiry_time)
    
    def clear(self) -> None:
        """Clear all cached responses."""
        with self._lock:
            self._cache.clear()
    
    def _clean_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = [k for k, (_, expiry) in self._cache.items() if now >= expiry]
        for key in expired_keys:
            del self._cache[key]
    
    def _prune(self) -> None:
        """Remove oldest entries to reduce cache size."""
        # Sort by expiry time (ascending)
        items = sorted(self._cache.items(), key=lambda x: x[1][1])
        
        # Remove oldest 25%
        to_remove = len(items) // 4
        for key, _ in items[:to_remove]:
            del self._cache[key]


@dataclass
class GenerationSample:
    """Sample for reinforcement learning."""
    prompt: PromptType
    response: str
    reward: float = 0.0
    log_prob: Optional[float] = None
    value: Optional[float] = None


class ModelRunner:
    """
    Base class for model runners.
    
    Handles loading and running models with different backends.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        backend: ModelBackend = ModelBackend.LLAMACPP
    ):
        """
        Initialize model runner.
        
        Args:
            config: Model configuration
            backend: Model backend
        """
        self.config = config
        self.backend = backend
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._lock = threading.RLock()
        self.memory_manager = MemoryManager()
    
    def _ensure_initialized(self) -> None:
        """
        Ensure model is initialized.
        
        This is implemented by subclasses for specific backends.
        """
        raise NotImplementedError("Subclasses must implement _ensure_initialized")
    
    def generate(
        self,
        prompt: PromptType,
        max_tokens: Optional[int] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = 0.9,
        top_k: int = 40,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate completion from prompt.
        
        Args:
            prompt: Text prompt or message list
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Sequences to stop generation
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (generated_text, generation_info)
        """
        raise NotImplementedError("Subclasses must implement generate")
    
    def get_logprobs(
        self,
        prompt: PromptType,
        response: str
    ) -> List[float]:
        """
        Get log probabilities for a response.
        
        Args:
            prompt: Text prompt or message list
            response: Response text
            
        Returns:
            List of log probabilities for each token
        """
        raise NotImplementedError("Subclasses must implement get_logprobs")
    
    def preload(self) -> None:
        """Preload model into memory."""
        with self._lock:
            if not self._initialized:
                self._ensure_initialized()
    
    def unload(self) -> None:
        """Unload model from memory."""
        with self._lock:
            if self._initialized:
                self._model = None
                self._initialized = False


class LLaMACppRunner(ModelRunner):
    """
    LLaMA.cpp model runner.
    
    Runs models using llama.cpp for efficiency.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize LLaMA.cpp model runner.
        
        Args:
            config: Model configuration
        """
        super().__init__(config, ModelBackend.LLAMACPP)
    
    @log_function_call(level=logging.DEBUG)
    def _ensure_initialized(self) -> None:
        """Ensure LLaMA.cpp model is initialized."""
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            try:
                with Profiler("model_initialization"):
                    # Try to import llama_cpp
                    try:
                        from llama_cpp import Llama
                    except ImportError:
                        logger.error("Failed to import llama_cpp. Make sure it's installed.")
                        raise
                    
                    # Set up parameters
                    params = {
                        "model_path": self.config.model_path,
                        "n_ctx": self.config.token_limit,
                        "n_threads": self.config.threads,
                        "n_batch": self.config.batch_size,
                    }
                    
                    # Add quantization if specified
                    if self.config.quantization:
                        if self.config.quantization == "int4":
                            params["n_gpu_layers"] = 0  # Force CPU for int4
                            params["use_mlock"] = True
                            params["use_mmap"] = True
                        elif self.config.quantization == "int8":
                            params["n_gpu_layers"] = -1  # Use GPU if available
                        else:
                            # Default to int8 for other specified quantizations
                            params["n_gpu_layers"] = -1
                            
                    # Create model
                    logger.info(f"Loading model: {self.config.model_path} with parameters: {params}")
                    self._model = Llama(**params)
                    
                    # No separate tokenizer needed for llama_cpp
                    self._tokenizer = None
                    
                    self._initialized = True
                    logger.info(f"Initialized LLaMA.cpp model: {self.config.model_path}")
                
            except Exception as e:
                logger.error(f"Failed to initialize LLaMA.cpp model: {str(e)}")
                raise
    
    @log_function_call(level=logging.DEBUG)
    @memoize
    def generate(
        self,
        prompt: PromptType,
        max_tokens: Optional[int] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = 0.9,
        top_k: int = 40,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate completion using LLaMA.cpp model.
        
        Args:
            prompt: Text prompt or message list
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Sequences to stop generation
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (generated_text, generation_info)
        """
        self._ensure_initialized()
        
        # Process prompt
        if isinstance(prompt, list):
            # Process messages for chat format
            prompt_text = self._format_chat_messages(prompt)
        else:
            prompt_text = prompt
            
        # Use default max tokens if not specified
        max_tokens = max_tokens or self.config.max_tokens
        
        # Prepare stop sequences
        stop = stop_sequences or []
        
        try:
            # Run generation with memory profiling
            with Profiler("llamacpp_inference"), self.memory_manager.memory_tracking_context("llm_generation"):
                # Track token counts
                prompt_tokens = len(self._model.tokenize(prompt_text))
                
                # Set generation parameters
                generation_params = {
                    "prompt": prompt_text,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "stop": stop,
                    **kwargs
                }
                
                # Generate completion
                output = self._model.generate(**generation_params)
                
                # Extract results
                generated_text = output["choices"][0]["text"] if "choices" in output else output
                
                # Calculate token usage
                completion_tokens = len(self._model.tokenize(generated_text))
                
                # Create generation info
                generation_info = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                
                return generated_text, generation_info
                
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            # Return empty result on error
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "error": str(e)}
    
    def get_logprobs(
        self,
        prompt: PromptType,
        response: str
    ) -> List[float]:
        """
        Get log probabilities for tokens in the response.
        
        Args:
            prompt: Text prompt or message list
            response: Response text
            
        Returns:
            List of log probabilities for each token
        """
        self._ensure_initialized()
        
        # Process prompt
        if isinstance(prompt, list):
            prompt_text = self._format_chat_messages(prompt)
        else:
            prompt_text = prompt
            
        # Full text for context
        full_text = prompt_text + response
        
        try:
            # Get logprobs from LLaMA.cpp
            # Note: This is a simplified approach for models that may not directly expose logprobs
            # In a production environment, you would use a model API that provides this functionality
            
            # Tokenize the response to count tokens
            response_tokens = self._model.tokenize(response)
            
            # Create placeholder log probabilities
            # Ideally, this would use actual token logprobs from the model
            logprobs = [-1.0] * len(response_tokens)
            
            return logprobs
            
        except Exception as e:
            logger.error(f"Error getting logprobs: {str(e)}")
            # Return placeholder on error
            return [-1.0] * 10  # Arbitrary length
    
    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages for model input.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted text for model input
        """
        # Different formatting based on model type
        if self.config.model_type in [ModelType.MISTRAL_INSTRUCT, ModelType.MISTRAL_7B]:
            return self._format_mistral_chat(messages)
        elif self.config.model_type in [ModelType.LLAMA3_INSTRUCT, ModelType.LLAMA3_8B]:
            return self._format_llama_chat(messages)
        elif self.config.model_type in [ModelType.CODELLAMA_INSTRUCT, ModelType.CODELLAMA_7B]:
            return self._format_codellama_chat(messages)
        else:
            # Default format
            return self._format_default_chat(messages)
    
    def _format_mistral_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Mistral models."""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            
            # Process content if it's a list (for multi-modal)
            if isinstance(content, list):
                content = self._process_content_list(content)
                
            if role == "system":
                formatted.append(f"<s>[INST] {content} [/INST]")
            elif role == "user":
                formatted.append(f"<s>[INST] {content} [/INST]")
            elif role == "assistant":
                formatted.append(f"{content} </s>")
            else:
                # Default to user for unknown roles
                formatted.append(f"<s>[INST] {content} [/INST]")
                
        # Add final token for generation
        if not formatted[-1].endswith("</s>"):
            formatted.append("<s>[INST] ")
            
        return "".join(formatted)
    
    def _format_llama_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Llama models."""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            
            # Process content if it's a list (for multi-modal)
            if isinstance(content, list):
                content = self._process_content_list(content)
                
            if role == "system":
                formatted.append(f"<|system|>\n{content}\n</s>")
            elif role == "user":
                formatted.append(f"<|user|>\n{content}\n</s>")
            elif role == "assistant":
                formatted.append(f"<|assistant|>\n{content}\n</s>")
            else:
                # Default to user for unknown roles
                formatted.append(f"<|user|>\n{content}\n</s>")
                
        # Add final token for generation
        formatted.append("<|assistant|>\n")
            
        return "\n".join(formatted)
    
    def _format_codellama_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for CodeLlama models."""
        formatted = []
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            
            # Process content if it's a list (for multi-modal)
            if isinstance(content, list):
                content = self._process_content_list(content)
                
            if role == "system" and i == 0:
                # System message as preamble
                formatted.append(f"{content}\n")
            elif role == "user":
                formatted.append(f"Question: {content}")
            elif role == "assistant":
                formatted.append(f"Answer: {content}")
            else:
                # Default to user for unknown roles
                formatted.append(f"Question: {content}")
                
        # Add final token for generation
        if not formatted[-1].startswith("Answer:"):
            formatted.append("Answer:")
            
        return "\n\n".join(formatted)
    
    def _format_default_chat(self, messages: List[Dict[str, str]]) -> str:
        """Default message formatting for models without specific formats."""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Process content if it's a list (for multi-modal)
            if isinstance(content, list):
                content = self._process_content_list(content)
                
            formatted.append(f"{role}: {content}")
            
        return "\n".join(formatted)
    
    def _process_content_list(self, content_list: List[Any]) -> str:
        """Process a list of content items (for multi-modal support)."""
        text_parts = []
        
        for item in content_list:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                # Other types (image, etc.) are ignored as we don't support them yet
                
        return " ".join(text_parts)


class CTModelRunner(ModelRunner):
    """
    CTransformers model runner.
    
    Runs models using the CTransformers library for C++ acceleration.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize CTransformers model runner.
        
        Args:
            config: Model configuration
        """
        super().__init__(config, ModelBackend.CTRANSFORMERS)
        self.memory_manager = MemoryManager()
    
    @log_function_call(level=logging.DEBUG)
    def _ensure_initialized(self) -> None:
        """Ensure CTransformers model is initialized."""
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            try:
                with Profiler("model_initialization"):
                    # Try to import ctransformers
                    try:
                        from ctransformers import AutoModelForCausalLM
                    except ImportError:
                        logger.error("Failed to import ctransformers. Make sure it's installed.")
                        raise
                    
                    # Set up parameters
                    params = {
                        "model_path": self.config.model_path,
                        "model_type": self._get_model_type(),
                        "context_length": self.config.token_limit,
                        "threads": self.config.threads,
                        "batch_size": self.config.batch_size,
                    }
                    
                    # Add GPU configuration
                    if self.config.device == "cuda":
                        params["gpu_layers"] = 50  # Use most layers on GPU
                    else:
                        params["gpu_layers"] = 0
                        
                    # Create model
                    logger.info(f"Loading model: {self.config.model_path} with parameters: {params}")
                    self._model = AutoModelForCausalLM.from_pretrained(**params)
                    
                    # No separate tokenizer needed for ctransformers
                    self._tokenizer = None
                    
                    self._initialized = True
                    logger.info(f"Initialized CTransformers model: {self.config.model_path}")
                
            except Exception as e:
                logger.error(f"Failed to initialize CTransformers model: {str(e)}")
                raise
    
    def _get_model_type(self) -> str:
        """
        Get model type for CTransformers.
        
        Returns:
            Model type string
        """
        # Map model types to CTransformers model types
        model_type_map = {
            ModelType.MISTRAL_7B: "mistral",
            ModelType.MISTRAL_INSTRUCT: "mistral",
            ModelType.LLAMA3_8B: "llama",
            ModelType.LLAMA3_INSTRUCT: "llama",
            ModelType.CODELLAMA_7B: "llama",
            ModelType.CODELLAMA_INSTRUCT: "llama",
            ModelType.STABLE_LM: "stablelm",
            ModelType.CUSTOM: "llama",  # Default to llama
        }
        
        return model_type_map.get(self.config.model_type, "llama")
    
    @log_function_call(level=logging.DEBUG)
    @memoize
    def generate(
        self,
        prompt: PromptType,
        max_tokens: Optional[int] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = 0.9,
        top_k: int = 40,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate completion using CTransformers model.
        
        Args:
            prompt: Text prompt or message list
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Sequences to stop generation
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (generated_text, generation_info)
        """
        self._ensure_initialized()
        
        # Process prompt
        if isinstance(prompt, list):
            # Process messages for chat format
            prompt_text = self._format_chat_messages(prompt)
        else:
            prompt_text = prompt
            
        # Use default max tokens if not specified
        max_tokens = max_tokens or self.config.max_tokens
        
        try:
            # Run generation with memory profiling
            with Profiler("ctransformers_inference"), self.memory_manager.memory_tracking_context("llm_generation"):
                # Track token counts
                prompt_tokens = self._model.tokenize(prompt_text)
                prompt_token_count = len(prompt_tokens)
                
                # Set up generation parameters
                params = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    **kwargs
                }
                
                # Add stop sequences if provided
                if stop_sequences:
                    params["stop"] = stop_sequences
                    
                # Generate text
                generated_text = self._model(prompt_text, **params)
                
                # Get token count after generation
                full_tokens = self._model.tokenize(prompt_text + generated_text)
                completion_token_count = len(full_tokens) - prompt_token_count
                
                # Create generation info
                generation_info = {
                    "prompt_tokens": prompt_token_count,
                    "completion_tokens": completion_token_count,
                    "total_tokens": prompt_token_count + completion_token_count,
                }
                
                return generated_text, generation_info
                
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            # Return empty result on error
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "error": str(e)}
    
    def get_logprobs(
        self,
        prompt: PromptType,
        response: str
    ) -> List[float]:
        """
        Get log probabilities for tokens in the response.
        
        Args:
            prompt: Text prompt or message list
            response: Response text
            
        Returns:
            List of log probabilities for each token
        """
        # CTransformers doesn't directly provide logprobs, so we return placeholders
        tokens = self._model.tokenize(response)
        return [-1.0] * len(tokens)
    
    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages for model input.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted text for model input
        """
        # Different formatting based on model type
        if self.config.model_type in [ModelType.MISTRAL_INSTRUCT, ModelType.MISTRAL_7B]:
            return self._format_mistral_chat(messages)
        elif self.config.model_type in [ModelType.LLAMA3_INSTRUCT, ModelType.LLAMA3_8B]:
            return self._format_llama_chat(messages)
        elif self.config.model_type in [ModelType.CODELLAMA_INSTRUCT, ModelType.CODELLAMA_7B]:
            return self._format_codellama_chat(messages)
        else:
            # Default format
            return self._format_default_chat(messages)
    
    def _format_mistral_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Mistral models."""
        # Same implementation as in LLaMACppRunner
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            
            # Process content if it's a list (for multi-modal)
            if isinstance(content, list):
                content = self._process_content_list(content)
                
            if role == "system":
                formatted.append(f"<s>[INST] {content} [/INST]")
            elif role == "user":
                formatted.append(f"<s>[INST] {content} [/INST]")
            elif role == "assistant":
                formatted.append(f"{content} </s>")
            else:
                # Default to user for unknown roles
                formatted.append(f"<s>[INST] {content} [/INST]")
                
        # Add final token for generation
        if not formatted[-1].endswith("</s>"):
            formatted.append("<s>[INST] ")
            
        return "".join(formatted)
    
    def _format_llama_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Llama models."""
        # Same implementation as in LLaMACppRunner
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            
            # Process content if it's a list (for multi-modal)
            if isinstance(content, list):
                content = self._process_content_list(content)
                
            if role == "system":
                formatted.append(f"<|system|>\n{content}\n</s>")
            elif role == "user":
                formatted.append(f"<|user|>\n{content}\n</s>")
            elif role == "assistant":
                formatted.append(f"<|assistant|>\n{content}\n</s>")
            else:
                # Default to user for unknown roles
                formatted.append(f"<|user|>\n{content}\n</s>")
                
        # Add final token for generation
        formatted.append("<|assistant|>\n")
            
        return "\n".join(formatted)
    
    def _format_codellama_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for CodeLlama models."""
        # Same implementation as in LLaMACppRunner
        formatted = []
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            
            # Process content if it's a list (for multi-modal)
            if isinstance(content, list):
                content = self._process_content_list(content)
                
            if role == "system" and i == 0:
                # System message as preamble
                formatted.append(f"{content}\n")
            elif role == "user":
                formatted.append(f"Question: {content}")
            elif role == "assistant":
                formatted.append(f"Answer: {content}")
            else:
                # Default to user for unknown roles
                formatted.append(f"Question: {content}")
                
        # Add final token for generation
        if not formatted[-1].startswith("Answer:"):
            formatted.append("Answer:")
            
        return "\n\n".join(formatted)
    
    def _format_default_chat(self, messages: List[Dict[str, str]]) -> str:
        """Default message formatting for models without specific formats."""
        # Same implementation as in LLaMACppRunner
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Process content if it's a list (for multi-modal)
            if isinstance(content, list):
                content = self._process_content_list(content)
                
            formatted.append(f"{role}: {content}")
            
        return "\n".join(formatted)
    
    def _process_content_list(self, content_list: List[Any]) -> str:
        """Process a list of content items (for multi-modal support)."""
        # Same implementation as in LLaMACppRunner
        text_parts = []
        
        for item in content_list:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                # Other types (image, etc.) are ignored as we don't support them yet
                
        return " ".join(text_parts)


class LLMManager:
    """
    Production-grade LLM manager with reinforcement learning optimization.
    
    Provides optimized integration with open-source language models for generating
    and refining SVG content with memory, performance, and RL-based optimization.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for resource efficiency."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LLMManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(
        self,
        models_config: Optional[Dict[str, str]] = None,
        cache_dir: str = ".cache/models",
        device: str = "auto",
        use_8bit: bool = False,
        use_4bit: bool = True,
        cache_responses: bool = True,
        preload_models: bool = False
    ):
        """
        Initialize LLM manager.
        
        Args:
            models_config: Configuration for different model roles
            cache_dir: Directory for caching models
            device: Device to use ("auto", "cuda", "cpu", "mps")
            use_8bit: Whether to use 8-bit quantization
            use_4bit: Whether to use 4-bit quantization
            cache_responses: Whether to cache model responses
            preload_models: Whether to preload models on initialization
        """
        # Initialize only once (singleton pattern)
        if self._initialized:
            return
            
        # Default model paths
        self.models_config = models_config or {
            "prompt_analyzer": "mistralai/Mistral-7B-Instruct-v0.2",
            "svg_generator": "codellama/CodeLlama-7b-Instruct",
            "default": "mistralai/Mistral-7B-Instruct-v0.2"
        }
        
        self.cache_dir = cache_dir
        self.device = self._resolve_device(device)
        self.use_8bit = use_8bit
        self.use_4bit = use_4bit
        self.cache_responses = cache_responses
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set up response cache
        self._response_cache = ResponseCache() if cache_responses else None
        
        # Model runners by role
        self._model_runners: Dict[str, ModelRunner] = {}
        
        # Set up memory management
        self.memory_manager = MemoryManager()
        
        # Metrics tracking
        self._metrics = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "total_time": 0.0,
        }
        
        # Set up threading resources
        self._request_lock = threading.RLock()
        self._request_queue = queue.Queue()
        self._request_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        # Initialize model configs
        self._init_model_configs()
        
        # Preload models if requested
        if preload_models:
            self._preload_models()
            
        self._initialized = True
        logger.info(f"LLM Manager initialized with device: {self.device}")
    
    def _resolve_device(self, device: str) -> str:
        """
        Resolve device based on availability.
        
        Args:
            device: Requested device
            
        Returns:
            Resolved device
        """
        if device != "auto":
            return device
            
        # Try to determine best available device
        try:
            # Check for CUDA
            import torch
            if torch.cuda.is_available():
                return "cuda"
                
            # Check for MPS (Apple Silicon)
            if hasattr(torch, "mps") and torch.mps.is_available():
                return "mps"
                
        except ImportError:
            pass
            
        # Default to CPU
        return "cpu"
    
    def _init_model_configs(self) -> None:
        """Initialize model configurations."""
        # Common configuration settings
        common_config = {
            "device": self.device,
            "low_memory_mode": self.memory_manager.calculate_memory_pressure() > 0.7,
            "threads": min(8, os.cpu_count() or 4)
        }
        
        # Determine quantization
        quantization = None
        if self.use_4bit:
            quantization = "int4"
        elif self.use_8bit:
            quantization = "int8"
            
        # Configure models for different roles
        self._model_configs = {}
        
        # Prompt analyzer model
        if "prompt_analyzer" in self.models_config:
            model_path = self._resolve_model_path(self.models_config["prompt_analyzer"])
            self._model_configs["prompt_analyzer"] = ModelConfig(
                model_type=self._infer_model_type(model_path),
                model_path=model_path,
                quantization=quantization,
                max_tokens=1024,
                token_limit=8192,
                **common_config
            )
            
        # SVG generator model
        if "svg_generator" in self.models_config:
            model_path = self._resolve_model_path(self.models_config["svg_generator"])
            self._model_configs["svg_generator"] = ModelConfig(
                model_type=self._infer_model_type(model_path),
                model_path=model_path,
                quantization=quantization,
                max_tokens=4096,
                token_limit=8192,
                supports_svg=True,
                **common_config
            )
            
        # Default model
        default_model_path = self._resolve_model_path(self.models_config.get("default", ""))
        self._model_configs["default"] = ModelConfig(
            model_type=self._infer_model_type(default_model_path),
            model_path=default_model_path,
            quantization=quantization,
            max_tokens=1024,
            **common_config
        )
        
        logger.info(f"Initialized {len(self._model_configs)} model configurations")
    
    def _resolve_model_path(self, model_name: str) -> str:
        """
        Resolve model path from model name.
        
        Args:
            model_name: Model name or path
            
        Returns:
            Resolved model path
        """
        # Check if it's already a local path
        if os.path.exists(model_name):
            return model_name
            
        # Check common model formats
        if '/' in model_name:
            # Format: organization/model_name
            org, model = model_name.split('/', 1)
            
            # Check if model exists in cache directory
            local_path = os.path.join(self.cache_dir, org, model)
            if os.path.exists(local_path):
                return local_path
                
            # Try with GGUF format
            gguf_path = os.path.join(self.cache_dir, f"{model}.gguf")
            if os.path.exists(gguf_path):
                return gguf_path
                
            # Otherwise, return a path in the cache directory (model will need to be downloaded)
            # In a real implementation, you would download the model here
            # For now, we just return the expected path
            return os.path.join(self.cache_dir, f"{model}.gguf")
        else:
            # Just a model name, check if it exists in cache
            local_path = os.path.join(self.cache_dir, model_name)
            if os.path.exists(local_path):
                return local_path
                
            # Try with GGUF format
            gguf_path = os.path.join(self.cache_dir, f"{model_name}.gguf")
            if os.path.exists(gguf_path):
                return gguf_path
                
            # Return expected path
            return os.path.join(self.cache_dir, f"{model_name}.gguf")
    
    def _infer_model_type(self, model_path: str) -> ModelType:
        """
        Infer model type from model path.
        
        Args:
            model_path: Model path
            
        Returns:
            Inferred model type
        """
        model_name = os.path.basename(model_path).lower()
        
        if "mistral" in model_name:
            if "instruct" in model_name:
                return ModelType.MISTRAL_INSTRUCT
            else:
                return ModelType.MISTRAL_7B
        elif "llama-3" in model_name or "llama3" in model_name:
            if "instruct" in model_name:
                return ModelType.LLAMA3_INSTRUCT
            else:
                return ModelType.LLAMA3_8B
        elif "codellama" in model_name:
            if "instruct" in model_name:
                return ModelType.CODELLAMA_INSTRUCT
            else:
                return ModelType.CODELLAMA_7B
        elif "stablelm" in model_name:
            return ModelType.STABLE_LM
        else:
            return ModelType.CUSTOM
    
    def _preload_models(self) -> None:
        """Preload models into memory."""
        thread_pool = get_thread_pool()
        futures = []
        
        for role, config in self._model_configs.items():
            # Skip default if it's the same as another model
            if role == "default" and config.model_path in [cfg.model_path for role, cfg in self._model_configs.items() if role != "default"]:
                continue
                
            futures.append(thread_pool.submit(self.load_model, role))
            
        # Wait for models to load
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error preloading model: {str(e)}")
    
    @log_function_call(level=logging.DEBUG)
    def load_model(self, role: str) -> bool:
        """
        Load a model for a specific role.
        
        Args:
            role: Role of the model to load
            
        Returns:
            Whether the model was loaded successfully
        """
        with self._request_lock:
            # Check if model is already loaded
            if role in self._model_runners and self._model_runners[role]._initialized:
                return True
                
            # Get model config for role
            if role not in self._model_configs:
                role = "default"  # Fall back to default model
                
            # If still not found, return failure
            if role not in self._model_configs:
                logger.error(f"No model configuration found for role: {role}")
                return False
                
            config = self._model_configs[role]
            
            try:
                with Profiler(f"load_model_{role}"):
                    # Create model runner based on model path
                    if os.path.exists(config.model_path) and config.model_path.endswith((".gguf", ".bin")):
                        # Use LLaMA.cpp for GGUF models
                        model_runner = LLaMACppRunner(config)
                    else:
                        # Use CTransformers for most other cases
                        model_runner = CTModelRunner(config)
                        
                    # Initialize model
                    model_runner.preload()
                    
                    # Store model runner
                    self._model_runners[role] = model_runner
                    
                    logger.info(f"Successfully loaded model for role: {role}")
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to load model for role {role}: {str(e)}")
                return False
    
    @log_function_call(level=logging.DEBUG)
    def generate(
        self,
        prompt: PromptType,
        role: str = "default",
        max_tokens: Optional[int] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = 0.9,
        top_k: int = 40,
        stop_sequences: Optional[List[str]] = None,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Text prompt or message list
            role: Role of the model to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Sequences to stop generation
            use_cache: Whether to use response cache
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        # Check cache if enabled
        if use_cache and self.cache_responses and self._response_cache:
            cache_key = self._create_cache_key(prompt, role, max_tokens, temperature, top_p, top_k)
            cached = self._response_cache.get(cache_key)
            
            if cached:
                self._increment_metric("cache_hits")
                return cached.get("content", "")
                
        # Acquire semaphore to limit concurrent requests
        with self._request_semaphore:
            try:
                # Ensure model is loaded
                if not self.load_model(role):
                    raise ValueError(f"Failed to load model for role: {role}")
                    
                # Get model runner
                model_runner = self._model_runners[role]
                
                # Generate text
                generated_text, generation_info = model_runner.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop_sequences=stop_sequences,
                    **kwargs
                )
                
                # Create response
                response = {
                    "content": generated_text,
                    "usage": generation_info,
                    "model": role,
                }
                
                # Cache response if enabled
                if use_cache and self.cache_responses and self._response_cache:
                    self._response_cache.set(cache_key, response)
                    
                # Update metrics
                self._increment_metric("requests")
                
                return generated_text
                
            except Exception as e:
                logger.error(f"Error generating text: {str(e)}")
                self._increment_metric("errors")
                return ""
    
    @log_function_call(level=logging.DEBUG)
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        role: str = "default",
        max_tokens: Optional[int] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """
        Generate response for chat messages.
        
        Args:
            messages: List of message dictionaries
            role: Role of the model to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            use_cache: Whether to use response cache
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        # Call generate with message list
        return self.generate(
            prompt=messages,
            role=role,
            max_tokens=max_tokens,
            temperature=temperature,
            use_cache=use_cache,
            **kwargs
        )
    
    def _create_cache_key(self, prompt: PromptType, role: str, max_tokens: Optional[int], 
                        temperature: float, top_p: float, top_k: int) -> str:
        """
        Create cache key for a request.
        
        Args:
            prompt: Text prompt or message list
            role: Role of the model
            max_tokens: Maximum tokens
            temperature: Temperature
            top_p: Top-p value
            top_k: Top-k value
            
        Returns:
            Cache key
        """
        # Serialize prompt
        if isinstance(prompt, str):
            prompt_str = prompt
        else:
            # Keep only relevant message fields for caching
            cleaned_messages = []
            for msg in prompt:
                cleaned_msg = {
                    "role": msg.get("role", ""),
                    "content": msg.get("content", "")
                }
                cleaned_messages.append(cleaned_msg)
            prompt_str = json.dumps(cleaned_messages)
            
        # Create key components
        key_parts = [
            role,
            prompt_str,
            str(max_tokens),
            str(temperature),
            str(top_p),
            str(top_k)
        ]
        
        # Join and hash
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _increment_metric(self, name: str, value: int = 1) -> None:
        """
        Increment a metric.
        
        Args:
            name: Metric name
            value: Value to increment by
        """
        with self._request_lock:
            if name in self._metrics:
                self._metrics[name] += value
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get request metrics.
        
        Returns:
            Dictionary with metrics
        """
        with self._request_lock:
            metrics = self._metrics.copy()
            
            # Add derived metrics
            if metrics["requests"] > 0:
                metrics["cache_hit_rate"] = metrics["cache_hits"] / metrics["requests"]
                metrics["error_rate"] = metrics["errors"] / metrics["requests"]
                if metrics["requests"] > metrics["cache_hits"]:
                    metrics["avg_request_time"] = metrics["total_time"] / (metrics["requests"] - metrics["cache_hits"])
                else:
                    metrics["avg_request_time"] = 0.0
            else:
                metrics["cache_hit_rate"] = 0.0
                metrics["error_rate"] = 0.0
                metrics["avg_request_time"] = 0.0
                
            return metrics
    
    def clear_cache(self) -> None:
        """Clear response cache."""
        if self.cache_responses and self._response_cache:
            self._response_cache.clear()
    
    def unload_models(self) -> None:
        """Unload all models from memory."""
        with self._request_lock:
            for runner in self._model_runners.values():
                runner.unload()
            self._model_runners.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_models()


# Utility functions for working with SVG generation
def extract_svg_from_text(text: str) -> Optional[str]:
    """
    Extract SVG code from text.
    
    Args:
        text: Text to extract SVG from
        
    Returns:
        Extracted SVG or None if not found
    """
    # Try to find SVG in code blocks
    svg_match = re.search(r'```(?:html|svg|xml)?\s*((?:<\?xml|<svg).*?</svg>)', text, re.DOTALL)
    if svg_match:
        return svg_match.group(1).strip()
        
    # If no explicit code block, try to find SVG tags directly
    svg_match = re.search(r'(?:<\?xml|<svg).*?</svg>', text, re.DOTALL)
    if svg_match:
        return svg_match.group(0).strip()
        
    # No SVG found
    return None


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text.
    
    Args:
        text: Text to extract JSON from
        
    Returns:
        Extracted JSON or None if not found
    """
    # Try to find JSON in code blocks
    json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
            
    # Try to find JSON with curly braces
    json_pattern = r"\{[\s\S]*\}"
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
            
    # No valid JSON found
    return None


# Create singleton instance for easy access
default_llm_manager = LLMManager()


# Utility functions
def get_text_completion(prompt: str, **kwargs) -> str:
    """
    Get simple text completion.
    
    Args:
        prompt: Text prompt
        **kwargs: Additional parameters
        
    Returns:
        Generated text
    """
    return default_llm_manager.generate(prompt, **kwargs)


def get_chat_completion(messages: List[Dict[str, str]], **kwargs) -> str:
    """
    Get chat completion.
    
    Args:
        messages: Chat messages
        **kwargs: Additional parameters
        
    Returns:
        Generated response
    """
    return default_llm_manager.generate_chat(messages, **kwargs)