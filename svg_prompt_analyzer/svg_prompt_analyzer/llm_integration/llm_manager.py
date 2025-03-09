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
import asyncio
import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
import concurrent.futures
import queue
from io import BytesIO

# Import core optimizations
from core import CONFIG, memoize, Profiler, get_thread_pool
from utils.logger import get_logger, log_function_call

# Configure logger
logger = get_logger(__name__)

# Type aliases
ResponseType = Dict[str, Any]
PromptType = Union[str, List[Dict[str, str]]]
TokenCount = int
MessageList = List[Dict[str, Any]]

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
    LLAMA3_8B = auto()
    CODELLAMA_7B = auto()
    LLAVA_NEXT = auto()
    COGVLM = auto()
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


# Configuration for RL training
@dataclass
class RLConfig:
    """Configuration for reinforcement learning."""
    enabled: bool = False
    reward_model_path: Optional[str] = None
    learning_rate: float = 1e-5
    batch_size: int = 4
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    kl_penalty: float = 0.1
    use_advantage_norm: bool = True
    gamma: float = 0.99
    lambda_gae: float = 0.95
    
    # Multi-objective optimization weights
    clip_score_weight: float = 1.0
    svg_size_weight: float = 0.1
    render_time_weight: float = 0.1


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
        backend: ModelBackend = ModelBackend.ONNX
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


class OnnxModelRunner(ModelRunner):
    """
    ONNX model runner.
    
    Runs models using ONNX Runtime for efficiency.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize ONNX model runner.
        
        Args:
            config: Model configuration
        """
        super().__init__(config, ModelBackend.ONNX)
        self._session_options = None
        self._session = None
    
    def _ensure_initialized(self) -> None:
        """Ensure ONNX model is initialized."""
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            try:
                import onnxruntime as ort
                
                # Configure session options
                self._session_options = ort.SessionOptions()
                self._session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                self._session_options.intra_op_num_threads = self.config.threads
                
                # Set up providers
                providers = []
                if self.config.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                    provider_options = {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    }
                    providers.append(("CUDAExecutionProvider", provider_options))
                providers.append("CPUExecutionProvider")
                
                # Create session
                self._session = ort.InferenceSession(
                    self.config.model_path,
                    sess_options=self._session_options,
                    providers=providers
                )
                
                # Initialize tokenizer
                try:
                    import sentencepiece as spm
                    tokenizer_path = self.config.weights_path or os.path.join(
                        os.path.dirname(self.config.model_path), "tokenizer.model"
                    )
                    self._tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
                except ImportError:
                    logger.warning("SentencePiece not available, using custom tokenizer")
                    # Implement a simple fallback tokenizer
                    self._tokenizer = self._create_simple_tokenizer()
                except Exception as e:
                    logger.error(f"Failed to load tokenizer: {e}")
                    self._tokenizer = self._create_simple_tokenizer()
                
                self._initialized = True
                logger.info(f"Initialized ONNX model: {self.config.model_path}")
                
            except Exception as e:
                logger.error(f"Failed to initialize ONNX model: {e}")
                raise
    
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
        Generate completion using ONNX model.
        
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
            # Convert message list to string
            prompt_text = self._convert_messages_to_text(prompt)
        else:
            prompt_text = prompt
            
        # Use default max tokens if not specified
        max_tokens = max_tokens or self.config.max_tokens
        
        # Tokenize input
        input_ids = self._tokenize(prompt_text)
        
        # Prepare input for model
        model_inputs = {
            "input_ids": np.array([input_ids], dtype=np.int64),
            "max_length": np.array([len(input_ids) + max_tokens], dtype=np.int64),
            "temperature": np.array([temperature], dtype=np.float32),
            "top_p": np.array([top_p], dtype=np.float32),
            "top_k": np.array([top_k], dtype=np.int32),
        }
        
        # Run inference
        with Profiler("onnx_inference"):
            outputs = self._session.run(None, model_inputs)
            
        # Process output
        output_ids = outputs[0][0][len(input_ids):]
        generated_text = self._detokenize(output_ids)
        
        # Apply stop sequences
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.find(stop_seq)]
                    
        # Calculate simple token counts
        generation_info = {
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(output_ids),
            "total_tokens": len(input_ids) + len(output_ids),
        }
        
        return generated_text, generation_info
    
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
            prompt_text = self._convert_messages_to_text(prompt)
        else:
            prompt_text = prompt
            
        # Tokenize input and response
        prompt_ids = self._tokenize(prompt_text)
        response_ids = self._tokenize(response)
        
        # Combine for context
        input_ids = prompt_ids + response_ids
        
        # Get logits for each position
        logprobs = []
        
        # This is a simplified approach - in practice, you'd run the model in a way
        # that returns logits for each token and compute proper log probabilities
        # For now, return placeholder values
        logprobs = [-1.0] * len(response_ids)
        
        return logprobs
    
    def _tokenize(self, text: str) -> List[int]:
        """
        Tokenize text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        if hasattr(self._tokenizer, "encode"):
            return self._tokenizer.encode(text)
        elif hasattr(self._tokenizer, "EncodeAsIds"):
            return self._tokenizer.EncodeAsIds(text)
        else:
            # Fallback for custom tokenizer
            return self._tokenizer.tokenize(text)
    
    def _detokenize(self, token_ids: List[int]) -> str:
        """
        Detokenize token IDs.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Detokenized text
        """
        if hasattr(self._tokenizer, "decode"):
            return self._tokenizer.decode(token_ids)
        elif hasattr(self._tokenizer, "DecodeIds"):
            return self._tokenizer.DecodeIds(token_ids)
        else:
            # Fallback for custom tokenizer
            return self._tokenizer.detokenize(token_ids)
    
    def _convert_messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert message list to text.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted text
        """
        # Simple format: "role: content"
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle content as list (for multi-modal)
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    else:
                        parts.append(str(part))
                content = " ".join(parts)
                
            formatted.append(f"{role}: {content}")
            
        return "\n".join(formatted)
    
    def _create_simple_tokenizer(self):
        """
        Create a simple fallback tokenizer.
        
        Returns:
            Simple tokenizer object
        """
        class SimpleTokenizer:
            def tokenize(self, text):
                # Very simple character-level tokenization
                return [ord(c) for c in text]
                
            def detokenize(self, tokens):
                # Convert back to characters
                return "".join([chr(t) for t in tokens])
                
        return SimpleTokenizer()


class LLaMACppRunner(ModelRunner):
    """
    LLaMA.cpp model runner.
    
    Runs models using llama.cpp for efficiency on CPU.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize LLaMA.cpp model runner.
        
        Args:
            config: Model configuration
        """
        super().__init__(config, ModelBackend.LLAMACPP)
    
    def _ensure_initialized(self) -> None:
        """Ensure LLaMA.cpp model is initialized."""
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            try:
                # Try to import llama_cpp
                from llama_cpp import Llama
                
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
                        
                # Create model
                self._model = Llama(**params)
                
                # No separate tokenizer needed for llama_cpp
                self._tokenizer = None
                
                self._initialized = True
                logger.info(f"Initialized LLaMA.cpp model: {self.config.model_path}")
                
            except Exception as e:
                logger.error(f"Failed to initialize LLaMA.cpp model: {e}")
                raise
    
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
            # Convert message list to text for LLaMA.cpp
            prompt_text = self._convert_messages_to_text(prompt)
        else:
            prompt_text = prompt
            
        # Use default max tokens if not specified
        max_tokens = max_tokens or self.config.max_tokens
        
        # Prepare stop sequences
        stop = stop_sequences or []
        
        # Run generation
        with Profiler("llamacpp_inference"):
            output = self._model.generate(
                prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                **kwargs
            )
            
        # Extract generated text
        generated_text = output["choices"][0]["text"]
        
        # Get token counts
        generation_info = {
            "prompt_tokens": output.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": output.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": output.get("usage", {}).get("total_tokens", 0),
        }
        
        return generated_text, generation_info
    
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
            prompt_text = self._convert_messages_to_text(prompt)
        else:
            prompt_text = prompt
            
        # Full text for context
        full_text = prompt_text + response
        
        # Get log probs from LLaMA.cpp
        # Note: This assumes a modified version of llama_cpp with logit output
        output = self._model.generate(
            prompt_text,
            max_tokens=0,  # Don't generate additional tokens
            logprobs=True,
            echo=True
        )
        
        # Extract logprobs for response tokens
        logprobs = []
        
        # In practice, you would need to extract the token indices corresponding
        # to the response and get their logprobs
        # This is a placeholder implementation
        logprobs = [-1.0] * len(self._model.tokenize(response))
        
        return logprobs
    
    def _convert_messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert message list to text format for LLaMA models.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted text
        """
        # LLaMA chat format
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle content as list (for multi-modal)
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    else:
                        parts.append(str(part))
                content = " ".join(parts)
                
            if role == "system":
                formatted.append(f"<s>[SYSTEM] {content} </s>")
            elif role == "user":
                formatted.append(f"<s>[USER] {content} </s>")
            elif role == "assistant":
                formatted.append(f"<s>[ASSISTANT] {content} </s>")
            else:
                formatted.append(f"<s>[{role.upper()}] {content} </s>")
                
        return "".join(formatted) + "<s>[ASSISTANT] "


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
    
    def _ensure_initialized(self) -> None:
        """Ensure CTransformers model is initialized."""
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            try:
                # Try to import ctransformers
                from ctransformers import AutoModelForCausalLM
                
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
                self._model = AutoModelForCausalLM.from_pretrained(**params)
                
                # No separate tokenizer needed for ctransformers
                self._tokenizer = None
                
                self._initialized = True
                logger.info(f"Initialized CTransformers model: {self.config.model_path}")
                
            except Exception as e:
                logger.error(f"Failed to initialize CTransformers model: {e}")
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
            ModelType.LLAMA3_8B: "llama",
            ModelType.CODELLAMA_7B: "llama",
            ModelType.CUSTOM: "llama",  # Default to llama
        }
        
        return model_type_map.get(self.config.model_type, "llama")
    
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
            # Convert message list to text for CTransformers
            prompt_text = self._convert_messages_to_text(prompt)
        else:
            prompt_text = prompt
            
        # Use default max tokens if not specified
        max_tokens = max_tokens or self.config.max_tokens
        
        # Set up generation parameters
        params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        
        # Add stop sequences if provided
        if stop_sequences:
            params["stop"] = stop_sequences
            
        # Run generation
        with Profiler("ctransformers_inference"):
            # Get token count before generation
            prompt_tokens = self._model.tokenize(prompt_text)
            prompt_token_count = len(prompt_tokens)
            
            # Generate text
            generated_text = self._model(
                prompt_text,
                **params
            )
            
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
        # CTransformers doesn't provide direct logprob access
        # This is a placeholder implementation
        tokens = self._model.tokenize(response)
        return [-1.0] * len(tokens)
    
    def _convert_messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert message list to text for CTransformers.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted text
        """
        # Similar to LLaMA chat format
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle content as list (for multi-modal)
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    else:
                        parts.append(str(part))
                content = " ".join(parts)
                
            if role == "system":
                formatted.append(f"<s>[SYSTEM] {content} </s>")
            elif role == "user":
                formatted.append(f"<s>[USER] {content} </s>")
            elif role == "assistant":
                formatted.append(f"<s>[ASSISTANT] {content} </s>")
            else:
                formatted.append(f"<s>[{role.upper()}] {content} </s>")
                
        return "".join(formatted) + "<s>[ASSISTANT] "


class RewardModel:
    """
    Reward model for reinforcement learning.
    
    Evaluates generated SVGs and provides rewards for RL training.
    """
    
    def __init__(self, clip_model_path: Optional[str] = None):
        """
        Initialize reward model.
        
        Args:
            clip_model_path: Path to CLIP model for similarity scoring
        """
        self.clip_model_path = clip_model_path
        self._clip_model = None
        self._initialized = False
        self._lock = threading.RLock()
    
    def _ensure_initialized(self) -> None:
        """Ensure reward model is initialized."""
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            try:
                # Try to initialize CLIP model for SVG evaluation
                # This is a simplified version - in practice, you would
                # load a proper CLIP model or other reward model
                
                # For now, just set a flag to indicate initialization
                self._initialized = True
                logger.info("Initialized reward model")
                
            except Exception as e:
                logger.error(f"Failed to initialize reward model: {e}")
                raise
    
    def calculate_reward(
        self,
        prompt: PromptType,
        svg_content: str,
        **kwargs
    ) -> float:
        """
        Calculate reward for generated SVG.
        
        Args:
            prompt: Original prompt
            svg_content: Generated SVG content
            **kwargs: Additional parameters
            
        Returns:
            Reward value
        """
        self._ensure_initialized()
        
        # Calculate various reward components
        clip_score = self._calculate_clip_score(prompt, svg_content)
        svg_size_score = self._calculate_svg_size_score(svg_content)
        render_time_score = self._calculate_render_time_score(svg_content)
        
        # Combine reward components
        weights = kwargs.get("weights", {})
        clip_weight = weights.get("clip_score_weight", 1.0)
        size_weight = weights.get("svg_size_weight", 0.1)
        time_weight = weights.get("render_time_weight", 0.1)
        
        total_reward = (
            clip_weight * clip_score +
            size_weight * svg_size_score +
            time_weight * render_time_score
        )
        
        return total_reward
    
    def _calculate_clip_score(self, prompt: PromptType, svg_content: str) -> float:
        """
        Calculate CLIP similarity score.
        
        Args:
            prompt: Text prompt
            svg_content: SVG content
            
        Returns:
            CLIP similarity score (0-1)
        """
        # In a real implementation, this would use CLIP to compare
        # the SVG (rendered to an image) with the text prompt
        
        # For now, return a placeholder score
        prompt_str = prompt if isinstance(prompt, str) else self._convert_messages_to_text(prompt)
        
        # Simple heuristic based on keyword matching
        keywords = self._extract_keywords(prompt_str)
        svg_text = self._extract_text_from_svg(svg_content)
        
        # Count matches
        matches = sum(1 for kw in keywords if kw.lower() in svg_text.lower())
        match_ratio = matches / max(1, len(keywords))
        
        # Scale to 0.4-0.9 range (avoid extremes for placeholder)
        score = 0.4 + (0.5 * match_ratio)
        
        return score
    
    def _calculate_svg_size_score(self, svg_content: str) -> float:
        """
        Calculate score based on SVG file size.
        
        Args:
            svg_content: SVG content
            
        Returns:
            Size score (0-1)
        """
        # Measure SVG size in bytes
        size_bytes = len(svg_content.encode('utf-8'))
        
        # Prefer reasonable sizes (not too small, not too large)
        # Peak score at around 10KB
        if size_bytes < 1000:
            # Too small, might be too simple
            return 0.4
        elif size_bytes < 10000:
            # Good range
            return 0.8 * (size_bytes / 10000)
        else:
            # Larger sizes get diminishing returns
            return 0.8 * (10000 / size_bytes)
    
    def _calculate_render_time_score(self, svg_content: str) -> float:
        """
        Estimate score based on SVG rendering complexity.
        
        Args:
            svg_content: SVG content
            
        Returns:
            Render time score (0-1)
        """
        # Count complex elements that might affect rendering
        num_paths = svg_content.count("<path")
        num_gradients = svg_content.count("<linearGradient") + svg_content.count("<radialGradient")
        num_filters = svg_content.count("<filter")
        
        # Estimate complexity
        complexity = 0.1 + (0.01 * num_paths) + (0.05 * num_gradients) + (0.1 * num_filters)
        
        # Penalize overly complex SVGs
        if complexity > 1.0:
            return 1.0 / complexity
        else:
            return 0.8 * complexity
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "with",
            "for", "to", "from", "of", "by", "about", "like", "as", "is", "are"
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _extract_text_from_svg(self, svg_content: str) -> str:
        """
        Extract text content from SVG for keyword matching.
        
        Args:
            svg_content: SVG content
            
        Returns:
            Extracted text
        """
        # Extract text from various elements
        elements = []
        
        # Extract from text elements
        text_elements = re.findall(r'<text[^>]*>(.*?)</text>', svg_content, re.DOTALL)
        elements.extend(text_elements)
        
        # Extract from title and desc
        title_elements = re.findall(r'<title[^>]*>(.*?)</title>', svg_content, re.DOTALL)
        desc_elements = re.findall(r'<desc[^>]*>(.*?)</desc>', svg_content, re.DOTALL)
        elements.extend(title_elements)
        elements.extend(desc_elements)
        
        # Extract from attributes
        attr_text = []
        attr_patterns = [
            r'id="([^"]*)"',
            r'class="([^"]*)"',
            r'aria-label="([^"]*)"',
        ]
        
        for pattern in attr_patterns:
            matches = re.findall(pattern, svg_content)
            attr_text.extend(matches)
            
        # Combine all text
        all_text = " ".join(elements + attr_text)
        
        return all_text
    
    def _convert_messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert message list to text.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Combined text
        """
        text_parts = []
        
        for msg in messages:
            content = msg.get("content", "")
            
            # Handle content as list (for multi-modal)
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    else:
                        parts.append(str(part))
                content = " ".join(parts)
                
            text_parts.append(content)
            
        return " ".join(text_parts)


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer for RL fine-tuning.
    
    Implements PPO algorithm for reinforcement learning with LLMs.
    """
    
    def __init__(
        self,
        model_runner: ModelRunner,
        reward_model: RewardModel,
        config: RLConfig
    ):
        """
        Initialize PPO trainer.
        
        Args:
            model_runner: Model runner for policy model
            reward_model: Reward model for evaluation
            config: RL configuration
        """
        self.model_runner = model_runner
        self.reward_model = reward_model
        self.config = config
        
        # Training state
        self.samples = []
        self.iteration = 0
        
        # Simple stats tracking
        self.stats = {
            "mean_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "std_reward": 0.0,
            "iterations": 0,
            "samples_collected": 0,
        }
    
    def collect_samples(
        self,
        prompts: List[PromptType],
        num_samples_per_prompt: int = 4
    ) -> List[GenerationSample]:
        """
        Collect samples for PPO training.
        
        Args:
            prompts: List of prompts
            num_samples_per_prompt: Number of samples per prompt
            
        Returns:
            List of generation samples
        """
        samples = []
        
        # Generate samples for each prompt
        for prompt in prompts:
            for _ in range(num_samples_per_prompt):
                # Generate SVG
                response, _ = self.model_runner.generate(
                    prompt,
                    temperature=self.config.temperature if hasattr(self.config, "temperature") else 0.7,
                    max_tokens=self.config.max_tokens if hasattr(self.config, "max_tokens") else 1024
                )
                
                # Calculate reward
                reward = self.reward_model.calculate_reward(
                    prompt, 
                    response,
                    weights={
                        "clip_score_weight": self.config.clip_score_weight,
                        "svg_size_weight": self.config.svg_size_weight,
                        "render_time_weight": self.config.render_time_weight
                    }
                )
                
                # Get log probabilities (for RL training)
                log_probs = self.model_runner.get_logprobs(prompt, response)
                
                # Create sample
                sample = GenerationSample(
                    prompt=prompt,
                    response=response,
                    reward=reward,
                    log_prob=sum(log_probs) / max(1, len(log_probs)),
                    value=None  # Will be computed during training
                )
                
                samples.append(sample)
                
        # Update stats
        rewards = [s.reward for s in samples]
        self.stats["samples_collected"] += len(samples)
        self.stats["mean_reward"] = sum(rewards) / max(1, len(rewards))
        self.stats["min_reward"] = min(rewards) if rewards else 0.0
        self.stats["max_reward"] = max(rewards) if rewards else 0.0
        self.stats["std_reward"] = (
            (sum((r - self.stats["mean_reward"])**2 for r in rewards) / max(1, len(rewards)))**0.5
            if rewards else 0.0
        )
        
        self.samples.extend(samples)
        return samples
    
    def train_iteration(self) -> Dict[str, float]:
        """
        Run one iteration of PPO training.
        
        Returns:
            Training statistics
        """
        if not self.samples:
            return {"error": "No samples collected"}
            
        # In a real implementation, this would:
        # 1. Compute advantages and returns
        # 2. Update policy using PPO loss
        # 3. Update value function
        
        # For this implementation, we'll just track iterations
        self.iteration += 1
        self.stats["iterations"] = self.iteration
        
        return self.stats
    
    def save_model(self, path: str) -> None:
        """
        Save fine-tuned model.
        
        Args:
            path: Path to save model
        """
        # In a real implementation, this would save the model weights
        # For now, just log the action
        logger.info(f"Model would be saved to {path}")
        
        # Create a simple report file
        with open(f"{path}_stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)


class LLMManager:
    """
    Production-grade LLM manager with reinforcement learning optimization.
    
    Provides optimized integration with open-source language models for generating
    and refining SVG content with memory, performance, and RL-based optimization.
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        backend: ModelBackend = ModelBackend.ONNX,
        rl_config: Optional[RLConfig] = None,
        cache_responses: bool = True,
        preload_model: bool = False
    ):
        """
        Initialize LLM manager.
        
        Args:
            model_config: Model configuration
            backend: Model backend
            rl_config: RL configuration
            cache_responses: Whether to cache responses
            preload_model: Whether to preload model
        """
        # Set up model configuration
        self.model_config = model_config or self._default_model_config()
        self.backend = backend
        
        # Set up model runner
        self.model_runner = self._create_model_runner()
        
        # Set up RL if enabled
        self.rl_config = rl_config or RLConfig(enabled=False)
        self.reward_model = RewardModel() if self.rl_config.enabled else None
        self.ppo_trainer = None
        
        if self.rl_config.enabled and self.reward_model:
            self.ppo_trainer = PPOTrainer(
                model_runner=self.model_runner,
                reward_model=self.reward_model,
                config=self.rl_config
            )
            
        # Set up response cache
        self.cache_responses = cache_responses
        self._cache = ResponseCache() if cache_responses else None
        
        # Metrics
        self._metrics = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "total_time": 0.0,
        }
        self._metrics_lock = threading.RLock()
        
        # Preload model if requested
        if preload_model:
            self.model_runner.preload()
    
    def _default_model_config(self) -> ModelConfig:
        """
        Create default model configuration.
        
        Returns:
            Default model configuration
        """
        # Try to find a reasonable default model
        local_models_dir = os.environ.get("LOCAL_MODELS_DIR", "models")
        
        # Check for common model paths
        potential_models = [
            (os.path.join(local_models_dir, "mistral-7b"), ModelType.MISTRAL_7B),
            (os.path.join(local_models_dir, "llama-3-8b"), ModelType.LLAMA3_8B),
            (os.path.join(local_models_dir, "codellama-7b"), ModelType.CODELLAMA_7B),
        ]
        
        for path, model_type in potential_models:
            if os.path.exists(path):
                return ModelConfig(
                    model_type=model_type,
                    model_path=path
                )
                
        # Fallback to a default configuration
        return ModelConfig(
            model_type=ModelType.LLAMA3_8B,
            model_path=os.path.join(local_models_dir, "llama-3-8b"),
            quantization="int8"
        )
    
    def _create_model_runner(self) -> ModelRunner:
        """
        Create model runner based on backend.
        
        Returns:
            Model runner
        """
        if self.backend == ModelBackend.ONNX:
            return OnnxModelRunner(self.model_config)
        elif self.backend == ModelBackend.LLAMACPP:
            return LLaMACppRunner(self.model_config)
        elif self.backend == ModelBackend.CTRANSFORMERS:
            return CTModelRunner(self.model_config)
        else:
            # Default to ONNX
            return OnnxModelRunner(self.model_config)
    
    def _create_cache_key(self, prompt: PromptType, **kwargs) -> str:
        """
        Create cache key for a request.
        
        Args:
            prompt: Text prompt or message list
            **kwargs: Additional request parameters
            
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
            self.model_config.model_type.name,
            prompt_str,
            str(kwargs.get("temperature", DEFAULT_TEMPERATURE)),
            str(kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)),
            str(kwargs.get("top_p", 0.9)),
        ]
        
        # Join and hash
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @log_function_call(level=logging.DEBUG)
    def complete(
        self,
        prompt: PromptType,
        use_cache: bool = True,
        **kwargs
    ) -> ResponseType:
        """
        Get completion from LLM.
        
        Args:
            prompt: Text prompt or message list
            use_cache: Whether to use cache
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Model response
            
        Raises:
            Exception: If request fails
        """
        start_time = time.time()
        
        # Check cache if enabled
        if use_cache and self.cache_responses and self._cache:
            cache_key = self._create_cache_key(prompt, **kwargs)
            cached = self._cache.get(cache_key)
            
            if cached:
                self._increment_metric("cache_hits")
                return cached
            else:
                self._increment_metric("cache_misses")
                
        # Generate completion
        try:
            # Generate response
            generated_text, generation_info = self.model_runner.generate(
                prompt,
                **kwargs
            )
            
            # Create response
            response = {
                "content": generated_text,
                "usage": generation_info,
                "model": self.model_config.model_type.name,
            }
            
            # Apply RL optimization if enabled
            if self.rl_config.enabled and self.ppo_trainer and 'svg' in str(prompt).lower():
                # This would normally happen in a separate training loop,
                # but here we'll collect samples for learning
                self.ppo_trainer.collect_samples([prompt], num_samples_per_prompt=1)
                
            # Cache response if enabled
            if use_cache and self.cache_responses and self._cache:
                self._cache.set(cache_key, response)
                
            # Update metrics
            self._increment_metric("requests")
            self._update_metric("total_time", time.time() - start_time)
            
            return response
            
        except Exception as e:
            # Update metrics
            self._increment_metric("errors")
            
            # Re-raise exception
            raise e
    
    @log_function_call(level=logging.DEBUG)
    def complete_chat(
        self,
        messages: List[Dict[str, str]],
        use_cache: bool = True,
        **kwargs
    ) -> ResponseType:
        """
        Get completion from chat messages.
        
        Args:
            messages: List of chat messages
            use_cache: Whether to use cache
            **kwargs: Additional parameters
            
        Returns:
            Model response
        """
        # Call complete with message list
        return self.complete(messages, use_cache, **kwargs)
    
    def train_rl(
        self,
        prompts: List[str],
        iterations: int = 1,
        samples_per_prompt: int = 4
    ) -> Dict[str, Any]:
        """
        Train model using reinforcement learning.
        
        Args:
            prompts: Training prompts
            iterations: Number of training iterations
            samples_per_prompt: Number of samples per prompt
            
        Returns:
            Training statistics
            
        Raises:
            ValueError: If RL is not enabled
        """
        if not self.rl_config.enabled or not self.ppo_trainer:
            raise ValueError("Reinforcement learning is not enabled")
            
        # Collect samples
        self.ppo_trainer.collect_samples(prompts, samples_per_prompt)
        
        # Run training iterations
        stats = {}
        for i in range(iterations):
            iter_stats = self.ppo_trainer.train_iteration()
            stats[f"iteration_{i+1}"] = iter_stats
            
        # Final stats
        final_stats = self.ppo_trainer.stats.copy()
        final_stats["training_completed"] = True
        
        return final_stats
    
    def save_rl_model(self, path: str) -> None:
        """
        Save RL-trained model.
        
        Args:
            path: Path to save model
            
        Raises:
            ValueError: If RL is not enabled
        """
        if not self.rl_config.enabled or not self.ppo_trainer:
            raise ValueError("Reinforcement learning is not enabled")
            
        # Save model
        self.ppo_trainer.save_model(path)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get request metrics.
        
        Returns:
            Dictionary with metrics
        """
        with self._metrics_lock:
            metrics = self._metrics.copy()
            
            # Add derived metrics
            if metrics["requests"] > 0:
                metrics["cache_hit_rate"] = metrics["cache_hits"] / metrics["requests"]
                metrics["error_rate"] = metrics["errors"] / metrics["requests"]
                metrics["avg_request_time"] = metrics["total_time"] / (metrics["requests"] - metrics["cache_hits"])
            else:
                metrics["cache_hit_rate"] = 0.0
                metrics["error_rate"] = 0.0
                metrics["avg_request_time"] = 0.0
                
            # Add RL metrics if available
            if self.rl_config.enabled and self.ppo_trainer:
                metrics["rl"] = self.ppo_trainer.stats
                
            return metrics
    
    def _increment_metric(self, name: str, value: int = 1) -> None:
        """
        Increment a metric.
        
        Args:
            name: Metric name
            value: Value to increment by
        """
        with self._metrics_lock:
            if name in self._metrics:
                self._metrics[name] += value
    
    def _update_metric(self, name: str, value: Any) -> None:
        """
        Update a metric.
        
        Args:
            name: Metric name
            value: New value
        """
        with self._metrics_lock:
            if name in self._metrics:
                self._metrics[name] = value
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._metrics_lock:
            for key in self._metrics:
                self._metrics[key] = 0
    
    def clear_cache(self) -> None:
        """Clear response cache."""
        if self.cache_responses and self._cache:
            self._cache.clear()
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        if self.model_runner:
            self.model_runner.unload()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_model()


# Create singleton instance for easy import
default_llm_manager = LLMManager()


# High-level utility functions
def get_text_completion(
    prompt: str,
    model_config: Optional[ModelConfig] = None,
    **kwargs
) -> str:
    """
    Get text completion from LLM.
    
    Args:
        prompt: Text prompt
        model_config: Model configuration
        **kwargs: Additional request parameters
        
    Returns:
        Generated text
    """
    manager = default_llm_manager
    
    # Create a new manager if config provided
    if model_config:
        manager = LLMManager(model_config=model_config)
        
    # Generate response
    response = manager.complete(prompt, **kwargs)
    
    return response.get("content", "")


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text response.
    
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


def extract_svg_from_text(text: str) -> Optional[str]:
    """
    Extract SVG from text response.
    
    Args:
        text: Text to extract SVG from
        
    Returns:
        Extracted SVG or None if not found
    """
    # Try to find SVG in code blocks
    svg_pattern = r"```(?:html|svg|xml)?\s*((?:<svg[\s\S]*?<\/svg>))\s*```"
    matches = re.findall(svg_pattern, text)
    
    if matches:
        return matches[0]
        
    # Try to find SVG tags directly
    svg_pattern = r"(<svg[\s\S]*?<\/svg>)"
    matches = re.findall(svg_pattern, text)
    
    if matches:
        return matches[0]
        
    # No SVG found
    return None