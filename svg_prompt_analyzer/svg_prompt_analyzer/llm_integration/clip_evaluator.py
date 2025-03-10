"""
Production-grade CLIP-based evaluator for image-text matching.
Provides optimized implementation for evaluating the quality of SVG images
against text prompts using CLIP embeddings with memory and performance 
optimizations.
"""

import os
import sys
import time
import tempfile
import threading
import queue
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from io import BytesIO
import logging
import hashlib
import base64
import numpy as np
import re

# Import core optimizations
from svg_prompt_analyzer.core import CONFIG, memoize, Profiler, get_thread_pool
from svg_prompt_analyzer.core.memory_manager import MemoryManager
from svg_prompt_analyzer.utils.logger import get_logger, log_function_call

# Configure logger
logger = get_logger(__name__)

# Type aliases
ClipScore = float
ImageEmbedding = np.ndarray
TextEmbedding = np.ndarray
BatchItem = List[Any]

# Constants
DEFAULT_IMAGE_SIZE = 224  # Default CLIP input size
DEFAULT_BATCH_SIZE = 32
CACHE_SIZE = 2048  # Number of embeddings to cache
WORKER_TIMEOUT = 0.1  # Seconds to wait for worker threads
DEFAULT_TOP_K = 5


class FeatureCache:
    """Thread-safe cache for CLIP embeddings and features."""
    
    def __init__(self, max_size: int = CACHE_SIZE):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached items
        """
        self.max_size = max_size
        self._text_cache: Dict[str, TextEmbedding] = {}
        self._image_cache: Dict[str, ImageEmbedding] = {}
        self._svg_cache: Dict[str, ImageEmbedding] = {}
        self._access_count: Dict[str, int] = {}
        self._lock = threading.RLock()
    
    def get_text_embedding(self, text: str) -> Optional[TextEmbedding]:
        """
        Get cached text embedding.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not cached
        """
        with self._lock:
            text_key = self._get_text_key(text)
            if text_key in self._text_cache:
                self._access_count[text_key] = self._access_count.get(text_key, 0) + 1
                return self._text_cache[text_key]
            return None
    
    def set_text_embedding(self, text: str, embedding: TextEmbedding) -> None:
        """
        Cache text embedding.
        
        Args:
            text: Text to cache embedding for
            embedding: Embedding to cache
        """
        with self._lock:
            text_key = self._get_text_key(text)
            self._text_cache[text_key] = embedding
            self._access_count[text_key] = 1
            self._prune_cache_if_needed()
    
    def get_image_embedding(self, image) -> Optional[ImageEmbedding]:
        """
        Get cached image embedding.
        
        Args:
            image: Image to get embedding for
            
        Returns:
            Cached embedding or None if not cached
        """
        with self._lock:
            image_key = self._get_image_key(image)
            if image_key in self._image_cache:
                self._access_count[image_key] = self._access_count.get(image_key, 0) + 1
                return self._image_cache[image_key]
            return None
    
    def set_image_embedding(self, image, embedding: ImageEmbedding) -> None:
        """
        Cache image embedding.
        
        Args:
            image: Image to cache embedding for
            embedding: Embedding to cache
        """
        with self._lock:
            image_key = self._get_image_key(image)
            self._image_cache[image_key] = embedding
            self._access_count[image_key] = 1
            self._prune_cache_if_needed()
    
    def get_svg_embedding(self, svg_content: str) -> Optional[ImageEmbedding]:
        """
        Get cached SVG embedding.
        
        Args:
            svg_content: SVG content to get embedding for
            
        Returns:
            Cached embedding or None if not cached
        """
        with self._lock:
            svg_key = self._get_svg_key(svg_content)
            if svg_key in self._svg_cache:
                self._access_count[svg_key] = self._access_count.get(svg_key, 0) + 1
                return self._svg_cache[svg_key]
            return None
    
    def set_svg_embedding(self, svg_content: str, embedding: ImageEmbedding) -> None:
        """
        Cache SVG embedding.
        
        Args:
            svg_content: SVG content to cache embedding for
            embedding: Embedding to cache
        """
        with self._lock:
            svg_key = self._get_svg_key(svg_content)
            self._svg_cache[svg_key] = embedding
            self._access_count[svg_key] = 1
            self._prune_cache_if_needed()
    
    def clear(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._text_cache.clear()
            self._image_cache.clear()
            self._svg_cache.clear()
            self._access_count.clear()
    
    def _get_text_key(self, text: str) -> str:
        """
        Generate cache key for text.
        
        Args:
            text: Text to generate key for
            
        Returns:
            Cache key
        """
        # Use MD5 for fast hashing
        return f"text_{hashlib.md5(text.encode('utf-8')).hexdigest()}"
    
    def _get_image_key(self, image) -> str:
        """
        Generate cache key for image.
        
        Args:
            image: Image to generate key for
            
        Returns:
            Cache key
        """
        # Convert image to bytes for hashing
        try:
            if hasattr(image, 'tobytes'):
                # NumPy array
                image_data = image.tobytes()
            elif hasattr(image, 'save'):
                # PIL Image
                image_bytes = BytesIO()
                image.save(image_bytes, format='PNG')
                image_data = image_bytes.getvalue()
            else:
                # Fallback - convert to string representation
                image_data = str(image).encode('utf-8')
        except Exception:
            # Last resort fallback
            image_data = str(id(image)).encode('utf-8')
            
        return f"image_{hashlib.md5(image_data).hexdigest()}"
    
    def _get_svg_key(self, svg_content: str) -> str:
        """
        Generate cache key for SVG content.
        
        Args:
            svg_content: SVG content to generate key for
            
        Returns:
            Cache key
        """
        return f"svg_{hashlib.md5(svg_content.encode('utf-8')).hexdigest()}"
    
    def _prune_cache_if_needed(self) -> None:
        """Prune cache if it exceeds maximum size."""
        # Check total cache size
        total_size = len(self._text_cache) + len(self._image_cache) + len(self._svg_cache)
        
        if total_size > self.max_size:
            # Get all keys with access counts
            all_keys = list(self._access_count.keys())
            all_counts = [self._access_count[k] for k in all_keys]
            
            # Sort keys by access count (ascending)
            sorted_keys = [k for _, k in sorted(zip(all_counts, all_keys))]
            
            # Calculate how many to remove
            to_remove = total_size - self.max_size + (self.max_size // 10)  # Remove extra 10% to avoid frequent pruning
            
            # Remove least accessed items
            for key in sorted_keys[:to_remove]:
                if key in self._text_cache:
                    del self._text_cache[key]
                if key in self._image_cache:
                    del self._image_cache[key]
                if key in self._svg_cache:
                    del self._svg_cache[key]
                del self._access_count[key]


class ClipModelLoader:
    """
    Handles lazy loading and initialization of CLIP models.
    
    This allows flexible model loading based on availability and system resources.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for resource efficiency."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ClipModelLoader, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "auto",
                 cache_dir: str = ".cache/clip",
                 use_fp16: bool = None):
        """
        Initialize the CLIP model loader.
        
        Args:
            model_name: Name of CLIP model
            device: Device to use (auto, cuda, cpu)
            cache_dir: Model cache directory
            use_fp16: Whether to use FP16 precision
        """
        # Initialize only once (singleton pattern)
        if hasattr(self, 'initialized'):
            return
            
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Determine device
        self.device = device
        if self.device == "auto":
            self.device = self._get_optimal_device()
            
        # Determine FP16 usage
        self.use_fp16 = use_fp16
        if self.use_fp16 is None:
            self.use_fp16 = (self.device == "cuda")
            
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Track initialization state for lazy loading
        self.is_onnx = "onnx" in model_name.lower()
        self.model_loaded = False
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        # Initialize memory manager
        self.memory_manager = MemoryManager()
        
        self.initialized = True
        logger.info(f"CLIP model loader initialized for {model_name} on {self.device}")
    
    def _get_optimal_device(self) -> str:
        """
        Determine optimal device for CLIP model.
        
        Returns:
            Device string (cuda, cpu)
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, "mps") and torch.mps.is_available():
                return "mps"
        except ImportError:
            pass
            
        return "cpu"
    
    @log_function_call(level=logging.DEBUG)
    def load_model(self) -> bool:
        """
        Load CLIP model into memory.
        
        Returns:
            Whether model was loaded successfully
        """
        if self.model_loaded:
            return True
            
        with Profiler("clip_model_loading"), self._lock:
            if self.model_loaded:
                return True
                
            try:
                logger.info(f"Loading CLIP model: {self.model_name} on {self.device}")
                
                # Check available memory
                memory_info = self.memory_manager.get_memory_stats()
                available_memory = memory_info.get("available_memory_gb", 0)
                
                if available_memory < 2.0 and self.device != "cpu":
                    logger.warning(f"Low memory available ({available_memory:.2f} GB), switching to CPU")
                    self.device = "cpu"
                
                # Determine model loading approach
                if self.is_onnx:
                    success = self._load_onnx_model()
                else:
                    success = self._load_transformers_model()
                    
                return success
                
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {str(e)}")
                return False
    
    def _load_transformers_model(self) -> bool:
        """
        Load CLIP model using transformers library.
        
        Returns:
            Whether model was loaded successfully
        """
        try:
            # First attempt with transformers if available
            import torch
            from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
            
            # Load model and processor
            self.model = CLIPModel.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.use_fp16 else torch.float32
            )
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Move model to device
            self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            self.model_loaded = True
            logger.info(f"Successfully loaded CLIP model via transformers")
            return True
            
        except ImportError:
            logger.warning("Transformers library not available, trying alternative methods")
            return self._load_alternative_model()
        except Exception as e:
            logger.error(f"Error loading model with transformers: {str(e)}")
            return self._load_alternative_model()
    
    def _load_onnx_model(self) -> bool:
        """
        Load CLIP model using ONNX runtime.
        
        Returns:
            Whether model was loaded successfully
        """
        try:
            import onnxruntime as ort
            
            # Set up ONNX runtime
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Set providers based on device
            providers = []
            if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            
            # Load ONNX model
            model_path = os.path.join(self.cache_dir, f"{os.path.basename(self.model_name)}.onnx")
            if not os.path.exists(model_path):
                logger.error(f"ONNX model not found at {model_path}")
                return False
                
            self.model = ort.InferenceSession(
                model_path,
                providers=providers,
                sess_options=session_options
            )
            
            # Load tokenizer (simplified implementation)
            self.tokenizer = SimpleTokenizer()
            
            self.model_loaded = True
            logger.info(f"Successfully loaded CLIP model via ONNX runtime")
            return True
            
        except ImportError:
            logger.error("ONNX runtime not available")
            return False
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            return False
    
    def _load_alternative_model(self) -> bool:
        """
        Attempt to load model using alternative methods.
        
        Returns:
            Whether model was loaded successfully
        """
        try:
            # Try with simple CLIP implementation if available
            import torch
            import clip
            
            # Map model name to CLIP model name
            clip_model_name = "ViT-B/32"  # Default
            if "vit-base" in self.model_name.lower():
                clip_model_name = "ViT-B/32"
            elif "vit-large" in self.model_name.lower():
                clip_model_name = "ViT-L/14"
            elif "resnet" in self.model_name.lower():
                clip_model_name = "RN50"
                
            # Load model
            self.model, self.processor = clip.load(
                clip_model_name,
                device=self.device,
                jit=False
            )
            
            # Create tokenizer adapter
            class TokenizerAdapter:
                def __init__(self, clip_model):
                    self.model = clip_model
                    
                def __call__(self, texts, **kwargs):
                    return clip.tokenize(texts)
                    
            self.tokenizer = TokenizerAdapter(self.model)
            
            # Set to evaluation mode
            self.model.eval()
            
            self.model_loaded = True
            logger.info(f"Successfully loaded CLIP model via alternate method")
            return True
            
        except ImportError:
            logger.error("Alternative CLIP implementations not available")
            return False
        except Exception as e:
            logger.error(f"Error loading model with alternative method: {str(e)}")
            return False


class SimpleTokenizer:
    """Simple tokenizer implementation for when full tokenizers are not available."""
    
    def __init__(self):
        """Initialize simple tokenizer."""
        # Basic vocabulary (would be expanded in real implementation)
        self.vocab = {word: i for i, word in enumerate(["<start>", "<end>", "<pad>"] + 
                                                     list("abcdefghijklmnopqrstuvwxyz"))}
        self.max_length = 77  # CLIP's standard context length
        
    def __call__(self, texts, **kwargs):
        """
        Tokenize texts.
        
        Args:
            texts: List of texts to tokenize
            **kwargs: Additional arguments
            
        Returns:
            Tokenized texts
        """
        if isinstance(texts, str):
            texts = [texts]
            
        batch_tokens = []
        for text in texts:
            # Simple character-based tokenization
            tokens = [self.vocab.get(c.lower(), len(self.vocab)) for c in text if c.strip()]
            
            # Add start token
            tokens = [0] + tokens
            
            # Pad or truncate
            if len(tokens) < self.max_length:
                tokens = tokens + [2] * (self.max_length - len(tokens))
            else:
                tokens = tokens[:self.max_length-1] + [1]  # Add end token
                
            batch_tokens.append(tokens)
            
        # Convert to appropriate format
        try:
            import torch
            return torch.tensor(batch_tokens)
        except ImportError:
            import numpy as np
            return np.array(batch_tokens)


class CLIPEvaluator:
    """
    Production-grade CLIP-based evaluator for image-text matching.
    
    Uses CLIP embeddings to evaluate how well images match text prompts.
    Includes optimizations for memory usage, batch processing, and caching.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        cache_dir: str = ".cache/clip",
        batch_size: int = None,
        cache_size: int = CACHE_SIZE,
        use_fp16: bool = None,
        thread_workers: int = None
    ):
        """
        Initialize CLIP evaluator.
        
        Args:
            model_name: CLIP model name
            device: Device to run on (None for auto-detection)
            cache_dir: Directory for model caching
            batch_size: Batch size for processing
            cache_size: Size of embedding cache
            use_fp16: Whether to use FP16 precision
            thread_workers: Number of worker threads
        """
        # Set parameters
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.use_fp16 = use_fp16
        
        # Initialize model loader
        self.model_loader = ClipModelLoader(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            use_fp16=use_fp16
        )
        
        # Property aliases for backward compatibility
        self.model_loaded = False
        
        # Set batch size based on available memory
        if batch_size is None:
            self.batch_size = self._calculate_optimal_batch_size()
        else:
            self.batch_size = batch_size
            
        # Initialize cache
        self.cache = FeatureCache(max_size=cache_size)
        
        # Set up thread workers
        self.thread_workers = thread_workers or CONFIG.get("thread_pool_size", min(32, os.cpu_count() or 4))
        
        # Set up memory manager
        self.memory_manager = MemoryManager()
        
        # Threading resources
        self._thread_pool = None
        
        logger.info(f"CLIP evaluator initialized with model {model_name}, batch size {self.batch_size}")
    
    def _calculate_optimal_batch_size(self) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Returns:
            Optimal batch size
        """
        if self.device == "cuda":
            try:
                # Get available GPU memory
                import torch
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                # Very conservative estimate - 100MB per item in batch
                return max(1, min(DEFAULT_BATCH_SIZE, int(free_memory / (100 * 1024 * 1024))))
            except:
                pass
                
        # Default values based on device
        if self.device == "cuda":
            return DEFAULT_BATCH_SIZE
        elif self.device == "mps":
            return 16  # Conservative for Apple Silicon
        else:
            return 8  # Conservative for CPU
    
    def _get_thread_pool(self):
        """Get thread pool for parallel processing."""
        if self._thread_pool is None:
            self._thread_pool = get_thread_pool()
        return self._thread_pool
    
    @property
    def model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model_loader.model_loaded
    
    @model_loaded.setter
    def model_loaded(self, value: bool):
        """Set model loaded flag (compatibility with old code)."""
        pass  # Ignore, we use model_loader.model_loaded instead
    
    def load_model(self) -> bool:
        """
        Load CLIP model.
        
        Returns:
            Whether model was successfully loaded
        """
        return self.model_loader.load_model()
    
    @log_function_call(level=logging.DEBUG)
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of text embeddings
        """
        if not self.load_model():
            raise RuntimeError("Failed to load CLIP model")
            
        with Profiler("encode_texts"):
            # Check cache for each text
            cached_embeddings = []
            texts_to_encode = []
            text_indices = []
            
            for i, text in enumerate(texts):
                cached = self.cache.get_text_embedding(text)
                if cached is not None:
                    cached_embeddings.append((i, cached))
                else:
                    texts_to_encode.append(text)
                    text_indices.append(i)
                    
            # Create result array
            result = np.zeros((len(texts), self._get_embedding_dim()), dtype=np.float32)
            
            # Add cached embeddings
            for i, embedding in cached_embeddings:
                result[i] = embedding
                
            # Process texts in batches if needed
            if texts_to_encode:
                text_batches = self._create_batches(texts_to_encode, self.batch_size)
                
                # Process batches
                batch_results = []
                for batch in text_batches:
                    batch_result = self._encode_text_batch(batch)
                    batch_results.append(batch_result)
                    
                # Combine batches
                all_embeddings = np.vstack(batch_results)
                
                # Add to result
                for i, embedding_idx in enumerate(text_indices):
                    result[embedding_idx] = all_embeddings[i]
                    
                    # Cache embedding
                    self.cache.set_text_embedding(texts_to_encode[i], all_embeddings[i])
                    
            return result
    
    @log_function_call(level=logging.DEBUG)
    def encode_images(self, images: List[Any]) -> np.ndarray:
        """
        Encode images into embeddings.
        
        Args:
            images: List of images to encode
            
        Returns:
            Array of image embeddings
        """
        if not self.load_model():
            raise RuntimeError("Failed to load CLIP model")
            
        with Profiler("encode_images"):
            # Check cache for each image
            cached_embeddings = []
            images_to_encode = []
            image_indices = []
            
            for i, image in enumerate(images):
                cached = self.cache.get_image_embedding(image)
                if cached is not None:
                    cached_embeddings.append((i, cached))
                else:
                    images_to_encode.append(image)
                    image_indices.append(i)
                    
            # Create result array
            result = np.zeros((len(images), self._get_embedding_dim()), dtype=np.float32)
            
            # Add cached embeddings
            for i, embedding in cached_embeddings:
                result[i] = embedding
                
            # Process images in batches if needed
            if images_to_encode:
                image_batches = self._create_batches(images_to_encode, self.batch_size)
                
                # Process batches
                batch_results = []
                for batch in image_batches:
                    batch_result = self._encode_image_batch(batch)
                    batch_results.append(batch_result)
                    
                # Combine batches
                all_embeddings = np.vstack(batch_results)
                
                # Add to result
                for i, embedding_idx in enumerate(image_indices):
                    result[embedding_idx] = all_embeddings[i]
                    
                    # Cache embedding
                    self.cache.set_image_embedding(images_to_encode[i], all_embeddings[i])
                    
            return result
    
    @log_function_call(level=logging.DEBUG)
    def encode_svg(self, svg_content: str, size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
        """
        Encode SVG content into an embedding.
        
        Args:
            svg_content: SVG content to encode
            size: Size of rasterized image
            
        Returns:
            Embedding for the SVG
        """
        if not self.load_model():
            raise RuntimeError("Failed to load CLIP model")
            
        with Profiler("encode_svg"):
            # Check cache
            cached = self.cache.get_svg_embedding(svg_content)
            if cached is not None:
                return cached
                
            # Convert SVG to image
            try:
                # Convert SVG to PNG
                png_data = self._svg_to_png(svg_content, size)
                
                # Convert PNG to image
                from PIL import Image
                image = Image.open(BytesIO(png_data))
                
                # Encode image
                embedding = self.encode_images([image])[0]
                
                # Cache embedding
                self.cache.set_svg_embedding(svg_content, embedding)
                
                return embedding
                
            except Exception as e:
                logger.error(f"Error encoding SVG: {str(e)}")
                
                # Return zero embedding on error
                embedding = np.zeros((self._get_embedding_dim(),), dtype=np.float32)
                return embedding
    
    @log_function_call(level=logging.DEBUG)
    def encode_svgs(self, svg_contents: List[str], size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
        """
        Encode multiple SVG contents into embeddings.
        
        Args:
            svg_contents: List of SVG contents to encode
            size: Size of rasterized images
            
        Returns:
            Array of SVG embeddings
        """
        if not self.load_model():
            raise RuntimeError("Failed to load CLIP model")
            
        with Profiler("encode_svgs"):
            # Check cache for each SVG
            cached_embeddings = []
            svgs_to_encode = []
            svg_indices = []
            
            for i, svg in enumerate(svg_contents):
                cached = self.cache.get_svg_embedding(svg)
                if cached is not None:
                    cached_embeddings.append((i, cached))
                else:
                    svgs_to_encode.append(svg)
                    svg_indices.append(i)
                    
            # Create result array
            result = np.zeros((len(svg_contents), self._get_embedding_dim()), dtype=np.float32)
            
            # Add cached embeddings
            for i, embedding in cached_embeddings:
                result[i] = embedding
                
            # Convert SVGs to images in parallel
            if svgs_to_encode:
                thread_pool = self._get_thread_pool()
                futures = []
                
                for svg in svgs_to_encode:
                    futures.append(thread_pool.submit(self._svg_to_image, svg, size))
                    
                # Collect results
                images = []
                for future in futures:
                    try:
                        image = future.result()
                        images.append(image)
                    except Exception as e:
                        logger.error(f"Error converting SVG to image: {str(e)}")
                        # Use a blank image as fallback
                        from PIL import Image
                        images.append(Image.new('RGB', (size, size), color='white'))
                        
                # Encode images
                image_embeddings = self.encode_images(images)
                
                # Add to result
                for i, embedding_idx in enumerate(svg_indices):
                    result[embedding_idx] = image_embeddings[i]
                    
                    # Cache embedding
                    self.cache.set_svg_embedding(svgs_to_encode[i], image_embeddings[i])
                    
            return result
    
    @log_function_call(level=logging.DEBUG)
    def compute_similarity(self, svg_content: str, prompt: str) -> float:
        """
        Compute similarity between SVG and prompt.
        
        Args:
            svg_content: SVG content
            prompt: Text prompt
            
        Returns:
            Similarity score (0-1)
        """
        # For a single pair, we can optimize by only encoding once
        with Profiler("compute_similarity"):
            try:
                # Check if model is loaded
                if not self.load_model():
                    raise RuntimeError("Failed to load CLIP model")
                    
                # Encode SVG
                svg_embedding = self.encode_svg(svg_content)
                    
                # Encode prompt
                text_embedding = self.encode_texts([prompt])[0]
                    
                # Compute similarity
                similarity = self._compute_similarity(text_embedding, svg_embedding)
                    
                return float(similarity)
                    
            except Exception as e:
                logger.error(f"Error computing similarity: {str(e)}")
                return 0.0
    
    @log_function_call(level=logging.DEBUG)
    def evaluate_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Evaluate similarity for multiple SVG-prompt pairs.
        
        Args:
            pairs: List of (svg_content, prompt) tuples
            
        Returns:
            List of similarity scores
        """
        with Profiler("evaluate_batch"):
            try:
                # Check if model is loaded
                if not self.load_model():
                    raise RuntimeError("Failed to load CLIP model")
                    
                # Separate SVGs and prompts
                svg_contents = [svg for svg, _ in pairs]
                prompts = [prompt for _, prompt in pairs]
                
                # Encode in batches
                svg_embeddings = self.encode_svgs(svg_contents)
                text_embeddings = self.encode_texts(prompts)
                
                # Calculate similarities
                similarities = []
                for i in range(len(pairs)):
                    similarity = self._compute_similarity(text_embeddings[i], svg_embeddings[i])
                    similarities.append(float(similarity))
                    
                return similarities
                
            except Exception as e:
                logger.error(f"Error in batch evaluation: {str(e)}")
                # Return zeros on error
                return [0.0] * len(pairs)
    
    @log_function_call(level=logging.DEBUG)
    def rank_svgs(
        self,
        query: str,
        svg_contents: List[str],
        top_k: int = DEFAULT_TOP_K
    ) -> List[Tuple[int, ClipScore]]:
        """
        Rank SVGs by similarity to query.
        
        Args:
            query: Text query
            svg_contents: List of SVG contents to rank
            top_k: Number of top results to return
            
        Returns:
            List of (svg_index, similarity_score) tuples
        """
        with Profiler("rank_svgs"):
            try:
                # Check if model is loaded
                if not self.load_model():
                    raise RuntimeError("Failed to load CLIP model")
                    
                # Encode query and SVGs
                text_embedding = self.encode_texts([query])[0]
                svg_embeddings = self.encode_svgs(svg_contents)
                
                # Calculate similarities
                similarities = np.zeros(len(svg_contents))
                for i, svg_embedding in enumerate(svg_embeddings):
                    similarities[i] = self._compute_similarity(text_embedding, svg_embedding)
                    
                # Get top indices
                top_indices = np.argsort(-similarities)[:top_k]
                
                # Create result
                result = [(int(idx), float(similarities[idx])) for idx in top_indices]
                return result
                
            except Exception as e:
                logger.error(f"Error ranking SVGs: {str(e)}")
                return [(0, 0.0)]
    
    def _encode_text_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts.
        
        Args:
            texts: Batch of texts to encode
            
        Returns:
            Batch text embeddings
        """
        # Memory monitoring context
        with self.memory_manager.memory_tracking_context("text_encoding"):
            try:
                model = self.model_loader.model
                tokenizer = self.model_loader.tokenizer
                
                # Handle different model types
                if hasattr(model, "get_text_features"):
                    # Transformers model
                    import torch
                    
                    with torch.no_grad():
                        # Tokenize texts
                        inputs = tokenizer(
                            texts,
                            padding=True,
                            truncation=True,
                            return_tensors="pt"
                        ).to(self.model_loader.device)
                        
                        # Get text embeddings
                        text_features = model.get_text_features(**inputs)
                        
                        # Normalize embeddings
                        text_embeddings = text_features / text_features.norm(dim=1, keepdim=True)
                        
                        # Convert to numpy
                        return text_embeddings.cpu().numpy()
                        
                elif hasattr(model, "encode_text"):
                    # CLIP library model
                    import torch
                    
                    with torch.no_grad():
                        # Tokenize and encode
                        tokens = tokenizer(texts).to(self.model_loader.device)
                        text_features = model.encode_text(tokens)
                        
                        # Normalize
                        text_embeddings = text_features / text_features.norm(dim=1, keepdim=True)
                        
                        # Convert to numpy
                        return text_embeddings.cpu().numpy()
                        
                elif hasattr(model, "run"):
                    # ONNX model
                    # Tokenize texts (simplified)
                    tokens = np.array([tokenizer(text) for text in texts])
                    
                    # Run ONNX model
                    outputs = model.run(
                        ["text_features"],
                        {"input_ids": tokens}
                    )
                    
                    # Normalize
                    text_features = outputs[0]
                    norms = np.linalg.norm(text_features, axis=1, keepdims=True)
                    text_embeddings = text_features / np.maximum(norms, 1e-8)
                    
                    return text_embeddings
                    
                else:
                    # Unsupported model
                    raise ValueError(f"Unsupported model type: {type(model)}")
                    
            except Exception as e:
                logger.error(f"Error encoding text batch: {str(e)}")
                # Return zero embeddings on error
                return np.zeros((len(texts), self._get_embedding_dim()), dtype=np.float32)
    
    def _encode_image_batch(self, images: List[Any]) -> np.ndarray:
        """
        Encode a batch of images.
        
        Args:
            images: Batch of images to encode
            
        Returns:
            Batch image embeddings
        """
        # Memory monitoring context
        with self.memory_manager.memory_tracking_context("image_encoding"):
            try:
                model = self.model_loader.model
                processor = self.model_loader.processor
                
                # Handle different model types
                if hasattr(model, "get_image_features"):
                    # Transformers model
                    import torch
                    
                    with torch.no_grad():
                        # Process images
                        inputs = processor(
                            images=images,
                            return_tensors="pt"
                        ).to(self.model_loader.device)
                        
                        # Get image embeddings
                        image_features = model.get_image_features(**inputs)
                        
                        # Normalize embeddings
                        image_embeddings = image_features / image_features.norm(dim=1, keepdim=True)
                        
                        # Convert to numpy
                        return image_embeddings.cpu().numpy()
                        
                elif hasattr(model, "encode_image"):
                    # CLIP library model
                    import torch
                    
                    with torch.no_grad():
                        # Preprocess and encode
                        preprocessed = torch.stack([processor(img) for img in images]).to(self.model_loader.device)
                        image_features = model.encode_image(preprocessed)
                        
                        # Normalize
                        image_embeddings = image_features / image_features.norm(dim=1, keepdim=True)
                        
                        # Convert to numpy
                        return image_embeddings.cpu().numpy()
                        
                elif hasattr(model, "run"):
                    # ONNX model
                    # Preprocess images (simplified)
                    from PIL import Image
                    import numpy as np
                    
                    preprocessed = []
                    for img in images:
                        if not isinstance(img, np.ndarray):
                            # Convert PIL Image to numpy array
                            img_array = np.array(img.resize((224, 224)))
                            if img_array.shape[2] == 4:  # RGBA
                                img_array = img_array[:, :, :3]  # Drop alpha
                        else:
                            img_array = img
                            
                        # Normalize
                        img_array = img_array / 255.0
                        img_array = (img_array - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
                        preprocessed.append(img_array.transpose(2, 0, 1).astype(np.float32))
                        
                    input_batch = np.stack(preprocessed)
                    
                    # Run ONNX model
                    outputs = model.run(
                        ["image_features"],
                        {"pixel_values": input_batch}
                    )
                    
                    # Normalize
                    image_features = outputs[0]
                    norms = np.linalg.norm(image_features, axis=1, keepdims=True)
                    image_embeddings = image_features / np.maximum(norms, 1e-8)
                    
                    return image_embeddings
                    
                else:
                    # Unsupported model
                    raise ValueError(f"Unsupported model type: {type(model)}")
                    
            except Exception as e:
                logger.error(f"Error encoding image batch: {str(e)}")
                # Return zero embeddings on error
                return np.zeros((len(images), self._get_embedding_dim()), dtype=np.float32)
    
    def _svg_to_image(self, svg_content: str, size: int) -> Any:
        """
        Convert SVG to PIL Image.
        
        Args:
            svg_content: SVG content
            size: Image size
            
        Returns:
            PIL Image
        """
        # Convert SVG to PNG
        png_data = self._svg_to_png(svg_content, size)
        
        # Convert PNG to image
        from PIL import Image
        return Image.open(BytesIO(png_data))
    
    def _svg_to_png(self, svg_content: str, size: int) -> bytes:
        """
        Convert SVG content to PNG data.
        
        Args:
            svg_content: SVG content to convert
            size: Size of output image
            
        Returns:
            PNG image data
        """
        with Profiler("svg_to_png"):
            try:
                # Try to use cairosvg for conversion
                try:
                    import cairosvg
                    return cairosvg.svg2png(
                        bytestring=svg_content.encode('utf-8'),
                        output_width=size,
                        output_height=size
                    )
                except ImportError:
                    logger.warning("cairosvg not available, trying alternative methods")
                    
                # Try other methods
                try:
                    from svglib.svglib import svg2rlg
                    from reportlab.graphics import renderPM
                    from io import StringIO
                    
                    drawing = svg2rlg(StringIO(svg_content))
                    png_data = BytesIO()
                    renderPM.drawToFile(drawing, png_data, fmt="PNG")
                    return png_data.getvalue()
                except ImportError:
                    logger.warning("svglib not available, falling back to basic method")
                    
                # Last resort - use PIL to create a simple rendering
                from PIL import Image, ImageDraw, ImageFont
                
                # Create a fallback image with basic info
                img = Image.new('RGB', (size, size), color='white')
                draw = ImageDraw.Draw(img)
                
                # Try to extract basic shapes as text
                shape_count = len(re.findall(r'<(rect|circle|path|line|polygon|ellipse)', svg_content))
                draw.text((10, 10), f"SVG: {shape_count} shapes", fill='black')
                
                # Save to bytes
                buf = BytesIO()
                img.save(buf, format='PNG')
                return buf.getvalue()
                
            except Exception as e:
                logger.error(f"Error converting SVG to PNG: {str(e)}")
                
                # Create a minimal fallback image on error
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (size, size), color='white')
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), "SVG Error", fill='red')
                
                # Save to bytes
                buf = BytesIO()
                img.save(buf, format='PNG')
                return buf.getvalue()
    
    def _compute_similarity(self, text_embedding: np.ndarray, image_embedding: np.ndarray) -> float:
        """
        Compute similarity between text and image embeddings.
        
        Args:
            text_embedding: Text embedding
            image_embedding: Image embedding
            
        Returns:
            Similarity score (0-1)
        """
        with Profiler("compute_similarity"):
            # Ensure embeddings are normalized
            text_norm = np.linalg.norm(text_embedding)
            image_norm = np.linalg.norm(image_embedding)
            
            if text_norm == 0 or image_norm == 0:
                return 0.0
                
            text_embedding = text_embedding / text_norm
            image_embedding = image_embedding / image_norm
            
            # Compute dot product similarity
            similarity = np.dot(text_embedding, image_embedding)
            
            # Scale to 0-1 range (cosine similarity is in [-1, 1])
            similarity = (similarity + 1) / 2
            
            return float(similarity)
    
    def _get_embedding_dim(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        # Try to determine embedding dimension
        if not self.load_model():
            # Return default dimension if model not loaded
            return 512
            
        try:
            model = self.model_loader.model
            
            # Different approaches based on model type
            if hasattr(model, "get_text_features"):
                # Use model config
                config = getattr(model, "config", None)
                if config and hasattr(config, "projection_dim"):
                    return config.projection_dim
                elif hasattr(model, "text_projection"):
                    return model.text_projection.shape[1]
                else:
                    # Default for CLIP models
                    return 512
                    
            elif hasattr(model, "encode_text"):
                # CLIP library model - use test encoding
                import torch
                with torch.no_grad():
                    # Create a dummy input
                    tokens = self.model_loader.tokenizer(["test"]).to(self.model_loader.device)
                    text_features = model.encode_text(tokens)
                    return text_features.shape[1]
                    
            elif hasattr(model, "run"):
                # ONNX model - infer from output shapes
                meta = model.get_outputs()
                if meta and len(meta) > 0 and hasattr(meta[0], "shape"):
                    return meta[0].shape[1]
                    
            # Default fallback
            return 512
            
        except Exception as e:
            logger.error(f"Error determining embedding dimension: {str(e)}")
            return 512
    
    @staticmethod
    def _create_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
        """
        Create batches from a list of items.
        
        Args:
            items: List of items to batch
            batch_size: Size of each batch
            
        Returns:
            List of batches
        """
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self.cache.clear()
    
    def close(self) -> None:
        """Release resources."""
        # Clear cache
        self.cache.clear()
        
        # Unload model
        if self.model_loader.model_loaded:
            try:
                # Free up memory
                self.model_loader.model = None
                self.model_loader.processor = None
                self.model_loader.tokenizer = None
                self.model_loader.model_loaded = False
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                    
                logger.info("CLIP evaluator resources released")
                
            except Exception as e:
                logger.error(f"Error releasing resources: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()