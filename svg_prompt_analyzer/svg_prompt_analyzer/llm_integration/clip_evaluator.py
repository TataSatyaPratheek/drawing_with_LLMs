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
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from io import BytesIO
from PIL import Image
import logging
import warnings
import queue
import cairosvg
import hashlib
import base64

# Import core optimizations
from core import CONFIG, memoize, Profiler, get_thread_pool
from utils.logger import get_logger

# Configure logger
logger = get_logger(__name__)

# Type aliases
ClipScore = float
ImageEmbedding = np.ndarray
TextEmbedding = np.ndarray
Batch = List[Any]

# Constants
DEFAULT_IMAGE_SIZE = 224  # Default CLIP input size
DEFAULT_BATCH_SIZE = 32
CACHE_SIZE = 2048  # Number of embeddings to cache
WORKER_TIMEOUT = 0.1  # Seconds to wait for worker threads
DEFAULT_TOP_K = 5


class ClipCache:
    """Thread-safe cache for CLIP embeddings."""
    
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
    
    def get_image_embedding(self, image: Union[Image.Image, np.ndarray]) -> Optional[ImageEmbedding]:
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
    
    def set_image_embedding(self, image: Union[Image.Image, np.ndarray], embedding: ImageEmbedding) -> None:
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
    
    def _get_image_key(self, image: Union[Image.Image, np.ndarray]) -> str:
        """
        Generate cache key for image.
        
        Args:
            image: Image to generate key for
            
        Returns:
            Cache key
        """
        # Convert image to bytes for hashing
        if isinstance(image, Image.Image):
            image_bytes = BytesIO()
            image.save(image_bytes, format='PNG')
            image_data = image_bytes.getvalue()
        else:
            # Numpy array
            image_data = image.tobytes()
            
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


class ClipEvaluator:
    """
    Production-grade CLIP-based evaluator for image-text matching.
    
    Uses CLIP embeddings to evaluate how well images match text prompts.
    Includes optimizations for memory usage, batch processing, and caching.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_size: int = CACHE_SIZE,
        use_jit: bool = None,
        use_fp16: bool = None,
        thread_workers: int = None
    ):
        """
        Initialize CLIP evaluator.
        
        Args:
            model_name: CLIP model name
            device: Device to run on (None for auto-detection)
            batch_size: Batch size for processing
            cache_size: Size of embedding cache
            use_jit: Whether to use JIT compilation
            use_fp16: Whether to use FP16 precision
            thread_workers: Number of worker threads
        """
        self.model_name = model_name
        
        # Auto-detect device if not provided
        self.device = device
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Set up parameters
        self.batch_size = batch_size
        
        # Use values from config if not provided
        self.use_jit = use_jit if use_jit is not None else CONFIG.get("enable_jit", True)
        self.use_fp16 = use_fp16 if use_fp16 is not None else (self.device == "cuda")
        
        # Set up thread workers
        self.thread_workers = thread_workers or CONFIG.get(
            "thread_pool_size",
            min(32, os.cpu_count() or 4)
        )
        
        # Initialize cache
        self.cache = ClipCache(max_size=cache_size)
        
        # Initialize model
        self._model = None
        self._processor = None
        self._tokenizer = None
        
        # Threading resources
        self._worker_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._workers = []
        
        # Lazy initialization flag
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """
        Ensure model is initialized.
        
        Lazy initialization to avoid loading the model until needed.
        """
        if self._initialized:
            return
            
        with Profiler("clip_init"):
            logger.info(f"Initializing CLIP model: {self.model_name} on {self.device}")
            
            try:
                # Import here for lazy loading
                import torch
                import transformers
                from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
                
                # Load model and processor
                self._model = CLIPModel.from_pretrained(self.model_name)
                self._processor = CLIPProcessor.from_pretrained(self.model_name)
                self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
                
                # Move model to device
                self._model.to(self.device)
                
                # Apply JIT if enabled
                if self.use_jit and hasattr(torch, "jit"):
                    logger.info("Using JIT compilation for CLIP model")
                    self._model = torch.jit.script(self._model)
                    
                # Use FP16 if enabled
                if self.use_fp16 and self.device == "cuda":
                    logger.info("Using FP16 precision for CLIP model")
                    self._model.half()
                    
                # Set to evaluation mode
                self._model.eval()
                
                # Start worker threads
                self._start_workers()
                
                self._initialized = True
                logger.info("CLIP model initialized successfully")
                
            except Exception as e:
                logger.error(f"Error initializing CLIP model: {str(e)}")
                raise
    
    def _start_workers(self) -> None:
        """Start worker threads for batch processing."""
        for i in range(self.thread_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"clip_worker_{i}"
            )
            worker.start()
            self._workers.append(worker)
            
        logger.debug(f"Started {self.thread_workers} CLIP worker threads")
    
    def _worker_loop(self) -> None:
        """Worker thread loop."""
        while not self._stop_event.is_set():
            try:
                # Get task with timeout
                task = self._worker_queue.get(timeout=WORKER_TIMEOUT)
                task_type, data, task_id = task
                
                # Process task
                if task_type == "text":
                    result = self._encode_text_batch(data)
                elif task_type == "image":
                    result = self._encode_image_batch(data)
                else:
                    logger.warning(f"Unknown task type: {task_type}")
                    result = None
                    
                # Put result in queue
                self._result_queue.put((task_id, result))
                
                # Mark task as done
                self._worker_queue.task_done()
                
            except queue.Empty:
                # No tasks available
                continue
            except Exception as e:
                logger.error(f"Error in CLIP worker thread: {str(e)}")
                
                # Put error in result queue
                self._result_queue.put((task_id, e))
                
                # Mark task as done
                self._worker_queue.task_done()
    
    def _encode_text_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Batch of text embeddings
        """
        import torch
        
        with torch.no_grad():
            # Tokenize texts
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get text embeddings
            text_features = self._model.get_text_features(**inputs)
            
            # Normalize embeddings
            text_embeddings = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Convert to numpy
            return text_embeddings.cpu().numpy()
    
    def _encode_image_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode a batch of images.
        
        Args:
            images: List of images to encode
            
        Returns:
            Batch of image embeddings
        """
        import torch
        
        with torch.no_grad():
            # Process images
            inputs = self._processor(
                images=images,
                return_tensors="pt"
            ).to(self.device)
            
            # Get image embeddings
            image_features = self._model.get_image_features(**inputs)
            
            # Normalize embeddings
            image_embeddings = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Convert to numpy
            return image_embeddings.cpu().numpy()
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of text embeddings
        """
        self._ensure_initialized()
        
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
                
                # Process batches in parallel
                batch_results = self._process_batches("text", text_batches)
                
                # Combine batches
                all_embeddings = np.vstack(batch_results)
                
                # Add to result
                for i, embedding_idx in enumerate(text_indices):
                    result[embedding_idx] = all_embeddings[i]
                    
                    # Cache embedding
                    self.cache.set_text_embedding(texts_to_encode[i], all_embeddings[i])
                    
            return result
    
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode images into embeddings.
        
        Args:
            images: List of images to encode
            
        Returns:
            Array of image embeddings
        """
        self._ensure_initialized()
        
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
                
                # Process batches in parallel
                batch_results = self._process_batches("image", image_batches)
                
                # Combine batches
                all_embeddings = np.vstack(batch_results)
                
                # Add to result
                for i, embedding_idx in enumerate(image_indices):
                    result[embedding_idx] = all_embeddings[i]
                    
                    # Cache embedding
                    self.cache.set_image_embedding(images_to_encode[i], all_embeddings[i])
                    
            return result
    
    def encode_svg(self, svg_content: str, size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
        """
        Encode SVG content into an embedding.
        
        Args:
            svg_content: SVG content to encode
            size: Size of rasterized image
            
        Returns:
            Embedding for the SVG
        """
        self._ensure_initialized()
        
        with Profiler("encode_svg"):
            # Check cache
            cached = self.cache.get_svg_embedding(svg_content)
            if cached is not None:
                return cached
                
            # Convert SVG to PNG
            png_data = self._svg_to_png(svg_content, size)
            
            # Convert PNG to image
            image = Image.open(BytesIO(png_data))
            
            # Encode image
            embedding = self.encode_images([image])[0]
            
            # Cache embedding
            self.cache.set_svg_embedding(svg_content, embedding)
            
            return embedding
    
    def encode_svgs(self, svg_contents: List[str], size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
        """
        Encode multiple SVG contents into embeddings.
        
        Args:
            svg_contents: List of SVG contents to encode
            size: Size of rasterized images
            
        Returns:
            Array of SVG embeddings
        """
        self._ensure_initialized()
        
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
                
            # Convert SVGs to images
            images = []
            for svg in svgs_to_encode:
                try:
                    # Convert SVG to PNG
                    png_data = self._svg_to_png(svg, size)
                    
                    # Convert PNG to image
                    image = Image.open(BytesIO(png_data))
                    images.append(image)
                except Exception as e:
                    logger.error(f"Error converting SVG to image: {str(e)}")
                    
                    # Use empty image as fallback
                    images.append(Image.new('RGB', (size, size), color='white'))
                    
            # Encode images
            if images:
                image_embeddings = self.encode_images(images)
                
                # Add to result
                for i, embedding_idx in enumerate(svg_indices):
                    result[embedding_idx] = image_embeddings[i]
                    
                    # Cache embedding
                    self.cache.set_svg_embedding(svgs_to_encode[i], image_embeddings[i])
                    
            return result
    
    def compute_similarity(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between text and image embeddings.
        
        Args:
            text_embeddings: Text embeddings
            image_embeddings: Image embeddings
            
        Returns:
            Similarity matrix [text_count, image_count]
        """
        with Profiler("compute_similarity"):
            # Ensure embeddings are normalized
            text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
            image_norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
            
            text_embeddings = text_embeddings / np.maximum(text_norms, 1e-10)
            image_embeddings = image_embeddings / np.maximum(image_norms, 1e-10)
            
            # Compute dot product similarity
            similarity = np.matmul(text_embeddings, image_embeddings.T)
            
            # Scale to 0-1 range
            similarity = (similarity + 1) / 2
            
            return similarity
    
    def rank_images(
        self,
        query: str,
        images: List[Image.Image],
        top_k: int = DEFAULT_TOP_K
    ) -> List[Tuple[int, ClipScore]]:
        """
        Rank images by similarity to query.
        
        Args:
            query: Text query
            images: List of images to rank
            top_k: Number of top results to return
            
        Returns:
            List of (image_index, similarity_score) tuples
        """
        self._ensure_initialized()
        
        with Profiler("rank_images"):
            # Encode query
            text_embedding = self.encode_texts([query])
            
            # Encode images
            image_embeddings = self.encode_images(images)
            
            # Compute similarity
            similarity = self.compute_similarity(text_embedding, image_embeddings)[0]
            
            # Get top-k indices
            top_indices = np.argsort(-similarity)[:top_k]
            
            # Create result
            result = [(int(idx), float(similarity[idx])) for idx in top_indices]
            
            return result
    
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
        self._ensure_initialized()
        
        with Profiler("rank_svgs"):
            # Encode query
            text_embedding = self.encode_texts([query])
            
            # Encode SVGs
            svg_embeddings = self.encode_svgs(svg_contents)
            
            # Compute similarity
            similarity = self.compute_similarity(text_embedding, svg_embeddings)[0]
            
            # Get top-k indices
            top_indices = np.argsort(-similarity)[:top_k]
            
            # Create result
            result = [(int(idx), float(similarity[idx])) for idx in top_indices]
            
            return result
    
    def evaluate_image(self, query: str, image: Image.Image) -> ClipScore:
        """
        Evaluate how well an image matches a query.
        
        Args:
            query: Text query
            image: Image to evaluate
            
        Returns:
            Similarity score (0-1)
        """
        self._ensure_initialized()
        
        with Profiler("evaluate_image"):
            # Encode query
            text_embedding = self.encode_texts([query])
            
            # Encode image
            image_embedding = self.encode_images([image])
            
            # Compute similarity
            similarity = self.compute_similarity(text_embedding, image_embedding)[0, 0]
            
            return float(similarity)
    
    def evaluate_svg(self, query: str, svg_content: str) -> ClipScore:
        """
        Evaluate how well an SVG matches a query.
        
        Args:
            query: Text query
            svg_content: SVG content to evaluate
            
        Returns:
            Similarity score (0-1)
        """
        self._ensure_initialized()
        
        with Profiler("evaluate_svg"):
            # Encode query
            text_embedding = self.encode_texts([query])
            
            # Encode SVG
            svg_embedding = self.encode_svg(svg_content).reshape(1, -1)
            
            # Compute similarity
            similarity = self.compute_similarity(text_embedding, svg_embedding)[0, 0]
            
            return float(similarity)
    
    def evaluate_batch(
        self,
        queries: List[str],
        svg_contents: List[str]
    ) -> np.ndarray:
        """
        Evaluate how well multiple SVGs match multiple queries.
        
        Args:
            queries: List of text queries
            svg_contents: List of SVG contents to evaluate
            
        Returns:
            Similarity matrix [query_count, svg_count]
        """
        self._ensure_initialized()
        
        with Profiler("evaluate_batch"):
            # Encode queries
            text_embeddings = self.encode_texts(queries)
            
            # Encode SVGs
            svg_embeddings = self.encode_svgs(svg_contents)
            
            # Compute similarity
            similarity = self.compute_similarity(text_embeddings, svg_embeddings)
            
            return similarity
    
    def close(self) -> None:
        """
        Clean up resources.
        
        Should be called when the evaluator is no longer needed.
        """
        # Stop worker threads
        self._stop_event.set()
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=0.5)
            
        # Clear model and processor
        self._model = None
        self._processor = None
        self._tokenizer = None
        
        # Clear cache
        self.cache.clear()
        
        # Reset initialization flag
        self._initialized = False
        
        logger.info("CLIP evaluator resources released")
    
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
            # Use cairosvg for conversion
            try:
                png_data = cairosvg.svg2png(
                    bytestring=svg_content.encode('utf-8'),
                    output_width=size,
                    output_height=size
                )
                return png_data
            except Exception as e:
                logger.error(f"Error converting SVG to PNG: {str(e)}")
                
                # Create a fallback image (white background with error message)
                img = Image.new('RGB', (size, size), color='white')
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), "SVG Error", fill='red')
                
                # Convert to PNG data
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG')
                return img_bytes.getvalue()
    
    def _get_embedding_dim(self) -> int:
        """
        Get dimension of embeddings.
        
        Returns:
            Embedding dimension
        """
        if not self._initialized:
            self._ensure_initialized()
            
        # Get embedding dimension from model
        import torch
        
        with torch.no_grad():
            # Create a dummy input
            input_ids = torch.zeros((1, 1), dtype=torch.long).to(self.device)
            attention_mask = torch.ones((1, 1), dtype=torch.long).to(self.device)
            
            # Get text features
            text_features = self._model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Return dimension
            return text_features.shape[1]
    
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
    
    def _process_batches(self, task_type: str, batches: List[Batch]) -> List[np.ndarray]:
        """
        Process batches in parallel.
        
        Args:
            task_type: Type of task ("text" or "image")
            batches: List of batches to process
            
        Returns:
            List of results for each batch
        """
        # Task IDs for tracking results
        task_ids = list(range(len(batches)))
        
        # Submit tasks to queue
        for i, batch in enumerate(batches):
            self._worker_queue.put((task_type, batch, i))
            
        # Wait for results
        results = [None] * len(batches)
        for _ in range(len(batches)):
            task_id, result = self._result_queue.get()
            
            # Check for error
            if isinstance(result, Exception):
                raise result
                
            results[task_id] = result
            
        return results
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()