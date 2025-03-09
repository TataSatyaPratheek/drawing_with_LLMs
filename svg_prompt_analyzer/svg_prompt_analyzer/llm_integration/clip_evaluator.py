"""
CLIP Evaluator Module - Optimized
==================
This module provides highly optimized functionality for evaluating SVG-text similarity 
using CLIP models with cross-platform hardware acceleration.
"""

import os
import io
import gc
import time
import logging
import tempfile
import threading
import multiprocessing
import platform
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Global variables for device detection and caching
DEVICE_INFO = {
    "platform": platform.system(),
    "processor": platform.processor(),
    "is_apple_silicon": False,
    "has_cuda": False,
    "has_mps": False,
    "has_rocm": False,
    "cpu_cores": multiprocessing.cpu_count()
}

# Detection for Apple Silicon
if DEVICE_INFO["platform"] == "Darwin" and "arm" in platform.processor().lower():
    DEVICE_INFO["is_apple_silicon"] = True

# Implement lazy imports for optional dependencies
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
                
                # Check for MPS (Apple Silicon)
                if hasattr(torch, 'mps') and torch.mps.is_available():
                    DEVICE_INFO["has_mps"] = True
                
                # Check for ROCm (AMD)
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    DEVICE_INFO["has_rocm"] = True
                
                logger.debug(f"PyTorch {torch.__version__} loaded successfully")
                return True
            except ImportError:
                logger.warning("PyTorch not installed. CLIP evaluation will be unavailable.")
                self.imported["torch"] = None
                return False
        return self.imported["torch"] is not None
    
    def import_transformers(self):
        """Import transformers modules on demand."""
        if "transformers" not in self.imported:
            try:
                from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel
                self.imported["AutoProcessor"] = AutoProcessor
                self.imported["AutoModel"] = AutoModel
                self.imported["CLIPProcessor"] = CLIPProcessor
                self.imported["CLIPModel"] = CLIPModel
                
                logger.debug("Transformers library loaded successfully")
                return True
            except ImportError:
                logger.warning("Transformers not installed. CLIP evaluation will be unavailable.")
                self.imported["transformers"] = None
                return False
        return "AutoProcessor" in self.imported
    
    def import_cairosvg(self):
        """Import CairoSVG for SVG rendering."""
        if "cairosvg" not in self.imported:
            try:
                import cairosvg
                self.imported["cairosvg"] = cairosvg
                
                logger.debug("CairoSVG library loaded successfully")
                return True
            except ImportError:
                logger.warning("CairoSVG not installed. SVG rendering will be limited.")
                self.imported["cairosvg"] = None
                return False
        return self.imported["cairosvg"] is not None
    
    def import_pil(self):
        """Import PIL for image processing."""
        if "PIL" not in self.imported:
            try:
                from PIL import Image
                self.imported["Image"] = Image
                
                logger.debug("PIL library loaded successfully")
                return True
            except ImportError:
                logger.warning("PIL not installed. Image processing will be unavailable.")
                self.imported["PIL"] = None
                return False
        return "Image" in self.imported
    
    def get_module(self, name):
        """Get an imported module by name."""
        return self.imported.get(name)


class CLIPEvaluator:
    """
    Optimized class for evaluating SVG-text similarity using CLIP-based models.
    Supports hardware acceleration across various platforms.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for resource efficiency."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CLIPEvaluator, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, 
                 model_name: str = "SigLIP-SoViT-400m",
                 device: str = "auto",
                 cache_dir: str = ".cache/clip",
                 use_normalized_score: bool = True,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 image_size: int = 224,
                 precision: str = "fp16"):
        """
        Initialize the CLIP evaluator with model configurations.
        
        Args:
            model_name: Name of the CLIP model to use
            device: Device to run models on ('cpu', 'cuda', 'mps', 'auto')
            cache_dir: Directory for model caching
            use_normalized_score: Whether to normalize similarity scores
            batch_size: Batch size for processing multiple images
            num_workers: Number of workers for data loading
            image_size: Size to resize images to
            precision: Computation precision ('fp32', 'fp16', 'bf16')
        """
        # Initialize only once (singleton pattern)
        if hasattr(self, 'initialized'):
            return
        
        self.lazy_importer = LazyImporter()
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_normalized_score = use_normalized_score
        self.batch_size = batch_size
        self.num_workers = min(num_workers, os.cpu_count() or 4)
        self.image_size = image_size
        self.precision = precision
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Determine optimal device
        self.device = self._determine_optimal_device(device)
        
        # SVG rendering cache
        self._svg_cache = {}
        self._cache_max_size = 100
        
        # Track model loading status
        self.model_loaded = False
        self.model = None
        self.processor = None
        
        # Prepare embedding cache
        self._text_embedding_cache = {}
        self._image_embedding_cache = {}
        
        # Flag for initialization
        self.initialized = True
        logger.info(f"CLIP Evaluator initialized with device: {self.device}")
        
    def _determine_optimal_device(self, requested_device: str) -> str:
        """
        Determine the optimal device based on available hardware and request.
        
        Args:
            requested_device: Requested device ('cpu', 'cuda', 'mps', 'auto')
            
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
        
    def load_model(self) -> bool:
        """
        Load the CLIP model for similarity evaluation with appropriate optimizations.
        
        Returns:
            Whether the model was successfully loaded
        """
        if self.model_loaded:
            return True
            
        try:
            # Try loading the requested model
            if self.model_name == "SigLIP-SoViT-400m":
                return self._load_siglip_model()
            else:
                return self._load_generic_clip_model()
                
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            return False
            
    def _load_siglip_model(self) -> bool:
        """Load the SigLIP model with optimizations for current hardware."""
        try:
            # Ensure required dependencies are available
            if not self.lazy_importer.import_torch() or not self.lazy_importer.import_transformers():
                logger.error("Required dependencies not available for SigLIP model")
                return False
                
            torch = self.lazy_importer.get_module("torch")
            AutoProcessor = self.lazy_importer.get_module("AutoProcessor")
            AutoModel = self.lazy_importer.get_module("AutoModel")
            
            start_time = time.time()
            
            # Determine optimal precision based on device and settings
            dtype = None
            if self.precision == "fp16" and self.device in ["cuda", "xpu"]:
                dtype = torch.float16
            elif self.precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            
            # Load processor
            logger.info("Loading SigLIP processor...")
            self.processor = AutoProcessor.from_pretrained(
                "google/siglip-base-patch16-224", 
                cache_dir=self.cache_dir
            )
            
            # Load model with optimizations
            logger.info("Loading SigLIP model...")
            self.model = AutoModel.from_pretrained(
                "google/siglip-base-patch16-224", 
                cache_dir=self.cache_dir,
                torch_dtype=dtype
            )
            
            # Move model to device and apply optimizations
            if self.device != "cpu":
                self.model = self.model.to(self.device)
                
            # Apply device-specific optimizations
            self._apply_model_optimizations()
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.model_loaded = True
            load_time = time.time() - start_time
            logger.info(f"Successfully loaded SigLIP model in {load_time:.2f} seconds")
            
            return True
            
        except ImportError:
            logger.error("Failed to import required libraries for SigLIP model")
            return False
        except Exception as e:
            logger.error(f"Error loading SigLIP model: {str(e)}")
            return False
            
    def _load_generic_clip_model(self) -> bool:
        """Load a generic CLIP model with optimizations."""
        try:
            # Ensure required dependencies are available
            if not self.lazy_importer.import_torch() or not self.lazy_importer.import_transformers():
                logger.error("Required dependencies not available for CLIP model")
                return False
                
            torch = self.lazy_importer.get_module("torch")
            CLIPProcessor = self.lazy_importer.get_module("CLIPProcessor")
            CLIPModel = self.lazy_importer.get_module("CLIPModel")
            
            start_time = time.time()
            
            # Determine model name to load
            model_name = "openai/clip-vit-base-patch32"
            
            # Determine optimal precision based on device and settings
            dtype = None
            if self.precision == "fp16" and self.device in ["cuda", "xpu"]:
                dtype = torch.float16
            elif self.precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            
            # Load processor and model
            logger.info(f"Loading CLIP processor for {model_name}...")
            self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=self.cache_dir)
            
            logger.info(f"Loading CLIP model {model_name}...")
            self.model = CLIPModel.from_pretrained(
                model_name, 
                cache_dir=self.cache_dir,
                torch_dtype=dtype
            )
            
            # Move model to device and apply optimizations
            if self.device != "cpu":
                self.model = self.model.to(self.device)
                
            # Apply device-specific optimizations
            self._apply_model_optimizations()
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.model_loaded = True
            load_time = time.time() - start_time
            logger.info(f"Successfully loaded CLIP model in {load_time:.2f} seconds")
            
            return True
            
        except ImportError:
            logger.error("Failed to import required libraries for CLIP model")
            return False
        except Exception as e:
            logger.error(f"Error loading CLIP model: {str(e)}")
            return False
    
    def _apply_model_optimizations(self) -> None:
        """Apply device-specific optimizations to the model."""
        if not self.lazy_importer.import_torch():
            return
            
        torch = self.lazy_importer.get_module("torch")
        
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
                # Use all available cores but one for system operations
                torch.set_num_threads(max(1, DEVICE_INFO["cpu_cores"] - 1))
                
            # Enable MKL optimizations if available
            if hasattr(torch, 'set_num_interop_threads'):
                torch.set_num_interop_threads(max(1, DEVICE_INFO["cpu_cores"] // 2))
        
        # Use torch script to optimize model if possible
        try:
            if hasattr(self.model, 'torchscript'):
                logger.info("Applying TorchScript optimization")
                self.model = torch.jit.script(self.model)
        except Exception as e:
            logger.debug(f"TorchScript optimization skipped: {str(e)}")
    
    @lru_cache(maxsize=128)
    def _get_cache_key(self, svg_code: str) -> str:
        """
        Generate a deterministic cache key for SVG content.
        
        Args:
            svg_code: SVG code to generate key for
            
        Returns:
            Cache key string
        """
        import hashlib
        # Create a deterministic hash of the SVG code
        return hashlib.md5(svg_code.encode()).hexdigest()
            
    def compute_similarity(self, svg_code: str, text_prompt: str) -> float:
        """
        Compute similarity between SVG and text using CLIP.
        
        Args:
            svg_code: SVG code to evaluate
            text_prompt: Text prompt to compare against
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Check if model is loaded
        if not self.model_loaded and not self.load_model():
            logger.error("CLIP model not loaded, cannot compute similarity")
            return 0.0
            
        try:
            # Convert SVG to image
            image = self._svg_to_image(svg_code)
            if image is None:
                logger.error("Failed to convert SVG to image for CLIP evaluation")
                return 0.0
                
            # Prepare text with prefix for CLIP matching
            if not text_prompt.startswith("SVG illustration of "):
                text_prompt = f"SVG illustration of {text_prompt}"
                
            # Get embeddings
            text_embedding = self._get_text_embedding(text_prompt)
            image_embedding = self._get_image_embedding(image)
            
            if text_embedding is None or image_embedding is None:
                logger.error("Failed to generate embeddings")
                return 0.0
                
            # Compute similarity
            similarity = self._compute_embedding_similarity(text_embedding, image_embedding)
            
            logger.info(f"CLIP similarity score: {similarity:.4f}")
            return similarity
            
        except Exception as e:
            logger.error(f"Error computing CLIP similarity: {str(e)}")
            return 0.0
    
    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get text embedding with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding array
        """
        # Check cache
        if text in self._text_embedding_cache:
            return self._text_embedding_cache[text]
            
        try:
            torch = self.lazy_importer.get_module("torch")
            
            # Process input
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            # Generate embedding
            with torch.no_grad():
                if self.model_name == "SigLIP-SoViT-400m":
                    # For SigLIP model
                    outputs = self.model.get_text_features(**inputs)
                    text_embedding = outputs.cpu().numpy()
                else:
                    # For standard CLIP model
                    outputs = self.model.get_text_features(**inputs)
                    text_embedding = outputs.cpu().numpy()
                    
            # Cache the embedding
            self._text_embedding_cache[text] = text_embedding
            
            # Limit cache size
            if len(self._text_embedding_cache) > 100:
                # Remove oldest key (arbitrary but consistent)
                oldest_key = next(iter(self._text_embedding_cache))
                del self._text_embedding_cache[oldest_key]
                
            return text_embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            return None
            
    def _get_image_embedding(self, image) -> Optional[np.ndarray]:
        """
        Get image embedding with caching.
        
        Args:
            image: PIL Image to embed
            
        Returns:
            Image embedding array
        """
        # Create a cache key from image data
        image_data = self._get_image_data(image)
        cache_key = hash(image_data)
        
        # Check cache
        if cache_key in self._image_embedding_cache:
            return self._image_embedding_cache[cache_key]
            
        try:
            torch = self.lazy_importer.get_module("torch")
            
            # Process input
            inputs = self.processor(
                images=[image],
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
            # Generate embedding
            with torch.no_grad():
                if self.model_name == "SigLIP-SoViT-400m":
                    # For SigLIP model
                    outputs = self.model.get_image_features(**inputs)
                    image_embedding = outputs.cpu().numpy()
                else:
                    # For standard CLIP model
                    outputs = self.model.get_image_features(**inputs)
                    image_embedding = outputs.cpu().numpy()
                    
            # Cache the embedding
            self._image_embedding_cache[cache_key] = image_embedding
            
            # Limit cache size
            if len(self._image_embedding_cache) > 100:
                # Remove oldest key (arbitrary but consistent)
                oldest_key = next(iter(self._image_embedding_cache))
                del self._image_embedding_cache[oldest_key]
                
            return image_embedding
            
        except Exception as e:
            logger.error(f"Error generating image embedding: {str(e)}")
            return None
    
    def _get_image_data(self, image) -> bytes:
        """Convert image to serializable bytes for caching."""
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return buffer.getvalue()
            
    def _compute_embedding_similarity(self, text_embedding: np.ndarray, image_embedding: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            text_embedding: Text embedding array
            image_embedding: Image embedding array
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Normalize embeddings
        text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)
        image_embedding = image_embedding / np.linalg.norm(image_embedding, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarity = np.sum(text_embedding * image_embedding)
        
        # Normalize score if requested
        if self.use_normalized_score:
            # Convert cosine similarity to a 0-1 range (it's originally -1 to 1)
            similarity = (similarity + 1) / 2
            
        return float(similarity)
            
    def _svg_to_image(self, svg_code: str) -> Optional[Any]:
        """
        Convert SVG code to a PIL Image for CLIP processing with caching.
        
        Args:
            svg_code: SVG code to convert
            
        Returns:
            PIL Image or None if conversion fails
        """
        # Get cache key
        cache_key = self._get_cache_key(svg_code)
        
        # Check cache
        if cache_key in self._svg_cache:
            return self._svg_cache[cache_key]
        
        try:
            # Ensure CairoSVG and PIL are available
            if not self.lazy_importer.import_cairosvg() or not self.lazy_importer.import_pil():
                logger.error("Required dependencies for SVG conversion not available")
                return None
                
            cairosvg = self.lazy_importer.get_module("cairosvg")
            Image = self.lazy_importer.get_module("Image")
            
            # Convert to PNG in memory
            png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
            
            # Load as PIL Image
            image = Image.open(io.BytesIO(png_data))
            
            # Resize for CLIP model
            image = image.resize((self.image_size, self.image_size))
            
            # Cache the result
            self._svg_cache[cache_key] = image
            
            # Limit cache size
            if len(self._svg_cache) > self._cache_max_size:
                # Remove oldest key
                oldest_key = next(iter(self._svg_cache))
                del self._svg_cache[oldest_key]
                
            return image
                
        except Exception as e:
            logger.error(f"Error converting SVG to image: {str(e)}")
            
            # Try fallback method with temp files if memory conversion failed
            return self._svg_to_image_fallback(svg_code)
            
    def _svg_to_image_fallback(self, svg_code: str) -> Optional[Any]:
        """
        Fallback method for SVG to image conversion using temporary files.
        
        Args:
            svg_code: SVG code to convert
            
        Returns:
            PIL Image or None if conversion fails
        """
        try:
            # Ensure CairoSVG and PIL are available
            if not self.lazy_importer.import_cairosvg() or not self.lazy_importer.import_pil():
                logger.error("Required dependencies for SVG conversion not available")
                return None
                
            cairosvg = self.lazy_importer.get_module("cairosvg")
            Image = self.lazy_importer.get_module("Image")
            
            # Create temp files
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as svg_file:
                svg_file.write(svg_code.encode('utf-8'))
                svg_path = svg_file.name
                
            try:
                # Create temp PNG file
                png_path = f"{svg_path}.png"
                
                # Convert SVG to PNG
                cairosvg.svg2png(url=svg_path, write_to=png_path)
                
                # Load image
                image = Image.open(png_path)
                
                # Resize for CLIP model
                image = image.resize((self.image_size, self.image_size))
                
                return image
                
            finally:
                # Clean up temp files
                if os.path.exists(svg_path):
                    os.unlink(svg_path)
                if os.path.exists(f"{svg_path}.png"):
                    os.unlink(f"{svg_path}.png")
                    
        except Exception as e:
            logger.error(f"Error in SVG to image fallback: {str(e)}")
            return None
            
    def evaluate_batch(self, 
                       svg_prompt_pairs: List[Tuple[str, str]], 
                       use_parallel: bool = True,
                       callback: Optional[Callable[[int, float], None]] = None) -> List[float]:
        """
        Compute similarity for a batch of SVG-prompt pairs with optimized processing.
        
        Args:
            svg_prompt_pairs: List of (svg_code, text_prompt) tuples
            use_parallel: Whether to use parallel processing
            callback: Optional callback function called with (index, score) after each evaluation
            
        Returns:
            List of similarity scores
        """
        # Process pairs in batches for better GPU utilization
        if use_parallel and len(svg_prompt_pairs) > 1:
            return self._evaluate_batch_parallel(svg_prompt_pairs, callback)
        else:
            return self._evaluate_batch_sequential(svg_prompt_pairs, callback)
    
    def _evaluate_batch_sequential(self, 
                                  svg_prompt_pairs: List[Tuple[str, str]], 
                                  callback: Optional[Callable[[int, float], None]] = None) -> List[float]:
        """
        Process SVG-prompt pairs sequentially.
        
        Args:
            svg_prompt_pairs: List of (svg_code, text_prompt) tuples
            callback: Optional callback function called with (index, score) after each evaluation
            
        Returns:
            List of similarity scores
        """
        scores = []
        
        for i, (svg_code, text_prompt) in enumerate(svg_prompt_pairs):
            score = self.compute_similarity(svg_code, text_prompt)
            scores.append(score)
            
            # Call callback if provided
            if callback:
                callback(i, score)
                
        return scores
    
    def _evaluate_batch_parallel(self, 
                               svg_prompt_pairs: List[Tuple[str, str]], 
                               callback: Optional[Callable[[int, float], None]] = None) -> List[float]:
        """
        Process SVG-prompt pairs in parallel using thread pool.
        
        Args:
            svg_prompt_pairs: List of (svg_code, text_prompt) tuples
            callback: Optional callback function called with (index, score) after each evaluation
            
        Returns:
            List of similarity scores
        """
        # Create thread pool
        max_workers = min(self.num_workers, len(svg_prompt_pairs))
        scores = [0.0] * len(svg_prompt_pairs)  # Pre-allocate results list
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks
            futures = {}
            for i, (svg_code, text_prompt) in enumerate(svg_prompt_pairs):
                future = executor.submit(self.compute_similarity, svg_code, text_prompt)
                futures[future] = i
                
            # Collect results as they complete
            for future in futures:
                i = futures[future]
                try:
                    score = future.result()
                    scores[i] = score
                    
                    # Call callback if provided
                    if callback:
                        callback(i, score)
                        
                except Exception as e:
                    logger.error(f"Error in parallel evaluation task {i}: {str(e)}")
                    scores[i] = 0.0
                    
                    # Call callback with error score
                    if callback:
                        callback(i, 0.0)
                        
        return scores
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded CLIP model.
        
        Returns:
            Dictionary with model information
        """
        if not self.model_loaded:
            return {"status": "not_loaded", "model_name": self.model_name}
            
        info = {
            "status": "loaded",
            "model_name": self.model_name,
            "device": self.device,
            "precision": self.precision,
            "image_size": self.image_size
        }
        
        # Add memory usage info if available
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            info["memory_usage"] = {
                "rss": f"{memory_info.rss / (1024 * 1024):.2f} MB",
                "vms": f"{memory_info.vms / (1024 * 1024):.2f} MB"
            }
        except:
            pass
            
        # Add GPU memory info if available
        try:
            torch = self.lazy_importer.get_module("torch")
            if torch and torch.cuda.is_available():
                info["gpu_memory"] = {
                    "allocated": f"{torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB",
                    "reserved": f"{torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB",
                    "max_allocated": f"{torch.cuda.max_memory_allocated() / (1024 * 1024):.2f} MB"
                }
        except:
            pass
            
        return info
    
    def unload_model(self) -> bool:
        """
        Unload the CLIP model to free up resources.
        
        Returns:
            Whether the model was successfully unloaded
        """
        if not self.model_loaded:
            return True
            
        try:
            torch = self.lazy_importer.get_module("torch")
            
            # Clear processor and model
            if self.processor:
                del self.processor
                self.processor = None
                
            if self.model:
                # Move model to CPU first
                if torch and self.device != "cpu":
                    self.model = self.model.to("cpu")
                    
                del self.model
                self.model = None
                
            # Clear embedding caches
            self._text_embedding_cache.clear()
            self._image_embedding_cache.clear()
                
            # Run garbage collection
            gc.collect()
            
            # Clear device caches
            if torch:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                if hasattr(torch, 'mps') and torch.mps.is_available():
                    torch.mps.empty_cache()
                    
            self.model_loaded = False
            logger.info("CLIP model unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading CLIP model: {str(e)}")
            return False
    
    def clear_caches(self) -> None:
        """Clear all memory caches."""
        self._svg_cache.clear()
        self._text_embedding_cache.clear()
        self._image_embedding_cache.clear()
        gc.collect()