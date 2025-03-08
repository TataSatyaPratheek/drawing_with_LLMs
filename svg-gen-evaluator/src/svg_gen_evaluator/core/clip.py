"""
CLIP similarity calculation for evaluating SVG quality.
"""
import logging
from typing import Optional, Union

import torch
from PIL import Image
from torch.nn.functional import cosine_similarity

logger = logging.getLogger(__name__)

# Define model constants
DEFAULT_MODEL_NAME = "google/siglip-so400m-patch14-384"
FALLBACK_MODEL_NAME = "google/siglip-base-patch16-224"


class CLIPSimilarityCalculator:
    """
    Calculates CLIP similarity between images and text descriptions.
    
    Uses SigLIP model to compute embeddings and similarity scores to evaluate
    how well generated SVGs match their text descriptions.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ):
        """
        Initialize the CLIP similarity calculator.
        
        Args:
            model_name: Name of the pretrained CLIP model to use
            device: Device to run the model on (cuda or cpu)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        
        # Load model and processor
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the SigLIP model and processor."""
        try:
            from transformers import AutoProcessor, AutoModel
        except ImportError:
            logger.error("transformers package not installed. Installing...")
            import subprocess
            import sys
            
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
            from transformers import AutoProcessor, AutoModel
        
        try:
            logger.info(f"Loading SigLIP model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.warning(f"Error loading primary model: {e}")
            logger.warning(f"Trying fallback model: {FALLBACK_MODEL_NAME}")
            
            try:
                self.model_name = FALLBACK_MODEL_NAME
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                logger.info(f"Fallback model loaded successfully on {self.device}")
            except Exception as e2:
                logger.error(f"Error loading fallback model: {e2}")
                raise RuntimeError(
                    "Could not load any SigLIP model. Ensure you have internet "
                    "access for the first run or the model is cached locally."
                )
    
    def compute_similarity(self, description: str, image: Union[Image.Image, str]) -> float:
        """
        Compute CLIP similarity between text description and image.
        
        Args:
            description: Text description
            image: PIL Image or path to image file
            
        Returns:
            Similarity score (cosine similarity between embeddings)
        """
        if self.model is None or self.processor is None:
            self._load_model()
        
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Prepend "SVG illustration of " to match competition evaluation
        full_description = f"SVG illustration of {description}"
        
        # Prepare inputs for CLIP model
        inputs = self.processor(
            text=[full_description],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Normalize embeddings and calculate similarity
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        similarity = cosine_similarity(text_embeds, image_embeds).item()
        
        return similarity