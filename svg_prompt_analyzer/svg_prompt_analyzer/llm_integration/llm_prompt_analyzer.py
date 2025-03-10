"""
Production-grade prompt analyzer for SVG generation.
Provides optimized implementation for analyzing and enhancing text prompts
to improve SVG generation results with memory and performance optimizations.
"""

import re
import json
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from enum import Enum, auto
from dataclasses import dataclass, field

# Import core optimizations
from svg_prompt_analyzer.core import CONFIG, memoize, Profiler
from svg_prompt_analyzer.core.memory_manager import MemoryManager
from svg_prompt_analyzer.utils.logger import get_logger, log_function_call

# Import LLM manager for prompt enhancement
from svg_prompt_analyzer.llm_integration.llm_manager import (
    LLMManager, extract_json_from_text
)

# Configure logger
logger = get_logger(__name__)

# Type aliases
PromptStr = str
JsonDict = Dict[str, Any]

# Get memory manager instance
memory_manager = MemoryManager()


class PromptCategory(Enum):
    """Categories of prompts for SVG generation."""
    SCENE = auto()         # Scene description
    OBJECT = auto()        # Object description
    STYLE = auto()         # Style description
    CONCEPT = auto()       # Abstract concept
    TECHNICAL = auto()     # Technical specification
    AMBIGUOUS = auto()     # Unclear or ambiguous
    COMPLEX = auto()       # Complex or compound
    MINIMAL = auto()       # Minimal or simple


@dataclass
class PromptAnalysis:
    """Analysis result for an SVG generation prompt."""
    
    # Original prompt
    original_prompt: str
    
    # Enhanced prompt
    enhanced_prompt: Optional[str] = None
    
    # Parsed components
    subject: Optional[str] = None
    objects: List[str] = field(default_factory=list)
    attributes: Dict[str, List[str]] = field(default_factory=dict)
    colors: List[str] = field(default_factory=list)
    style_cues: List[str] = field(default_factory=list)
    
    # Categorization
    category: Optional[PromptCategory] = None
    complexity: float = 0.0  # 0.0-1.0
    
    # Potential issues
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Keywords for searching
    keywords: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_time: float = 0.0
    
    def to_dict(self) -> JsonDict:
        """Convert analysis to dictionary."""
        return {
            "original_prompt": self.original_prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "subject": self.subject,
            "objects": self.objects,
            "attributes": self.attributes,
            "colors": self.colors,
            "style_cues": self.style_cues,
            "category": self.category.name if self.category else None,
            "complexity": self.complexity,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "keywords": self.keywords,
            "processing_time": self.processing_time
        }
    
    @classmethod
    def from_dict(cls, data: JsonDict) -> 'PromptAnalysis':
        """Create analysis from dictionary."""
        # Convert category string to enum if present
        category = None
        if 'category' in data and data['category']:
            try:
                category = PromptCategory[data['category']]
            except KeyError:
                pass
                
        # Create instance with basic fields
        analysis = cls(
            original_prompt=data.get('original_prompt', ''),
            enhanced_prompt=data.get('enhanced_prompt'),
            subject=data.get('subject'),
            objects=data.get('objects', []),
            colors=data.get('colors', []),
            style_cues=data.get('style_cues', []),
            category=category,
            complexity=data.get('complexity', 0.0),
            issues=data.get('issues', []),
            suggestions=data.get('suggestions', []),
            keywords=data.get('keywords', []),
            processing_time=data.get('processing_time', 0.0)
        )
        
        # Add attributes if present
        if 'attributes' in data:
            analysis.attributes = data['attributes']
            
        return analysis


class KeywordExtractor:
    """Extracts keywords from text for search and matching."""
    
    def __init__(self):
        """Initialize keyword extractor."""
        # Common stop words to exclude
        self.stop_words = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "from", "by", "on", "off", "for", "in", "out", "over", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "any", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "can", "will", "just", "should", "now"
        }
    
    @memoize
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Tokenize and clean text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stop words
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        # Sort by frequency (descending)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, _ in sorted_words[:max_keywords]]


class LLMPromptAnalyzer:
    """
    Production-grade prompt analyzer for SVG generation using LLMs.
    
    Analyzes and enhances text prompts for SVG generation using
    a combination of rule-based techniques and LLM-based enhancement.
    """
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        use_fallback: bool = True,
        fallback_threshold: float = 0.7,
        use_caching: bool = True,
        cache_size: int = 100
    ):
        """
        Initialize prompt analyzer.
        
        Args:
            llm_manager: LLM manager for advanced analysis
            use_fallback: Whether to use rule-based fallback if LLM fails
            fallback_threshold: Confidence threshold for fallback
            use_caching: Whether to cache analysis results
            cache_size: Maximum cache size
        """
        self.llm_manager = llm_manager or LLMManager()
        self.use_fallback = use_fallback
        self.fallback_threshold = fallback_threshold
        
        # Set up keyword extractor
        self.keyword_extractor = KeywordExtractor()
        
        # Set up cache
        self.use_caching = use_caching
        self._cache = {}
        self._cache_size = cache_size
    
    @log_function_call()
    def analyze_prompt(self, prompt_id: str, prompt: str) -> Dict[str, Any]:
        """
        Analyze prompt to create scene representation.
        
        Args:
            prompt_id: Unique identifier for the prompt
            prompt: Text prompt to analyze
            
        Returns:
            Scene representation dictionary
        """
        # Check cache
        if self.use_caching:
            cache_key = self._create_cache_key(prompt)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        with Profiler("llm_prompt_analysis"):
            try:
                # Create system prompt for LLM
                system_prompt = """You are an expert SVG designer analyzing prompts for SVG generation. 
                Create a structured scene representation from the prompt that can be used to generate an SVG image.

                Return a JSON object with the following structure:
                {
                  "scene": {
                    "id": "scene_[unique_id]",
                    "prompt": "[original_prompt]",
                    "width": 800,
                    "height": 600,
                    "background_color": "[color]",
                    "objects": [
                      {
                        "id": "obj_[index]", 
                        "name": "[object_name]",
                        "object_type": "[object_type]",
                        "position": [x, y],
                        "size": 0.2,
                        "z_index": [layer_depth],
                        "color": {"name": "[color_name]", "hex_code": "[hex_code]"},
                        "material": {"name": "[material_name]", "texture": "[texture_description]"},
                        "rotation": [rotation_angle],
                        "shapes": [{"shape_type": "[shape_type]", "attributes": []}]
                      }
                    ],
                    "relationships": []
                  }
                }

                Ensure each object has appropriate:
                - Position (x,y coordinates from 0.0-1.0)
                - Size (relative from 0.0-1.0)
                - Color
                - Shape type
                - Z-index for layering (higher numbers appear on top)

                Make the scene visually coherent and directly represent what's described in the prompt."""

                user_prompt = f'Create a scene representation for this prompt: "{prompt}"'
                
                # Generate scene with LLM
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # Generate response with LLM
                response = self.llm_manager.generate(
                    role="prompt_analyzer",
                    prompt=messages,
                    max_tokens=2048,
                    temperature=0.2
                )
                
                # Extract JSON from response
                scene_data = extract_json_from_text(response)
                
                if not scene_data or "scene" not in scene_data:
                    raise ValueError("Failed to extract valid scene data from LLM response")
                
                # Ensure the scene has the original prompt
                scene = scene_data["scene"]
                scene["prompt"] = prompt
                scene["id"] = prompt_id
                
                # Cache the result
                if self.use_caching:
                    self._cache_result(cache_key, scene)
                
                return scene
                
            except Exception as e:
                logger.error(f"Error in LLM prompt analysis: {str(e)}")
                
                # Use fallback if enabled
                if self.use_fallback:
                    logger.info(f"Using fallback analysis for prompt: {prompt_id}")
                    return self._fallback_analysis(prompt_id, prompt)
                
                # Raise error if fallback not enabled
                raise
    
    @memory_manager.memory_efficient_function
    def _fallback_analysis(self, prompt_id: str, prompt: str) -> Dict[str, Any]:
        """
        Fallback rule-based analysis when LLM fails.
        
        Args:
            prompt_id: Unique identifier for the prompt
            prompt: Text prompt to analyze
            
        Returns:
            Scene representation dictionary
        """
        with Profiler("fallback_analysis"):
            # Extract keywords for object detection
            keywords = self.keyword_extractor.extract_keywords(prompt, max_keywords=5)
            
            # Basic color detection
            colors = self._extract_colors(prompt)
            default_color = {"name": "blue", "hex_code": "#4285F4"} if not colors else {"name": colors[0], "hex_code": self._color_to_hex(colors[0])}
            
            # Create simple objects based on keywords
            objects = []
            for i, keyword in enumerate(keywords):
                # Skip common words that aren't likely to be visual objects
                if keyword in ["background", "scene", "image", "picture", "svg"]:
                    continue
                    
                # Create basic object
                obj = {
                    "id": f"obj_{i}",
                    "name": keyword,
                    "object_type": "generic",
                    "position": [0.3 + (i * 0.15), 0.5],  # Position objects horizontally
                    "size": 0.2,
                    "z_index": i + 1,
                    "color": default_color,
                    "rotation": 0,
                    "shapes": [{"shape_type": "rectangle", "attributes": []}]
                }
                objects.append(obj)
            
            # Create basic scene
            scene = {
                "id": prompt_id,
                "prompt": prompt,
                "width": 800,
                "height": 600,
                "background_color": "#FFFFFF",
                "objects": objects,
                "relationships": []
            }
            
            return scene
    
    def _extract_colors(self, text: str) -> List[str]:
        """
        Extract color references from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of color references
        """
        # Common colors to detect
        common_colors = [
            "red", "green", "blue", "yellow", "orange", "purple", "pink",
            "brown", "black", "white", "gray", "grey", "cyan", "magenta",
            "violet", "teal", "maroon", "navy", "olive", "silver", "gold"
        ]
        
        found_colors = []
        for color in common_colors:
            if re.search(r'\b' + color + r'\b', text.lower()):
                found_colors.append(color)
                
        return found_colors
    
    def _color_to_hex(self, color_name: str) -> str:
        """
        Convert color name to hex code.
        
        Args:
            color_name: Color name
            
        Returns:
            Hex color code
        """
        color_map = {
            "red": "#FF0000",
            "green": "#00FF00",
            "blue": "#0000FF",
            "yellow": "#FFFF00",
            "orange": "#FFA500",
            "purple": "#800080",
            "pink": "#FFC0CB",
            "brown": "#A52A2A",
            "black": "#000000",
            "white": "#FFFFFF",
            "gray": "#808080",
            "grey": "#808080",
            "cyan": "#00FFFF",
            "magenta": "#FF00FF",
            "violet": "#8A2BE2",
            "teal": "#008080",
            "maroon": "#800000",
            "navy": "#000080",
            "olive": "#808000",
            "silver": "#C0C0C0",
            "gold": "#FFD700"
        }
        
        return color_map.get(color_name.lower(), "#4285F4")  # Default to blue
    
    def _create_cache_key(self, prompt: str) -> str:
        """
        Create cache key for a prompt.
        
        Args:
            prompt: Prompt to create key for
            
        Returns:
            Cache key
        """
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()
    
    def _cache_result(self, key: str, result: Dict[str, Any]) -> None:
        """
        Cache analysis result.
        
        Args:
            key: Cache key
            result: Analysis result to cache
        """
        # Manage cache size
        if len(self._cache) >= self._cache_size:
            # Remove random item (simple strategy)
            keys = list(self._cache.keys())
            if keys:
                del self._cache[keys[0]]
        
        # Add to cache
        self._cache[key] = result
    
    @log_function_call()
    def enhance_prompt(self, prompt: str, target_length: int = 150) -> str:
        """
        Enhance a prompt for better SVG generation.
        
        Args:
            prompt: Original prompt
            target_length: Target token length
            
        Returns:
            Enhanced prompt
        """
        with Profiler("prompt_enhancement"):
            system_prompt = """You are an expert SVG designer. Your task is to enhance prompts for SVG image generation.
            Make the prompt more detailed and specific to create better SVG output.
            Focus on visual elements, colors, composition, and styling.
            Be concise and specific about visual details.
            Return only the enhanced prompt without explanations."""
            
            user_prompt = f"""Enhance this prompt for SVG generation:
            
            Original: "{prompt}"
            
            Create a more detailed and specific version with:
            - Clear visual elements
            - Specific colors and styling
            - Improved spatial relationships
            - Better composition guidance
            
            Enhanced prompt:"""
            
            try:
                # Use LLM to enhance prompt
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                response = self.llm_manager.generate(
                    role="prompt_analyzer",
                    prompt=messages,
                    max_tokens=target_length,
                    temperature=0.7
                )
                
                # Clean up response
                enhanced = response.strip()
                
                # Remove quotation marks if present
                if enhanced.startswith('"') and enhanced.endswith('"'):
                    enhanced = enhanced[1:-1]
                
                return enhanced
                
            except Exception as e:
                logger.error(f"Error enhancing prompt: {str(e)}")
                return prompt


# Create singleton instance for easy import
default_prompt_analyzer = LLMPromptAnalyzer()


# Utility functions for direct use
def analyze_prompt(prompt_id: str, prompt: str) -> Dict[str, Any]:
    """
    Analyze a prompt for SVG generation.
    
    Args:
        prompt_id: Unique identifier for the prompt
        prompt: Text prompt to analyze
        
    Returns:
        Scene representation dictionary
    """
    return default_prompt_analyzer.analyze_prompt(prompt_id, prompt)


def enhance_prompt(prompt: str) -> str:
    """
    Enhance a prompt for better SVG generation.
    
    Args:
        prompt: Original prompt
        
    Returns:
        Enhanced prompt
    """
    return default_prompt_analyzer.enhance_prompt(prompt)