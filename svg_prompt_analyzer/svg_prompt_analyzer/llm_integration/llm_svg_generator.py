"""
LLM SVG Generator Module - Optimized
=====================
This module provides functionality for generating SVG code using LLMs with
optimization for visual quality and CLIP similarity scores.
"""

import os
import time
import logging
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple, Set

from svg_prompt_analyzer.generation.svg_generator import SVGGenerator
from svg_prompt_analyzer.llm_integration.llm_manager import LLMManager
from svg_prompt_analyzer.models.scene import Scene

logger = logging.getLogger(__name__)


class LLMSVGGenerator:
    """
    Generator class for producing high-quality SVG code using LLMs with
    optimization for visual fidelity and CLIP similarity.
    """
    
    def __init__(self, 
                 llm_manager: Optional[LLMManager] = None,
                 output_dir: str = "output",
                 use_fallback: bool = True,
                 fallback_threshold: float = 0.6,
                 cache_dir: str = ".cache/svg_generation",
                 max_cache_size: int = 500,
                 use_caching: bool = True,
                 max_svg_size: int = 9500):
        """
        Initialize the LLM-based SVG generator.
        
        Args:
            llm_manager: LLM manager instance for model access
            output_dir: Directory where SVG files will be saved
            use_fallback: Whether to fall back to traditional generator if LLM fails
            fallback_threshold: Confidence threshold below which fallback is triggered
            cache_dir: Directory for generated SVG caching
            max_cache_size: Maximum number of cached SVGs to keep
            use_caching: Whether to cache generated SVGs
            max_svg_size: Maximum size (in bytes) for generated SVG code
        """
        self.llm_manager = llm_manager or LLMManager()
        self.original_generator = SVGGenerator(output_dir=output_dir) if use_fallback else None
        self.use_fallback = use_fallback
        self.fallback_threshold = fallback_threshold
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.use_caching = use_caching
        self.max_svg_size = max_svg_size
        
        # Create output and cache directories
        os.makedirs(output_dir, exist_ok=True)
        if self.use_caching:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Memory cache for recent generations
        self._svg_cache = {}
        
        # Track model loading status
        self.model_loaded = False
        
    def generate_svg(self, scene: Scene) -> str:
        """
        Generate SVG code for a scene using LLM with fallback to traditional generation.
        
        Args:
            scene: Scene object containing all visual elements
            
        Returns:
            SVG code as string
        """
        # Check cache
        if self.use_caching:
            cache_key = self._get_cache_key(scene)
            # Check memory cache
            if cache_key in self._svg_cache:
                logger.info(f"Using cached SVG for scene {scene.id}")
                return self._svg_cache[cache_key]
                
            # Check disk cache
            cached_svg = self._load_from_cache(cache_key)
            if cached_svg:
                logger.info(f"Loaded cached SVG from disk for scene {scene.id}")
                # Update memory cache
                self._svg_cache[cache_key] = cached_svg
                return cached_svg
        
        logger.info(f"Generating SVG for scene {scene.id} with LLM")
        
        # If LLM is not available and fallback is disabled, raise an error
        if not self.model_loaded and not self.llm_manager.load_model("svg_generator") and not self.use_fallback:
            raise RuntimeError("LLM not available for SVG generation and fallback is disabled")
            
        # If LLM is not available but fallback is enabled, use traditional generator
        if not self.model_loaded and not self.llm_manager.load_model("svg_generator") and self.use_fallback:
            logger.info(f"Using fallback generator for scene {scene.id}")
            svg_code = self.original_generator.generate_svg(scene)
            return svg_code
            
        try:
            # Generate SVG using LLM
            start_time = time.time()
            
            # Prepare scene data as input for the LLM
            llm_input = self._create_generation_prompt(scene)
            
            # Get response from LLM
            llm_response = self.llm_manager.generate(
                role="svg_generator",
                prompt=llm_input,
                max_tokens=self.max_svg_size * 2,  # Ensure enough tokens for full SVG
                temperature=0.2,  # Low temperature for deterministic generation
                stop_sequences=["```"]  # Stop when SVG code block ends
            )
            
            # Extract SVG code from LLM response
            svg_code = self._extract_svg_code(llm_response)
            
            # Validate the SVG code
            is_valid, confidence = self._validate_svg_code(svg_code)
            
            if not is_valid or confidence < self.fallback_threshold:
                if self.use_fallback:
                    logger.warning(f"LLM generated invalid SVG (confidence: {confidence:.2f}). Using fallback.")
                    svg_code = self.original_generator.generate_svg(scene)
                else:
                    logger.warning(f"LLM generated low confidence SVG ({confidence:.2f}). Using as-is.")
            
            generation_time = time.time() - start_time
            logger.info(f"Generated SVG for scene {scene.id} in {generation_time:.2f}s (confidence: {confidence:.2f})")
            
            # Ensure the SVG size doesn't exceed the maximum
            if len(svg_code) > self.max_svg_size:
                logger.warning(f"Generated SVG exceeds max size ({len(svg_code)} > {self.max_svg_size}). Truncating.")
                svg_code = self._truncate_svg_code(svg_code)
            
            # Cache the result
            if self.use_caching:
                self._save_to_cache(cache_key, svg_code)
            
            return svg_code
            
        except Exception as e:
            logger.error(f"Error in LLM SVG generation: {str(e)}")
            
            # Fall back to traditional generator if enabled
            if self.use_fallback:
                logger.info(f"Falling back to traditional generator for scene {scene.id}")
                svg_code = self.original_generator.generate_svg(scene)
                return svg_code
            else:
                raise
                
    def save_svg(self, scene: Scene) -> str:
        """
        Generate SVG and save it to a file.
        
        Args:
            scene: Scene object containing all visual elements
            
        Returns:
            Path to the saved SVG file
        """
        svg_code = self.generate_svg(scene)
        
        # Create filename
        filename = f"{scene.id}.svg"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(svg_code)
        
        logger.info(f"SVG saved to {filepath}")
        
        return filepath
        
    def _create_generation_prompt(self, scene: Scene) -> str:
        """
        Create a structured prompt for the LLM to generate SVG code.
        
        Args:
            scene: Scene object containing visual elements
            
        Returns:
            Formatted prompt for the LLM
        """
        # Extract key information from the scene
        objects_info = []
        for obj in scene.objects:
            obj_info = {
                "name": obj.name,
                "type": obj.object_type.value,
                "color": obj.color.name if obj.color else None,
                "color_hex": obj.color.hex_code if obj.color else None,
                "material": obj.material.name if obj.material else None,
                "shape": obj.shapes[0].shape_type if obj.shapes else None,
                "position": [f"{p:.2f}" for p in obj.position],
                "size": f"{obj.size:.2f}",
                "z_index": obj.z_index
            }
            objects_info.append(obj_info)
            
        scene_info = {
            "id": scene.id,
            "prompt": scene.prompt,
            "background_color": scene.background_color,
            "width": scene.width,
            "height": scene.height,
            "objects": objects_info
        }
        
        scene_json = json.dumps(scene_info, indent=2)
        
        # Create a detailed prompt for SVG generation
        prompt = f"""You are an expert SVG illustrator specialized in creating high-quality SVG code from scene descriptions. Your task is to generate SVG code for the following scene.

IMPORTANT GUIDELINES:
1. Create clean, semantically meaningful SVG code optimized for visual clarity
2. Use gradients, patterns, and visual effects where appropriate
3. Pay special attention to colors, shapes, and spatial relationships
4. Generate complete, valid SVG code that renders correctly
5. Focus on visual fidelity to the original prompt
6. Include descriptive comments in the SVG code
7. Ensure the SVG code is optimized for CLIP similarity scoring

Here is the detailed scene information in JSON format:
{scene_json}

Please generate complete, well-formed SVG code for this scene. The SVG should be semantically rich, visually appealing, and directly correspond to the original prompt.

SVG code:
```xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{scene.width}" height="{scene.height}" viewBox="0 0 {scene.width} {scene.height}"
    xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>SVG illustration of {scene.prompt}</title>
    <desc>Generated from prompt: {scene.prompt}</desc>
    
    <!-- Definitions for patterns, gradients, etc. -->
    <defs>
"""
        
        return prompt
        
    def _extract_svg_code(self, llm_response: str) -> str:
        """
        Extract SVG code from LLM response.
        
        Args:
            llm_response: Raw response from the LLM
            
        Returns:
            Extracted SVG code
        """
        # Look for SVG code in code blocks
        svg_match = re.search(r'```(?:xml|svg)?\s*((?:<\?xml|<svg).*?</svg>)', llm_response, re.DOTALL)
        if svg_match:
            return svg_match.group(1).strip()
            
        # If no explicit code block, try to find SVG tags directly
        svg_match = re.search(r'(?:<\?xml|<svg).*?</svg>', llm_response, re.DOTALL)
        if svg_match:
            return svg_match.group(0).strip()
            
        # If still no match, check if response starts with XML declaration or SVG tag
        if llm_response.strip().startswith(('<?xml', '<svg')):
            # Find the closing SVG tag
            if '</svg>' in llm_response:
                end_index = llm_response.rindex('</svg>') + 6
                return llm_response[:end_index].strip()
                
        # Fallback: Return a minimal valid SVG with error message
        logger.warning("Could not extract valid SVG code from LLM response")
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="800" height="600" viewBox="0 0 800 600"
    xmlns="http://www.w3.org/2000/svg">
    <title>Error in SVG generation</title>
    <desc>Failed to generate SVG from LLM response</desc>
    <rect width="100%" height="100%" fill="#F8F8F8" />
    <text x="400" y="300" text-anchor="middle" font-family="Arial" font-size="20" fill="#FF0000">
        Error: Failed to generate SVG
    </text>
</svg>"""
        
    def _validate_svg_code(self, svg_code: str) -> Tuple[bool, float]:
        """
        Validate the generated SVG code and compute confidence score.
        
        Args:
            svg_code: SVG code to validate
            
        Returns:
            Tuple of (is_valid, confidence_score)
        """
        # Check basic validity
        if not svg_code or not svg_code.strip():
            return False, 0.0
            
        if not svg_code.strip().startswith(('<?xml', '<svg')):
            return False, 0.0
            
        if '</svg>' not in svg_code:
            return False, 0.0
            
        # Compute confidence score
        confidence = 0.5  # Start with base confidence
        
        # Check for essential SVG elements
        if '<title>' in svg_code:
            confidence += 0.1
            
        if '<desc>' in svg_code:
            confidence += 0.1
            
        if '<defs>' in svg_code:
            confidence += 0.1
            
        # Count visual elements
        element_count = sum(1 for tag in ['<rect', '<circle', '<path', '<polygon', '<line', '<ellipse'] 
                            if tag in svg_code)
        
        # More elements increase confidence
        if element_count > 0:
            confidence += min(0.1 * element_count, 0.5)
            
        # Check for comments (sign of well-structured SVG)
        comment_count = svg_code.count('<!--')
        if comment_count > 0:
            confidence += min(0.05 * comment_count, 0.2)
            
        # Check for proper nesting
        if svg_code.count('<') == svg_code.count('>'):
            confidence += 0.1
            
        # Cap confidence at 1.0
        return True, min(confidence, 1.0)
        
    def _truncate_svg_code(self, svg_code: str) -> str:
        """
        Truncate SVG code to stay within size limits while maintaining validity.
        
        Args:
            svg_code: Original SVG code
            
        Returns:
            Truncated SVG code
        """
        # If already within limits, return as-is
        if len(svg_code) <= self.max_svg_size:
            return svg_code
            
        # Find SVG opening and closing tags
        svg_start_match = re.search(r'<svg\s[^>]*>', svg_code)
        if not svg_start_match:
            logger.error("Could not find SVG opening tag for truncation")
            return self._get_error_svg("SVG truncation failed - no opening tag")
            
        svg_start = svg_start_match.start()
        svg_end = svg_code.rindex('</svg>') + 6 if '</svg>' in svg_code else len(svg_code)
        
        # Extract header (everything before SVG content)
        header = svg_code[:svg_start_match.end()]
        
        # Extract SVG content
        content = svg_code[svg_start_match.end():svg_end - 6]  # Exclude closing </svg> tag
        
        # Count elements by finding all tag starts
        element_tags = re.findall(r'<([a-zA-Z]+)[^>]*>', content)
        element_count = len(element_tags)
        
        # If there are too many elements, keep only a subset
        if element_count > 0:
            # Parse content into elements (simplistic approach)
            elements = []
            depth = 0
            current_element = ""
            
            for char in content:
                current_element += char
                if char == '<':
                    depth += 1
                elif char == '>':
                    depth -= 1
                    if depth == 0:
                        elements.append(current_element)
                        current_element = ""
            
            # Calculate how many elements we can keep
            # Preserve critical elements (defs, etc.)
            critical_elements = [e for e in elements if any(tag in e.lower() for tag in ['defs', 'gradient', 'filter'])]
            regular_elements = [e for e in elements if e not in critical_elements]
            
            # Find a number of elements that will fit
            total_size = len(header) + len('</svg>') + sum(len(e) for e in critical_elements)
            
            # Add as many regular elements as will fit
            elements_to_keep = critical_elements[:]
            for element in regular_elements:
                if total_size + len(element) < self.max_svg_size - 100:  # Leave some margin
                    elements_to_keep.append(element)
                    total_size += len(element)
                else:
                    break
                    
            # Reconstruct truncated SVG
            truncated_svg = header + ''.join(elements_to_keep) + '</svg>'
            
            logger.info(f"Truncated SVG from {len(svg_code)} to {len(truncated_svg)} bytes "
                      f"({len(elements_to_keep)}/{element_count} elements)")
            
            return truncated_svg
        else:
            # Simplistic truncation if we can't identify elements
            truncated_svg = svg_code[:self.max_svg_size - 7] + '</svg>'
            logger.info(f"Basic truncation from {len(svg_code)} to {len(truncated_svg)} bytes")
            return truncated_svg
            
    def _get_error_svg(self, error_message: str) -> str:
        """Generate a simple error SVG with the provided message."""
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="800" height="600" viewBox="0 0 800 600"
    xmlns="http://www.w3.org/2000/svg">
    <title>Error in SVG generation</title>
    <desc>{error_message}</desc>
    <rect width="100%" height="100%" fill="#F8F8F8" />
    <text x="400" y="300" text-anchor="middle" font-family="Arial" font-size="20" fill="#FF0000">
        Error: {error_message}
    </text>
</svg>"""
        
    def _get_cache_key(self, scene: Scene) -> str:
        """Generate a cache key for the scene."""
        import hashlib
        # Create a deterministic hash of the scene information
        scene_info = f"{scene.id}:{scene.prompt}:{scene.background_color}"
        for obj in scene.objects:
            scene_info += f":{obj.name}:{obj.object_type.value}"
            if obj.color:
                scene_info += f":{obj.color.hex_code}"
        return hashlib.md5(scene_info.encode()).hexdigest()
        
    def _save_to_cache(self, cache_key: str, svg_code: str) -> None:
        """Save generated SVG to both memory and disk cache."""
        try:
            # Update memory cache
            self._svg_cache[cache_key] = svg_code
            
            # Limit memory cache size
            if len(self._svg_cache) > self.max_cache_size:
                # Remove oldest key
                oldest_key = next(iter(self._svg_cache))
                self._svg_cache.pop(oldest_key, None)
                
            # Save to disk cache
            if self.use_caching:
                cache_path = Path(self.cache_dir) / f"{cache_key}.svg"
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(svg_code)
                    
        except Exception as e:
            logger.error(f"Error saving SVG to cache: {str(e)}")
            
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load SVG from disk cache."""
        try:
            cache_path = Path(self.cache_dir) / f"{cache_key}.svg"
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return None
        except Exception as e:
            logger.error(f"Error loading SVG from cache: {str(e)}")
            return None
            
    def clear_cache(self) -> None:
        """Clear both memory and disk caches."""
        # Clear memory cache
        self._svg_cache.clear()
        
        # Clear disk cache
        if self.use_caching:
            try:
                for cache_file in Path(self.cache_dir).glob("*.svg"):
                    try:
                        cache_file.unlink()
                    except:
                        pass
            except Exception as e:
                logger.error(f"Error clearing disk cache: {str(e)}")