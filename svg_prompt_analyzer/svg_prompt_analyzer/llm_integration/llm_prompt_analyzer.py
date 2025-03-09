"""
LLM Prompt Analyzer Module - Optimized
========================
This module uses an LLM to analyze text prompts and extract detailed information
for SVG generation with enhanced semantic understanding, optimized for performance
across different hardware platforms.
"""

import json
import logging
import re
import gc
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from svg_prompt_analyzer.analysis.prompt_analyzer import PromptAnalyzer
from svg_prompt_analyzer.llm_integration.llm_manager import LLMManager
from svg_prompt_analyzer.models.scene import Scene
from svg_prompt_analyzer.models.visual_object import VisualObject, ObjectType
from svg_prompt_analyzer.models.color import Color
from svg_prompt_analyzer.models.material import Material
from svg_prompt_analyzer.models.shape import Shape, Attribute
from svg_prompt_analyzer.models.spatial import SpatialRelation

logger = logging.getLogger(__name__)

class LLMPromptAnalyzer:
    """
    Enhanced prompt analyzer that uses an LLM to extract detailed information from text prompts.
    This class augments the existing PromptAnalyzer with deeper semantic understanding,
    optimized for cross-platform performance.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for resource efficiency."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LLMPromptAnalyzer, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, 
                 llm_manager: Optional[LLMManager] = None,
                 use_fallback: bool = True,
                 fallback_threshold: float = 0.7,
                 cache_dir: str = ".cache/prompt_analysis",
                 max_cache_size: int = 500,
                 use_caching: bool = True,
                 num_workers: int = 4,
                 preload_model: bool = False):
        """
        Initialize the LLM-based prompt analyzer.
        
        Args:
            llm_manager: LLM manager instance for model access
            use_fallback: Whether to fall back to traditional analyzer if LLM fails
            fallback_threshold: Confidence threshold below which fallback is triggered
            cache_dir: Directory for analysis result caching
            max_cache_size: Maximum number of cached analyses to keep
            use_caching: Whether to cache analysis results
            num_workers: Number of worker threads for parallel processing
            preload_model: Whether to preload the LLM model at initialization
        """
        # Initialize only once (singleton pattern)
        if hasattr(self, 'initialized'):
            return
            
        self.llm_manager = llm_manager or LLMManager()
        self.original_analyzer = PromptAnalyzer() if use_fallback else None
        self.use_fallback = use_fallback
        self.fallback_threshold = fallback_threshold
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.use_caching = use_caching
        self.num_workers = min(num_workers, 16)  # Cap at reasonable maximum
        
        # Create cache directory
        if self.use_caching:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Memory cache for recent analyses
        self._analysis_cache = {}
        
        # Track model loading status
        self.model_loaded = False
        
        # Thread pool for parallel tasks
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        # Load the LLM model for prompt analysis if requested
        if preload_model:
            self.model_loaded = self.llm_manager.load_model("prompt_analyzer")
            if not self.model_loaded:
                logger.warning("Failed to preload LLM for prompt analysis. Will use fallback if enabled.")
        
        # Flag initialization complete
        self.initialized = True
        logger.info(f"LLM Prompt Analyzer initialized with fallback: {use_fallback}")
        
    def analyze_prompt(self, prompt_id: str, prompt_text: str) -> Scene:
        """
        Analyze a prompt using LLM-enhanced understanding to extract detailed scene information.
        
        Args:
            prompt_id: Identifier for the prompt
            prompt_text: Text of the prompt to analyze
            
        Returns:
            Scene object containing all extracted information
        """
        # Check memory cache first
        cache_key = self._get_cache_key(prompt_id, prompt_text)
        if self.use_caching and cache_key in self._analysis_cache:
            logger.info(f"Using cached analysis for prompt {prompt_id}")
            return self._analysis_cache[cache_key]
            
        # Check disk cache if memory cache missed
        if self.use_caching:
            cached_scene = self._load_from_cache(cache_key)
            if cached_scene:
                logger.info(f"Loaded cached analysis from disk for prompt {prompt_id}")
                # Update memory cache
                self._analysis_cache[cache_key] = cached_scene
                return cached_scene
        
        logger.info(f"Analyzing prompt {prompt_id} with LLM: {prompt_text}")
        
        # If LLM is not available and fallback is disabled, raise an error
        if not self.model_loaded and not self.llm_manager.load_model("prompt_analyzer") and not self.use_fallback:
            raise RuntimeError("LLM not available for prompt analysis and fallback is disabled")
            
        # If LLM is not available but fallback is enabled, use traditional analyzer
        if not self.model_loaded and not self.llm_manager.load_model("prompt_analyzer") and self.use_fallback:
            logger.info(f"Using fallback analyzer for prompt {prompt_id}")
            scene = self.original_analyzer.analyze_prompt(prompt_id, prompt_text)
            return scene
            
        start_time = time.time()
        try:
            # Generate the LLM prompt for scene analysis
            llm_input = self._create_analysis_prompt(prompt_text)
            
            # Get response from LLM
            llm_response = self.llm_manager.generate(
                role="prompt_analyzer",
                prompt=llm_input,
                max_tokens=2048,
                temperature=0.2,  # Low temperature for more deterministic analysis
                stop_sequences=["</scene_analysis>"]
            )
            
            # Parse the LLM response
            scene_data = self._parse_llm_response(llm_response)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(scene_data)
            
            if confidence < self.fallback_threshold and self.use_fallback:
                logger.warning(f"Low confidence in LLM analysis ({confidence:.2f}). Using fallback.")
                scene = self.original_analyzer.analyze_prompt(prompt_id, prompt_text)
            else:
                # Create scene from parsed data
                scene = self._create_scene(prompt_id, prompt_text, scene_data)
                
            analysis_time = time.time() - start_time
            logger.info(f"Successfully analyzed prompt {prompt_id} in {analysis_time:.2f}s (confidence: {confidence:.2f})")
            
            # Cache the result
            if self.use_caching:
                self._save_to_cache(cache_key, scene)
            
            return scene
            
        except Exception as e:
            logger.error(f"Error in LLM prompt analysis: {str(e)}")
            
            # Fall back to traditional analyzer if enabled
            if self.use_fallback:
                logger.info(f"Falling back to traditional analyzer for prompt {prompt_id}")
                return self.original_analyzer.analyze_prompt(prompt_id, prompt_text)
            else:
                raise
                
    def batch_analyze(self, prompt_data: List[Dict[str, str]], 
                      use_parallel: bool = True) -> Dict[str, Scene]:
        """
        Analyze multiple prompts efficiently.
        
        Args:
            prompt_data: List of dictionaries with 'id' and 'description' keys
            use_parallel: Whether to process prompts in parallel
            
        Returns:
            Dictionary mapping prompt IDs to Scene objects
        """
        results = {}
        
        if use_parallel and len(prompt_data) > 1:
            # Process in parallel
            futures = {}
            
            for item in prompt_data:
                prompt_id = item['id']
                prompt_text = item['description']
                
                # Submit analysis task
                future = self._executor.submit(self.analyze_prompt, prompt_id, prompt_text)
                futures[future] = prompt_id
                
            # Collect results as they complete
            for future in futures:
                prompt_id = futures[future]
                try:
                    scene = future.result()
                    results[prompt_id] = scene
                except Exception as e:
                    logger.error(f"Error analyzing prompt {prompt_id}: {str(e)}")
                    # Use fallback for errors if enabled
                    if self.use_fallback:
                        logger.info(f"Using fallback for failed prompt {prompt_id}")
                        results[prompt_id] = self.original_analyzer.analyze_prompt(prompt_id, prompt_data)
        else:
            # Process sequentially
            for item in prompt_data:
                prompt_id = item['id']
                prompt_text = item['description']
                
                try:
                    scene = self.analyze_prompt(prompt_id, prompt_text)
                    results[prompt_id] = scene
                except Exception as e:
                    logger.error(f"Error analyzing prompt {prompt_id}: {str(e)}")
                    # Use fallback for errors if enabled
                    if self.use_fallback:
                        logger.info(f"Using fallback for failed prompt {prompt_id}")
                        results[prompt_id] = self.original_analyzer.analyze_prompt(prompt_id, prompt_text)
                        
        return results
                
    def _create_analysis_prompt(self, prompt_text: str) -> str:
        """
        Create a structured prompt for the LLM to analyze.
        
        Args:
            prompt_text: Original prompt text
            
        Returns:
            Formatted prompt for the LLM
        """
        return f"""You are an expert visual scene analyzer specialized in SVG generation. Your task is to analyze a text prompt and extract detailed information to create an SVG illustration.

Here's how you should analyze the prompt:
1. Identify all key objects in the prompt.
2. Determine their visual properties (colors, materials, shapes, sizes).
3. Understand spatial relationships between objects.
4. Define the overall scene composition and background.
5. Identify any special visual effects or patterns.

For each object, provide:
- Type: The category of object (geometric, nature, clothing, architecture, etc.)
- Color: Detailed color information including hex codes when possible
- Material: Any material properties mentioned or implied
- Shape: Geometric shape or form of the object
- Size: Relative size in the scene
- Position: Where the object should be placed
- Visual effects: Special visual characteristics

Prompt to analyze: "{prompt_text}"

Analyze this thoroughly and output in JSON format with the following structure:
{{
  "objects": [
    {{
      "name": "object name",
      "type": "object type",
      "color": "color name",
      "hex_code": "#RRGGBB",
      "material": "material name",
      "shape": "shape name",
      "position": "position description",
      "size": "size description",
      "visual_effects": "effects description"
    }}
  ],
  "background": {{
    "color": "background color",
    "hex_code": "#RRGGBB"
  }},
  "composition": {{
    "description": "scene composition description"
  }},
  "special_effects": ["effect1", "effect2"]
}}

<scene_analysis>
"""
        
    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured data.
        
        Args:
            llm_response: Raw response from the LLM
            
        Returns:
            Structured dictionary of scene information
        """
        # Ensure we only process content between tags if present
        if "<scene_analysis>" in llm_response and "</scene_analysis>" in llm_response:
            content = re.search(r'<scene_analysis>(.*?)</scene_analysis>', 
                               llm_response, re.DOTALL)
            if content:
                llm_response = content.group(1).strip()
        
        # Try to find and parse JSON in the response
        json_match = re.search(r'```json(.*?)```', llm_response, re.DOTALL) or re.search(r'```(.*?)```', llm_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from LLM response")
        
        # Try to find JSON without code blocks
        try:
            # Look for JSON-like structure
            json_like = re.search(r'(\{.*\})', llm_response, re.DOTALL)
            if json_like:
                potential_json = json_like.group(1)
                return json.loads(potential_json)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Failed to parse JSON-like structure from LLM response")
        
        # If no JSON found or parsing failed, extract structured data from text
        try:
            return self._extract_structured_data(llm_response)
        except Exception as e:
            logger.error(f"Failed to extract structured data: {str(e)}")
            return {"error": "Failed to parse LLM response", "raw_response": llm_response}
            
    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data from text response when JSON parsing fails.
        
        Args:
            text: Text to parse
            
        Returns:
            Structured dictionary
        """
        result = {
            "objects": [],
            "background": {},
            "composition": {},
            "special_effects": []
        }
        
        # Extract background information
        bg_match = re.search(r'Background:?\s*(.*?)(?:\n\n|\n\w+:)', text, re.DOTALL)
        if bg_match:
            bg_text = bg_match.group(1).strip()
            # Extract color
            color_match = re.search(r'colou?r:?\s*([^,\n]+)', bg_text, re.IGNORECASE)
            if color_match:
                result["background"]["color"] = color_match.group(1).strip()
            # Extract hex code if present
            hex_match = re.search(r'#[0-9A-Fa-f]{6}', bg_text)
            if hex_match:
                result["background"]["hex_code"] = hex_match.group(0)
        
        # Extract objects
        object_blocks = re.finditer(r'Object\s*\d*:?\s*(.*?)(?:\n\n|\n(?:Object|Background|Composition|Special Effects):)', 
                                    text + "\n\nEND", re.DOTALL)
        
        for block in object_blocks:
            obj_text = block.group(1).strip()
            obj = {}
            
            # Extract object name
            name_match = re.search(r'Name:?\s*([^,\n]+)', obj_text, re.IGNORECASE)
            if name_match:
                obj["name"] = name_match.group(1).strip()
            else:
                # Try to infer name from first line
                first_line = obj_text.split('\n')[0].strip()
                if ':' in first_line:
                    obj["name"] = first_line.split(':', 1)[1].strip()
                else:
                    obj["name"] = first_line
            
            # Extract object type
            type_match = re.search(r'Type:?\s*([^,\n]+)', obj_text, re.IGNORECASE)
            if type_match:
                obj["type"] = type_match.group(1).strip()
                
            # Extract color
            color_match = re.search(r'Colou?r:?\s*([^,\n]+)', obj_text, re.IGNORECASE)
            if color_match:
                obj["color"] = color_match.group(1).strip()
                # Look for hex code
                hex_match = re.search(r'#[0-9A-Fa-f]{6}', obj_text)
                if hex_match:
                    obj["hex_code"] = hex_match.group(0)
                    
            # Extract material
            material_match = re.search(r'Material:?\s*([^,\n]+)', obj_text, re.IGNORECASE)
            if material_match:
                obj["material"] = material_match.group(1).strip()
                
            # Extract shape
            shape_match = re.search(r'Shape:?\s*([^,\n]+)', obj_text, re.IGNORECASE)
            if shape_match:
                obj["shape"] = shape_match.group(1).strip()
                
            # Extract position
            position_match = re.search(r'Position:?\s*([^,\n]+)', obj_text, re.IGNORECASE)
            if position_match:
                obj["position"] = position_match.group(1).strip()
                
            # Extract size
            size_match = re.search(r'Size:?\s*([^,\n]+)', obj_text, re.IGNORECASE)
            if size_match:
                obj["size"] = size_match.group(1).strip()
                
            # Extract visual effects
            effects_match = re.search(r'(Visual )?Effects:?\s*([^,\n]+)', obj_text, re.IGNORECASE)
            if effects_match:
                obj["visual_effects"] = effects_match.group(2).strip()
                
            # Add object to result if it has a name
            if "name" in obj:
                result["objects"].append(obj)
                
        # Extract composition
        comp_match = re.search(r'Composition:?\s*(.*?)(?:\n\n|\n\w+:)', text, re.DOTALL)
        if comp_match:
            result["composition"]["description"] = comp_match.group(1).strip()
            
        # Extract special effects
        effects_match = re.search(r'Special Effects:?\s*(.*?)(?:\n\n|$)', text, re.DOTALL)
        if effects_match:
            effects_text = effects_match.group(1).strip()
            effects_list = [effect.strip() for effect in re.split(r'[,;-]\s*|\n+', effects_text) if effect.strip()]
            result["special_effects"] = effects_list
            
        return result
        
    def _calculate_confidence(self, scene_data: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the LLM analysis.
        
        Args:
            scene_data: Parsed scene data from LLM
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.0
        
        # Check for errors
        if "error" in scene_data:
            return 0.0
            
        # More objects increases confidence
        num_objects = len(scene_data.get("objects", []))
        confidence += min(num_objects * 0.1, 0.5)
        
        # Check for essential object data
        objects_with_color = sum(1 for obj in scene_data.get("objects", []) if "color" in obj)
        objects_with_shape = sum(1 for obj in scene_data.get("objects", []) if "shape" in obj)
        
        if num_objects > 0:
            color_ratio = objects_with_color / num_objects
            shape_ratio = objects_with_shape / num_objects
            confidence += color_ratio * 0.2
            confidence += shape_ratio * 0.2
            
        # Background information increases confidence
        if scene_data.get("background", {}).get("color"):
            confidence += 0.1
            
        # Cap confidence at 1.0
        return min(confidence, 1.0)
        
    def _create_scene(self, prompt_id: str, prompt_text: str, scene_data: Dict[str, Any]) -> Scene:
        """
        Create a Scene object from the parsed LLM response.
        
        Args:
            prompt_id: Identifier for the prompt
            prompt_text: Original prompt text
            scene_data: Parsed scene data from LLM
            
        Returns:
            Scene object with extracted information
        """
        # Create scene with background
        background_color = "#FFFFFF"  # Default white
        if "background" in scene_data and "hex_code" in scene_data["background"]:
            background_color = scene_data["background"]["hex_code"]
        elif "background" in scene_data and "color" in scene_data["background"]:
            color_name = scene_data["background"]["color"]
            color = Color(name=color_name)
            background_color = color.hex_code
            
        scene = Scene(
            id=prompt_id,
            prompt=prompt_text,
            background_color=background_color,
            objects=[]
        )
        
        # Add objects to scene using thread pool for parallel processing
        if len(scene_data.get("objects", [])) > 5:
            # Process objects in parallel for larger scenes
            futures = {}
            for i, obj_data in enumerate(scene_data.get("objects", [])):
                obj_id = f"{prompt_id}_{i}"
                futures[self._executor.submit(self._create_visual_object, obj_id, obj_data)] = i
                
            # Collect results and add to scene in original order
            objects = [None] * len(futures)
            for future in futures:
                i = futures[future]
                try:
                    objects[i] = future.result()
                except Exception as e:
                    logger.error(f"Error creating visual object {i}: {str(e)}")
                    
            # Filter out None values (failed object creations)
            scene.objects = [obj for obj in objects if obj is not None]
        else:
            # Process sequentially for smaller scenes
            for i, obj_data in enumerate(scene_data.get("objects", [])):
                visual_object = self._create_visual_object(f"{prompt_id}_{i}", obj_data)
                scene.objects.append(visual_object)
            
        # Add patterns if needed
        scene.patterns = self._create_patterns_for_scene(scene)
        
        # Apply special effects if any
        for effect in scene_data.get("special_effects", []):
            if "gradient" in effect.lower():
                scene.defs.append(self._create_gradient_def())
            elif "pattern" in effect.lower():
                scene.defs.append(self._create_pattern_def())
                
        return scene
        
    def _create_visual_object(self, obj_id: str, obj_data: Dict[str, Any]) -> VisualObject:
        """
        Create a VisualObject from object data.
        
        Args:
            obj_id: Identifier for the object
            obj_data: Object data from parsed LLM response
            
        Returns:
            VisualObject instance
        """
        # Determine object type
        obj_type = ObjectType.ABSTRACT
        if "type" in obj_data:
            type_text = obj_data["type"].lower()
            for enum_type in ObjectType:
                if enum_type.value in type_text:
                    obj_type = enum_type
                    break
                    
        # Create color if present
        color = None
        if "color" in obj_data:
            color_name = obj_data["color"]
            hex_code = obj_data.get("hex_code")
            color = Color(name=color_name, hex_code=hex_code)
            
        # Create material if present
        material = None
        if "material" in obj_data:
            material_name = obj_data["material"]
            material = Material(name=material_name, texture=material_name)
            
        # Create shape if present
        shapes = []
        if "shape" in obj_data:
            shape_type = obj_data["shape"].lower()
            shape = Shape(
                shape_type=shape_type,
                attributes=[
                    Attribute("fill", color.hex_code if color else "#808080"),
                    Attribute("stroke", "#000000"),
                    Attribute("stroke-width", 1)
                ]
            )
            shapes.append(shape)
            
        # Determine position
        position = (0.5, 0.5)  # Default center
        if "position" in obj_data:
            position_text = obj_data["position"].lower()
            if "top" in position_text and "left" in position_text:
                position = (0.25, 0.25)
            elif "top" in position_text and "right" in position_text:
                position = (0.75, 0.25)
            elif "bottom" in position_text and "left" in position_text:
                position = (0.25, 0.75)
            elif "bottom" in position_text and "right" in position_text:
                position = (0.75, 0.75)
            elif "top" in position_text:
                position = (0.5, 0.25)
            elif "bottom" in position_text:
                position = (0.5, 0.75)
            elif "left" in position_text:
                position = (0.25, 0.5)
            elif "right" in position_text:
                position = (0.75, 0.5)
                
        # Determine size
        size = 0.2  # Default size
        if "size" in obj_data:
            size_text = obj_data["size"].lower()
            if "large" in size_text or "big" in size_text:
                size = 0.4
            elif "small" in size_text or "tiny" in size_text:
                size = 0.1
            elif "medium" in size_text:
                size = 0.2
                
        # Create visual object
        visual_object = VisualObject(
            id=obj_id,
            name=obj_data.get("name", "Unknown Object"),
            object_type=obj_type,
            shapes=shapes,
            color=color,
            material=material,
            position=position,
            size=size
        )
        
        # Add visual effects if present
        if "visual_effects" in obj_data:
            effects_text = obj_data["visual_effects"].lower()
            if "glow" in effects_text:
                visual_object.add_visual_effect("glow")
            if "shimmer" in effects_text:
                visual_object.add_visual_effect("shimmering")
            if "shadow" in effects_text:
                visual_object.add_visual_effect("shadow")
            if "gradient" in effects_text:
                visual_object.add_visual_effect("gradient")
            if "pattern" in effects_text:
                visual_object.add_visual_effect("pattern")
            if "texture" in effects_text:
                visual_object.add_visual_effect("texture")
                
        return visual_object
        
    def _create_patterns_for_scene(self, scene: Scene) -> Dict[str, str]:
        """
        Create patterns needed for the scene.
        
        Args:
            scene: Scene object
            
        Returns:
            Dictionary mapping pattern IDs to pattern code
        """
        patterns = {}
        materials_seen: Set[str] = set()
        effects_seen: Set[str] = set()
        
        # Process all objects to collect unique materials and effects
        for obj in scene.objects:
            # Check for materials
            if obj.material and obj.material.texture:
                materials_seen.add(obj.material.texture.lower())
                
            # Check for visual effects
            if hasattr(obj, 'visual_effects'):
                for effect in obj.visual_effects:
                    effects_seen.add(effect)
        
        # Create patterns for unique materials
        for material in materials_seen:
            if "silk" in material and "patternSilk" not in patterns:
                patterns["patternSilk"] = self._create_silk_pattern(
                    self._get_color_for_material(scene, "silk")
                )
            elif "wool" in material and "patternWool" not in patterns:
                patterns["patternWool"] = self._create_wool_pattern(
                    self._get_color_for_material(scene, "wool")
                )
            elif "corduroy" in material and "patternCorduroy" not in patterns:
                patterns["patternCorduroy"] = self._create_corduroy_pattern(
                    self._get_color_for_material(scene, "corduroy")
                )
            elif "fur" in material and "patternFur" not in patterns:
                patterns["patternFur"] = self._create_fur_pattern(
                    self._get_color_for_material(scene, "fur")
                )
        
        # Create patterns for effects
        for effect in effects_seen:
            if effect == "shimmering" and "patternShimmering" not in patterns:
                patterns["patternShimmering"] = self._create_shimmering_pattern(
                    self._get_color_for_effect(scene, "shimmering")
                )
            elif effect == "ribbed" and "patternRibbed" not in patterns:
                patterns["patternRibbed"] = self._create_ribbed_pattern(
                    self._get_color_for_effect(scene, "ribbed")
                )
        
        return patterns
    
    def _get_color_for_material(self, scene: Scene, material_name: str) -> str:
        """
        Get a color to use for a material pattern from objects using that material.
        
        Args:
            scene: Scene object
            material_name: Material to find color for
            
        Returns:
            Hex color code
        """
        for obj in scene.objects:
            if obj.material and material_name in obj.material.texture.lower() and obj.color:
                return obj.color.hex_code
        return "#FFFFFF"  # Default white
    
    def _get_color_for_effect(self, scene: Scene, effect_name: str) -> str:
        """
        Get a color to use for an effect pattern.
        
        Args:
            scene: Scene object
            effect_name: Effect to find color for
            
        Returns:
            Hex color code
        """
        for obj in scene.objects:
            if hasattr(obj, 'visual_effects') and effect_name in obj.visual_effects and obj.color:
                return obj.color.hex_code
        return "#FFFFFF"  # Default white
    
    @lru_cache(maxsize=256)
    def _create_silk_pattern(self, color: str) -> str:
        """Create a silk-like pattern with the given color."""
        return f'''
<pattern id="patternSilk" patternUnits="userSpaceOnUse" width="20" height="20">
    <rect width="20" height="20" fill="{color}" opacity="0.9"/>
    <path d="M0,0 L20,20 M20,0 L0,20" stroke="#FFFFFF" stroke-width="0.5" opacity="0.7"/>
</pattern>'''
        
    @lru_cache(maxsize=256)
    def _create_wool_pattern(self, color: str) -> str:
        """Create a wool-like pattern with the given color."""
        return f'''
<pattern id="patternWool" patternUnits="userSpaceOnUse" width="10" height="10">
    <rect width="10" height="10" fill="{color}"/>
    <path d="M0,0 Q2.5,2.5 5,0 Q7.5,2.5 10,0 M0,5 Q2.5,7.5 5,5 Q7.5,7.5 10,5" 
          stroke="#FFFFFF" stroke-width="1" fill="none" opacity="0.5"/>
</pattern>'''
        
    @lru_cache(maxsize=256)
    def _create_corduroy_pattern(self, color: str) -> str:
        """Create a corduroy-like pattern with the given color."""
        return f'''
<pattern id="patternCorduroy" patternUnits="userSpaceOnUse" width="8" height="8">
    <rect width="8" height="8" fill="{color}"/>
    <rect x="0" y="0" width="8" height="1" fill="#FFFFFF" opacity="0.3"/>
    <rect x="0" y="3" width="8" height="1" fill="#FFFFFF" opacity="0.3"/>
    <rect x="0" y="6" width="8" height="1" fill="#FFFFFF" opacity="0.3"/>
</pattern>'''
        
    @lru_cache(maxsize=256)
    def _create_fur_pattern(self, color: str) -> str:
        """Create a fur-like pattern with the given color."""
        return f'''
<pattern id="patternFur" patternUnits="userSpaceOnUse" width="20" height="20">
    <rect width="20" height="20" fill="{color}"/>
    <path d="M5,0 L5,8 M10,0 L10,10 M15,0 L15,7 M0,5 L8,5 M0,10 L10,10 M0,15 L7,15" 
          stroke="#FFFFFF" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>
</pattern>'''
        
    @lru_cache(maxsize=256)
    def _create_shimmering_pattern(self, color: str) -> str:
        """Create a shimmering effect pattern with the given color."""
        return f'''
<pattern id="patternShimmering" patternUnits="userSpaceOnUse" width="40" height="40">
    <rect width="40" height="40" fill="{color}"/>
    <path d="M0,0 L40,40 M40,0 L0,40" stroke="#FFFFFF" stroke-width="0.5" opacity="0.6"/>
    <path d="M20,0 L20,40 M0,20 L40,20" stroke="#FFFFFF" stroke-width="0.3" opacity="0.4"/>
    <rect width="40" height="40" fill="url(#shimmerGradient)"/>
</pattern>
<linearGradient id="shimmerGradient" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.2"/>
    <stop offset="50%" stop-color="#FFFFFF" stop-opacity="0.1"/>
    <stop offset="100%" stop-color="#FFFFFF" stop-opacity="0.2"/>
</linearGradient>'''
        
    @lru_cache(maxsize=256)
    def _create_ribbed_pattern(self, color: str) -> str:
        """Create a ribbed pattern with the given color."""
        return f'''
<pattern id="patternRibbed" patternUnits="userSpaceOnUse" width="10" height="10">
    <rect width="10" height="10" fill="{color}"/>
    <rect x="0" y="0" width="10" height="2" fill="#000000" opacity="0.2"/>
    <rect x="0" y="4" width="10" height="2" fill="#000000" opacity="0.2"/>
    <rect x="0" y="8" width="10" height="2" fill="#000000" opacity="0.2"/>
</pattern>'''
        
    @lru_cache(maxsize=256)
    def _create_gradient_def(self) -> str:
        """Create a default gradient definition."""
        return '''
<linearGradient id="defaultGradient" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.9"/>
    <stop offset="100%" stop-color="#000000" stop-opacity="0.2"/>
</linearGradient>'''
        
    @lru_cache(maxsize=256)
    def _create_pattern_def(self) -> str:
        """Create a default pattern definition."""
        return '''
<pattern id="defaultPattern" patternUnits="userSpaceOnUse" width="10" height="10">
    <rect width="10" height="10" fill="#FFFFFF"/>
    <circle cx="5" cy="5" r="2" fill="#000000" opacity="0.5"/>
</pattern>'''
    
    def _get_cache_key(self, prompt_id: str, prompt_text: str) -> str:
        """Generate a cache key for the prompt."""
        import hashlib
        return hashlib.md5(f"{prompt_id}:{prompt_text}".encode()).hexdigest()
    
    def _save_to_cache(self, cache_key: str, scene: Scene) -> None:
        """Save a scene to both memory and disk cache."""
        try:
            # Update memory cache
            self._analysis_cache[cache_key] = scene
            
            # Limit memory cache size
            if len(self._analysis_cache) > self.max_cache_size:
                # Remove a random key to avoid synchronized removals in parallel processing
                keys = list(self._analysis_cache.keys())
                if len(keys) > 0:
                    random_key = keys[0]  # Take the first (oldest) key
                    self._analysis_cache.pop(random_key, None)
            
            # Save to disk cache
            cache_path = Path(self.cache_dir) / f"{cache_key}.json"
            
            # Serialize scene to dict
            scene_dict = {
                "id": scene.id,
                "prompt": scene.prompt,
                "background_color": scene.background_color,
                "width": scene.width,
                "height": scene.height,
                "patterns": scene.patterns,
                "objects": []
            }
            
            # Add objects
            for obj in scene.objects:
                obj_dict = {
                    "id": obj.id,
                    "name": obj.name,
                    "object_type": obj.object_type.value,
                    "position": obj.position,
                    "size": obj.size,
                    "z_index": obj.z_index,
                }
                
                # Add color
                if obj.color:
                    obj_dict["color"] = {
                        "name": obj.color.name,
                        "hex_code": obj.color.hex_code
                    }
                
                # Add material
                if obj.material:
                    obj_dict["material"] = {
                        "name": obj.material.name,
                        "texture": obj.material.texture,
                        "transparency": obj.material.transparency
                    }
                
                # Add shapes
                if obj.shapes:
                    obj_dict["shapes"] = []
                    for shape in obj.shapes:
                        shape_dict = {
                            "shape_type": shape.shape_type,
                            "attributes": [{"name": attr.name, "value": attr.value} for attr in shape.attributes],
                            "visual_effects": shape.visual_effects
                        }
                        if shape.rotation is not None:
                            shape_dict["rotation"] = shape.rotation
                        obj_dict["shapes"].append(shape_dict)
                
                # Add visual effects
                if hasattr(obj, 'visual_effects') and obj.visual_effects:
                    obj_dict["visual_effects"] = obj.visual_effects
                
                scene_dict["objects"].append(obj_dict)
            
            # Write to disk asynchronously
            self._executor.submit(self._write_cache_file, cache_path, scene_dict)
            
        except Exception as e:
            logger.error(f"Error saving scene to cache: {str(e)}")
    
    def _write_cache_file(self, path: Path, data: Dict[str, Any]) -> None:
        """Write data to cache file."""
        try:
            with open(path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error writing cache file {path}: {str(e)}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Scene]:
        """
        Load a scene from disk cache.
        
        Args:
            cache_key: Cache key to load
            
        Returns:
            Scene object or None if not cached
        """
        cache_path = Path(self.cache_dir) / f"{cache_key}.json"
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r') as f:
                scene_dict = json.load(f)
                
            # Reconstruct scene
            scene = Scene(
                id=scene_dict["id"],
                prompt=scene_dict["prompt"],
                background_color=scene_dict["background_color"],
                width=scene_dict["width"],
                height=scene_dict["height"],
                patterns=scene_dict.get("patterns", {})
            )
            
            # Add objects
            for obj_dict in scene_dict.get("objects", []):
                # Create color if present
                color = None
                if "color" in obj_dict:
                    color_dict = obj_dict["color"]
                    color = Color(
                        name=color_dict["name"],
                        hex_code=color_dict.get("hex_code")
                    )
                
                # Create material if present
                material = None
                if "material" in obj_dict:
                    material_dict = obj_dict["material"]
                    material = Material(
                        name=material_dict["name"],
                        texture=material_dict.get("texture"),
                        transparency=material_dict.get("transparency", 0.0)
                    )
                
                # Create shapes
                shapes = []
                for shape_dict in obj_dict.get("shapes", []):
                    shape = Shape(
                        shape_type=shape_dict["shape_type"],
                        attributes=[
                            Attribute(attr["name"], attr["value"]) 
                            for attr in shape_dict.get("attributes", [])
                        ],
                        visual_effects=shape_dict.get("visual_effects", {})
                    )
                    if "rotation" in shape_dict:
                        shape.rotation = shape_dict["rotation"]
                    shapes.append(shape)
                
                # Create object
                visual_object = VisualObject(
                    id=obj_dict["id"],
                    name=obj_dict["name"],
                    object_type=ObjectType(obj_dict["object_type"]),
                    shapes=shapes,
                    color=color,
                    material=material,
                    position=tuple(obj_dict["position"]),
                    size=obj_dict["size"],
                    z_index=obj_dict["z_index"]
                )
                
                # Add visual effects
                if "visual_effects" in obj_dict:
                    for effect, value in obj_dict["visual_effects"].items():
                        visual_object.add_visual_effect(effect, value)
                
                scene.objects.append(visual_object)
                
            return scene
            
        except Exception as e:
            logger.error(f"Error loading scene from cache: {str(e)}")
            # Remove corrupted cache file
            try:
                cache_path.unlink(missing_ok=True)
            except:
                pass
            return None
    
    def clear_cache(self) -> None:
        """Clear both memory and disk caches."""
        # Clear memory cache
        self._analysis_cache.clear()
        
        # Clear disk cache
        try:
            for cache_file in Path(self.cache_dir).glob("*.json"):
                try:
                    cache_file.unlink()
                except:
                    pass
        except Exception as e:
            logger.error(f"Error clearing disk cache: {str(e)}")
    
    def shutdown(self) -> None:
        """Cleanup resources and shut down the analyzer."""
        self._executor.shutdown(wait=True)
        gc.collect()