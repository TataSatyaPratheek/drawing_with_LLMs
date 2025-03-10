"""
Scene Elaborator Module
====================
This module provides a dedicated scene elaboration component that enhances
scene details, improves visual coherence, and ensures proper spatial relationships.
"""

import os
import time
import logging
import json
import threading
import re
import math
import gc
from typing import Dict, Any, Optional, List, Tuple, Set, Union
from concurrent.futures import ThreadPoolExecutor

# Import core optimizations
from svg_prompt_analyzer.core import CONFIG, memoize, Profiler
from svg_prompt_analyzer.core.memory_manager import MemoryManager
from svg_prompt_analyzer.core.batch_processor import BatchProcessor
from svg_prompt_analyzer.core.hardware_manager import HardwareManager

# Import LLM manager
from svg_prompt_analyzer.llm_integration.llm_manager import LLMManager

# Import models
from svg_prompt_analyzer.models.scene import Scene
from svg_prompt_analyzer.models.visual_object import VisualObject, ObjectType
from svg_prompt_analyzer.models.color import Color
from svg_prompt_analyzer.models.shape import Shape, Attribute

logger = logging.getLogger(__name__)

# Get memory manager instance
memory_manager = MemoryManager()


class SceneElaborator:
    """
    Dedicated component that enhances scene representations with more detailed
    visual information, improved spatial coherence, and richer visual elements.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for resource efficiency."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SceneElaborator, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, 
                 llm_manager: Optional[LLMManager] = None,
                 use_caching: bool = True,
                 cache_dir: str = ".cache/scene_elaboration",
                 max_cache_size: int = 500,
                 num_workers: int = 4,
                 reference_database: Optional[str] = None):
        """
        Initialize the scene elaborator.
        
        Args:
            llm_manager: LLM manager instance for model access
            use_caching: Whether to cache elaboration results
            cache_dir: Directory for caching elaboration results
            max_cache_size: Maximum number of cached elaborations to keep
            num_workers: Number of worker threads for parallel processing
            reference_database: Optional path to visual reference database
        """
        # Initialize only once (singleton pattern)
        if hasattr(self, 'initialized'):
            return
            
        self.llm_manager = llm_manager or LLMManager()
        self.use_caching = use_caching
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.hardware_manager = HardwareManager()
        
        # Adjust workers based on hardware capabilities
        optimal_device = self.hardware_manager.get_optimal_device()
        if optimal_device == "cpu":
            # Reduce worker count on CPU to avoid overload
            num_workers = min(num_workers, self.hardware_manager.hardware_info["processor_count"] // 2)
        self.num_workers = max(1, min(num_workers, 16))
        
        self.reference_database = reference_database
        
        # Create cache directory if needed
        if self.use_caching:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Memory cache for recent elaborations
        self._elaboration_cache = {}
        
        # Batch processor for parallel processing
        self.batch_processor = BatchProcessor(
            process_func=self._process_elaboration_batch,
            optimal_batch_size=8,
            max_batch_size=16,
            adaptive_batching=True,
            memory_manager=memory_manager
        )
        self.batch_processor.start()
        
        # Visual reference database (if available)
        self.visual_references = self._load_reference_database()
        
        # Track initialization
        self.initialized = True
        logger.info(f"Scene Elaborator initialized with {self.num_workers} workers on {optimal_device} device")
    
    @memory_manager.memory_efficient_function
    def elaborate_scene(self, scene: Scene) -> Scene:
        """
        Enhance a scene with more detailed visual information.
        
        Args:
            scene: Original scene to elaborate
            
        Returns:
            Elaborated scene with enhanced details
        """
        # Check cache first
        if self.use_caching:
            cache_key = self._get_cache_key(scene)
            if cache_key in self._elaboration_cache:
                logger.info(f"Using cached elaboration for scene {scene.id}")
                return self._elaboration_cache[cache_key]
        
        logger.info(f"Elaborating scene {scene.id}")
        
        try:
            with Profiler("scene_elaboration"):
                # 1. Enhance colors and materials
                scene = self._enhance_visual_properties(scene)
                
                # Memory checkpoint
                memory_manager.operation_checkpoint()
                
                # 2. Refine spatial relationships
                scene = self._refine_spatial_relationships(scene)
                
                # Memory checkpoint
                memory_manager.operation_checkpoint()
                
                # 3. Add missing visual elements
                scene = self._add_missing_elements(scene)
                
                # Memory checkpoint
                memory_manager.operation_checkpoint()
                
                # 4. Create visual coherence
                scene = self._ensure_visual_coherence(scene)
                
                # Memory checkpoint
                memory_manager.operation_checkpoint()
                
                # 5. Optimize for CLIP similarity
                scene = self._optimize_for_clip(scene)
                
                # Cache the result
                if self.use_caching:
                    self._cache_result(cache_key, scene)
            
            return scene
            
        except Exception as e:
            logger.error(f"Error during scene elaboration: {str(e)}")
            # Return original scene on error
            return scene
    
    def elaborate_scenes_batch(self, scenes: List[Scene]) -> List[Scene]:
        """
        Elaborate multiple scenes in parallel.
        
        Args:
            scenes: List of scenes to elaborate
            
        Returns:
            List of elaborated scenes
        """
        if not scenes:
            return []
            
        logger.info(f"Elaborating batch of {len(scenes)} scenes")
        
        # Check cache for each scene
        results = []
        scenes_to_process = []
        
        for scene in scenes:
            if self.use_caching:
                cache_key = self._get_cache_key(scene)
                if cache_key in self._elaboration_cache:
                    logger.debug(f"Using cached elaboration for scene {scene.id}")
                    results.append(self._elaboration_cache[cache_key])
                    continue
            
            scenes_to_process.append(scene)
        
        if not scenes_to_process:
            return results
            
        # Process remaining scenes in parallel
        try:
            with Profiler("batch_scene_elaboration"):
                # Add tasks to batch processor
                futures = []
                for scene in scenes_to_process:
                    futures.append(self.batch_processor.add_item(
                        scene.id, 
                        scene, 
                        priority=1
                    ))
                
                # Collect results with timeout
                processed_scenes = []
                for scene in scenes_to_process:
                    try:
                        elaborated_scene = self.batch_processor.get_result(scene.id, timeout=60.0)
                        if elaborated_scene:
                            processed_scenes.append(elaborated_scene)
                        else:
                            # Fall back to sequential processing if batch processing failed
                            processed_scenes.append(self.elaborate_scene(scene))
                    except Exception as e:
                        logger.error(f"Error in batch processing for scene {scene.id}: {str(e)}")
                        # Fall back to original scene
                        processed_scenes.append(scene)
                
                results.extend(processed_scenes)
                
        except Exception as e:
            logger.error(f"Error during batch scene elaboration: {str(e)}")
            # Fall back to sequential processing
            for scene in scenes_to_process:
                results.append(self.elaborate_scene(scene))
        
        return results
    
    def _process_elaboration_batch(self, scenes: List[Scene]) -> List[Scene]:
        """Process a batch of scenes for the batch processor."""
        return [self.elaborate_scene(scene) for scene in scenes]
    
    @memoize
    def _enhance_visual_properties(self, scene: Scene) -> Scene:
        """
        Enhance visual properties of objects in the scene.
        
        Args:
            scene: Scene to enhance
            
        Returns:
            Scene with enhanced visual properties
        """
        # Create LLM prompt for visual enhancement
        prompt = self._create_visual_properties_prompt(scene)
        
        # Get LLM response
        try:
            response = self.llm_manager.generate(
                role="scene_elaborator",
                prompt=prompt,
                max_tokens=2048,
                temperature=0.3
            )
            
            # Parse the response
            enhancements = self._parse_visual_properties_response(response)
            
            # Apply enhancements to scene
            return self._apply_visual_enhancements(scene, enhancements)
            
        except Exception as e:
            logger.warning(f"Error in visual properties enhancement: {str(e)}")
            return scene
    
    @memoize
    def _refine_spatial_relationships(self, scene: Scene) -> Scene:
        """
        Refine spatial relationships between objects.
        
        Args:
            scene: Scene to refine
            
        Returns:
            Scene with refined spatial relationships
        """
        # If no objects or only one object, no relationships to refine
        if len(scene.objects) <= 1:
            return scene
        
        # Create LLM prompt for spatial refinement
        prompt = self._create_spatial_relationships_prompt(scene)
        
        # Get LLM response
        try:
            response = self.llm_manager.generate(
                role="scene_elaborator",
                prompt=prompt,
                max_tokens=2048,
                temperature=0.2
            )
            
            # Parse the response
            relationships = self._parse_spatial_relationships_response(response)
            
            # Apply relationships to scene
            return self._apply_spatial_relationships(scene, relationships)
            
        except Exception as e:
            logger.warning(f"Error in spatial relationship refinement: {str(e)}")
            return scene
    
    @memoize
    def _add_missing_elements(self, scene: Scene) -> Scene:
        """
        Add missing visual elements to the scene.
        
        Args:
            scene: Scene to enhance
            
        Returns:
            Scene with added elements
        """
        # Create LLM prompt for missing elements
        prompt = self._create_missing_elements_prompt(scene)
        
        # Get LLM response
        try:
            response = self.llm_manager.generate(
                role="scene_elaborator",
                prompt=prompt,
                max_tokens=2048,
                temperature=0.4
            )
            
            # Parse the response
            missing_elements = self._parse_missing_elements_response(response)
            
            # Add elements to scene
            return self._add_elements_to_scene(scene, missing_elements)
            
        except Exception as e:
            logger.warning(f"Error in adding missing elements: {str(e)}")
            return scene
    
    @memoize
    def _ensure_visual_coherence(self, scene: Scene) -> Scene:
        """
        Ensure visual coherence across the scene.
        
        Args:
            scene: Scene to ensure coherence for
            
        Returns:
            Scene with improved visual coherence
        """
        # Create a coherence prompt
        prompt = self._create_visual_coherence_prompt(scene)
        
        # Get LLM response
        try:
            response = self.llm_manager.generate(
                role="scene_elaborator",
                prompt=prompt,
                max_tokens=1024,
                temperature=0.2
            )
            
            # Parse the response
            coherence_adjustments = self._parse_visual_coherence_response(response)
            
            # Apply coherence adjustments
            return self._apply_coherence_adjustments(scene, coherence_adjustments)
            
        except Exception as e:
            logger.warning(f"Error in ensuring visual coherence: {str(e)}")
            return scene
    
    @memoize
    def _optimize_for_clip(self, scene: Scene) -> Scene:
        """
        Make scene adjustments to optimize for CLIP similarity.
        
        Args:
            scene: Scene to optimize
            
        Returns:
            Scene optimized for CLIP similarity
        """
        # Create optimization prompt
        prompt = self._create_clip_optimization_prompt(scene)
        
        # Get LLM response
        try:
            response = self.llm_manager.generate(
                role="scene_elaborator",
                prompt=prompt,
                max_tokens=1024,
                temperature=0.2
            )
            
            # Parse the response
            optimizations = self._parse_clip_optimization_response(response)
            
            # Apply optimizations
            return self._apply_clip_optimizations(scene, optimizations)
            
        except Exception as e:
            logger.warning(f"Error in CLIP optimization: {str(e)}")
            return scene
    
    def _create_visual_properties_prompt(self, scene: Scene) -> str:
        """
        Create a prompt for enhancing visual properties.
        
        Args:
            scene: Scene to create prompt for
            
        Returns:
            Prompt text
        """
        objects_info = [
            {
                "name": obj.name,
                "type": obj.object_type.value,
                "color": obj.color.name if obj.color else None,
                "material": obj.material.name if obj.material else None,
                "shape": obj.shapes[0].shape_type if obj.shapes and len(obj.shapes) > 0 else None
            }
            for obj in scene.objects
        ]
        
        scene_info = {
            "prompt": scene.prompt,
            "objects": objects_info
        }
        
        scene_json = json.dumps(scene_info, indent=2)
        
        return f"""You are an expert visual designer specializing in SVG illustrations. Enhance the visual properties of objects in this scene to make them more visually appealing and detailed.

For each object, provide enhanced specifications including:
1. Detailed color information (specific shade, gradient potential)
2. Rich material properties (texture, reflectivity, patterns)
3. Visual effect suggestions (shadows, highlights, patterns)

Scene information:
{scene_json}

Provide your enhancements as a JSON object with the following structure:
{{
  "object_enhancements": [
    {{
      "name": "object name",
      "color": {{
        "name": "enhanced color name",
        "hex_code": "#RRGGBB",
        "is_gradient": true/false,
        "gradient_direction": "direction" (if applicable)
      }},
      "material": {{
        "name": "enhanced material name", 
        "texture": "texture description",
        "transparency": 0.0-1.0
      }},
      "visual_effects": ["effect1", "effect2"]
    }}
  ]
}}"""
    
    def _create_spatial_relationships_prompt(self, scene: Scene) -> str:
        """
        Create a prompt for refining spatial relationships.
        
        Args:
            scene: Scene to create prompt for
            
        Returns:
            Prompt text
        """
        objects_info = [
            {
                "name": obj.name,
                "id": obj.id,
                "type": obj.object_type.value,
                "position": [f"{p:.2f}" for p in obj.position],
                "size": f"{obj.size:.2f}"
            }
            for obj in scene.objects
        ]
        
        scene_info = {
            "prompt": scene.prompt,
            "objects": objects_info
        }
        
        scene_json = json.dumps(scene_info, indent=2)
        
        return f"""You are an expert in visual composition. Refine the spatial relationships between objects in this scene to create a visually appealing and coherent composition.

Consider principles of visual design:
1. Balance and harmony
2. Visual hierarchy
3. Proximity and grouping
4. Alignment and flow

Scene information:
{scene_json}

Define clear spatial relationships between objects using the following types:
- ABOVE/BELOW: Vertical relationships
- LEFT_OF/RIGHT_OF: Horizontal relationships 
- INSIDE/CONTAINING: Containment relationships
- OVERLAPPING: Z-index relationships
- NEAR/FAR: Proximity relationships
- ALIGNED_WITH: Alignment relationships
- FACING: Directional relationships

Provide your refinements as a JSON object with the following structure:
{{
  "spatial_relationships": [
    {{
      "object1_id": "id of first object",
      "object2_id": "id of second object",
      "relationship": "relationship type",
      "offset": [0.0, 0.0], // optional offset adjustment
      "z_index_adjustment": 0 // optional z-index change
    }}
  ]
}}"""
    
    def _create_missing_elements_prompt(self, scene: Scene) -> str:
        """
        Create a prompt for identifying missing visual elements.
        
        Args:
            scene: Scene to create prompt for
            
        Returns:
            Prompt text
        """
        objects_info = [obj.name for obj in scene.objects]
        
        scene_info = {
            "prompt": scene.prompt,
            "existing_objects": objects_info
        }
        
        scene_json = json.dumps(scene_info, indent=2)
        
        return f"""You are an expert visual scene designer. Analyze this scene and suggest missing visual elements that would enhance the illustration based on the original prompt.

Consider:
1. Implied but missing objects from the prompt
2. Background elements that would enhance the context
3. Complementary objects that would improve the composition
4. Details that would make the scene more engaging and complete

Scene information:
{scene_json}

Provide your suggestions as a JSON object with the following structure:
{{
  "missing_elements": [
    {{
      "name": "element name",
      "type": "element type",
      "description": "detailed description",
      "color": "suggested color",
      "position": "suggested position description",
      "importance": 1-10,
      "reason": "reason for adding this element"
    }}
  ]
}}"""
    
    def _create_visual_coherence_prompt(self, scene: Scene) -> str:
        """
        Create a prompt for ensuring visual coherence.
        
        Args:
            scene: Scene to create prompt for
            
        Returns:
            Prompt text
        """
        objects_info = [
            {
                "name": obj.name,
                "id": obj.id,
                "type": obj.object_type.value,
                "color": obj.color.name if obj.color else None,
                "material": obj.material.name if obj.material else None
            }
            for obj in scene.objects
        ]
        
        scene_info = {
            "prompt": scene.prompt,
            "background_color": scene.background_color,
            "objects": objects_info
        }
        
        scene_json = json.dumps(scene_info, indent=2)
        
        return f"""You are an expert in visual design and color theory. Ensure visual coherence across all elements in this scene.

Analyze the scene for:
1. Color harmony and palette consistency
2. Material compatibility and style consistency  
3. Visual theme coherence
4. Background-foreground harmony

Scene information:
{scene_json}

Provide coherence adjustments as a JSON object with the following structure:
{{
  "color_adjustments": [
    {{
      "object_id": "object id",
      "current_color": "current color",
      "adjusted_color": "adjusted color",
      "reason": "reason for adjustment"
    }}
  ],
  "material_adjustments": [
    {{
      "object_id": "object id",
      "current_material": "current material",
      "adjusted_material": "adjusted material",
      "reason": "reason for adjustment"
    }}
  ],
  "background_adjustment": {{
    "current_color": "current background color",
    "suggested_color": "suggested background color",
    "reason": "reason for adjustment"
  }}
}}"""
    
    def _create_clip_optimization_prompt(self, scene: Scene) -> str:
        """
        Create a prompt for CLIP similarity optimization.
        
        Args:
            scene: Scene to create prompt for
            
        Returns:
            Prompt text
        """
        scene_info = {
            "prompt": scene.prompt,
            "object_count": len(scene.objects)
        }
        
        scene_json = json.dumps(scene_info, indent=2)
        
        return f"""You are an expert in optimizing images for CLIP similarity scoring. Suggest optimizations to make this scene more closely aligned with its text description for CLIP evaluation.

Consider these CLIP optimization best practices:
1. Emphasize key semantic elements from the original prompt
2. Ensure visual prominence of mentioned objects
3. Add descriptive metadata that aligns with the prompt
4. Enhance visual clarity and recognizability of elements

Scene information:
{scene_json}

Provide optimization suggestions as a JSON object with the following structure:
{{
  "semantic_emphasis": [
    {{
      "key_term": "term from prompt",
      "emphasis_method": "method to emphasize",
      "importance": 1-10
    }}
  ],
  "visual_adjustments": [
    {{
      "element_type": "type of element",
      "adjustment": "adjustment to make",
      "purpose": "purpose of adjustment"
    }}
  ],
  "metadata_enhancements": [
    {{
      "field": "title/description/etc",
      "content": "suggested content",
      "rationale": "why this helps CLIP scoring"
    }}
  ]
}}"""
    
    def _parse_visual_properties_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response for visual property enhancements.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed enhancements
        """
        try:
            # Try to find and parse JSON in the response
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL) or re.search(r'```(.*?)```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1).strip())
            
            # Try to find JSON without code blocks
            json_like = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_like:
                return json.loads(json_like.group(1))
            
            # If no JSON found, return empty result
            logger.warning("Failed to parse visual properties response from LLM")
            return {"object_enhancements": []}
            
        except Exception as e:
            logger.error(f"Error parsing visual properties response: {str(e)}")
            return {"object_enhancements": []}
    
    def _parse_spatial_relationships_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response for spatial relationship refinements.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed relationships
        """
        try:
            # Try to find and parse JSON in the response
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL) or re.search(r'```(.*?)```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1).strip())
            
            # Try to find JSON without code blocks
            json_like = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_like:
                return json.loads(json_like.group(1))
            
            # If no JSON found, return empty result
            logger.warning("Failed to parse spatial relationships response from LLM")
            return {"spatial_relationships": []}
            
        except Exception as e:
            logger.error(f"Error parsing spatial relationships response: {str(e)}")
            return {"spatial_relationships": []}
    
    def _parse_missing_elements_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response for missing elements.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed missing elements
        """
        try:
            # Try to find and parse JSON in the response
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL) or re.search(r'```(.*?)```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1).strip())
            
            # Try to find JSON without code blocks
            json_like = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_like:
                return json.loads(json_like.group(1))
            
            # If no JSON found, return empty result
            logger.warning("Failed to parse missing elements response from LLM")
            return {"missing_elements": []}
            
        except Exception as e:
            logger.error(f"Error parsing missing elements response: {str(e)}")
            return {"missing_elements": []}
    
    def _parse_visual_coherence_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response for visual coherence adjustments.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed coherence adjustments
        """
        try:
            # Try to find and parse JSON in the response
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL) or re.search(r'```(.*?)```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1).strip())
            
            # Try to find JSON without code blocks
            json_like = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_like:
                return json.loads(json_like.group(1))
            
            # If no JSON found, return empty result
            logger.warning("Failed to parse visual coherence response from LLM")
            return {"color_adjustments": [], "material_adjustments": [], "background_adjustment": None}
            
        except Exception as e:
            logger.error(f"Error parsing visual coherence response: {str(e)}")
            return {"color_adjustments": [], "material_adjustments": [], "background_adjustment": None}
    
    def _parse_clip_optimization_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response for CLIP optimizations.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed CLIP optimizations
        """
        try:
            # Try to find and parse JSON in the response
            json_match = re.search(r'```json(.*?)```', response, re.DOTALL) or re.search(r'```(.*?)```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1).strip())
            
            # Try to find JSON without code blocks
            json_like = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_like:
                return json.loads(json_like.group(1))
            
            # If no JSON found, return empty result
            logger.warning("Failed to parse CLIP optimization response from LLM")
            return {"semantic_emphasis": [], "visual_adjustments": [], "metadata_enhancements": []}
            
        except Exception as e:
            logger.error(f"Error parsing CLIP optimization response: {str(e)}")
            return {"semantic_emphasis": [], "visual_adjustments": [], "metadata_enhancements": []}
    
    def _apply_visual_enhancements(self, scene: Scene, enhancements: Dict[str, Any]) -> Scene:
        """
        Apply visual enhancements to a scene.
        
        Args:
            scene: Original scene
            enhancements: Enhancement data
            
        Returns:
            Enhanced scene
        """
        # Create a copy of the scene to avoid modifying the original
        enhanced_scene = Scene(
            id=scene.id,
            prompt=scene.prompt,
            background_color=scene.background_color,
            objects=scene.objects.copy(),
            width=scene.width,
            height=scene.height,
            patterns=scene.patterns.copy() if hasattr(scene, 'patterns') and scene.patterns else {},
            defs=scene.defs.copy() if hasattr(scene, 'defs') and scene.defs else [],
            special_elements=scene.special_elements.copy() if hasattr(scene, 'special_elements') else []
        )
        
        # Apply object enhancements
        object_enhancements = enhancements.get("object_enhancements", [])
        for enhancement in object_enhancements:
            obj_name = enhancement.get("name")
            
            # Find matching object
            for obj in enhanced_scene.objects:
                if obj.name.lower() == obj_name.lower():
                    # Apply color enhancement
                    if "color" in enhancement:
                        color_data = enhancement["color"]
                        color_name = color_data.get("name", obj.color.name if obj.color else "default")
                        hex_code = color_data.get("hex_code")
                        
                        # Create a new color object
                        try:
                            obj.color = Color(
                                name=color_name,
                                hex_code=hex_code
                            )
                            
                            # Check for gradient
                            if color_data.get("is_gradient", False):
                                # If we have gradient functionality, apply it here
                                gradient_direction = color_data.get("gradient_direction", "vertical")
                                
                                # Add gradient definition if necessary
                                gradient_id = f"gradient_{obj.id}"
                                if not hasattr(enhanced_scene, 'defs'):
                                    enhanced_scene.defs = []
                                
                                # Create a basic gradient definition
                                if hasattr(enhanced_scene, 'defs'):
                                    gradient_def = {
                                        "type": "linearGradient",
                                        "id": gradient_id,
                                        "direction": gradient_direction,
                                        "stops": [
                                            {"offset": 0, "color": hex_code, "opacity": 1.0},
                                            {"offset": 1, "color": self._adjust_color_brightness(hex_code, -0.3), "opacity": 1.0}
                                        ]
                                    }
                                    enhanced_scene.defs.append(gradient_def)
                                    
                                    # Update object to use gradient
                                    if obj.shapes and len(obj.shapes) > 0:
                                        for shape in obj.shapes:
                                            for attr in shape.attributes:
                                                if attr.name == "fill":
                                                    attr.value = f"url(#{gradient_id})"
                        except Exception as e:
                            logger.warning(f"Error applying color enhancement: {e}")
                        
                    # Apply material enhancement
                    if "material" in enhancement:
                        material_data = enhancement["material"]
                        material_name = material_data.get("name", "default")
                        texture = material_data.get("texture")
                        transparency = material_data.get("transparency", 0.0)
                        
                        # Create a new material object
                        try:
                            obj.material = Material(
                                name=material_name,
                                texture=texture,
                                transparency=transparency
                            )
                            
                            # Apply transparency to object
                            if transparency > 0:
                                # Update shape opacity
                                if obj.shapes and len(obj.shapes) > 0:
                                    for shape in obj.shapes:
                                        opacity_attr = None
                                        for attr in shape.attributes:
                                            if attr.name == "opacity":
                                                opacity_attr = attr
                                                break
                                        
                                        if opacity_attr:
                                            opacity_attr.value = str(1.0 - transparency)
                                        else:
                                            shape.attributes.append(Attribute("opacity", str(1.0 - transparency)))
                        except Exception as e:
                            logger.warning(f"Error applying material enhancement: {e}")
                        
                    # Apply visual effects
                    if "visual_effects" in enhancement:
                        for effect in enhancement["visual_effects"]:
                            # Add effect handling
                            # This would depend on how visual effects are implemented in the system
                            if hasattr(obj, 'add_visual_effect'):
                                obj.add_visual_effect(effect)
                            else:
                                # Simple fallback for shadow effect
                                if effect.lower() == "shadow" and obj.shapes and len(obj.shapes) > 0:
                                    # Create a shadow effect by adding a filter
                                    filter_id = f"shadow_{obj.id}"
                                    
                                    if hasattr(enhanced_scene, 'defs'):
                                        shadow_def = {
                                            "type": "filter",
                                            "id": filter_id,
                                            "elements": [{
                                                "type": "feDropShadow",
                                                "dx": 3,
                                                "dy": 3,
                                                "stdDeviation": 3,
                                                "flood-color": "#000000",
                                                "flood-opacity": 0.3
                                            }]
                                        }
                                        enhanced_scene.defs.append(shadow_def)
                                        
                                        # Apply filter to object
                                        for shape in obj.shapes:
                                            has_filter = False
                                            for attr in shape.attributes:
                                                if attr.name == "filter":
                                                    attr.value = f"url(#{filter_id})"
                                                    has_filter = True
                                                    break
                                            
                                            if not has_filter:
                                                shape.attributes.append(Attribute("filter", f"url(#{filter_id})"))
        
        return enhanced_scene
    
    def _adjust_color_brightness(self, hex_code: str, factor: float) -> str:
        """Adjust the brightness of a color by a factor (-1.0 to 1.0)."""
        if not hex_code or not hex_code.startswith('#'):
            return hex_code
            
        try:
            # Convert hex to RGB
            r = int(hex_code[1:3], 16)
            g = int(hex_code[3:5], 16)
            b = int(hex_code[5:7], 16)
            
            # Adjust brightness
            if factor > 0:
                # Brighten
                r = min(255, int(r + (255 - r) * factor))
                g = min(255, int(g + (255 - g) * factor))
                b = min(255, int(b + (255 - b) * factor))
            else:
                # Darken
                factor = abs(factor)
                r = max(0, int(r * (1 - factor)))
                g = max(0, int(g * (1 - factor)))
                b = max(0, int(b * (1 - factor)))
                
            # Convert back to hex
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return hex_code
    
    def _apply_spatial_relationships(self, scene: Scene, relationships: Dict[str, Any]) -> Scene:
        """
        Apply spatial relationship refinements to a scene.
        
        Args:
            scene: Original scene
            relationships: Relationship data
            
        Returns:
            Refined scene
        """
        # Create a copy of the scene to avoid modifying the original
        refined_scene = Scene(
            id=scene.id,
            prompt=scene.prompt,
            background_color=scene.background_color,
            objects=scene.objects.copy(),
            width=scene.width,
            height=scene.height,
            patterns=scene.patterns.copy() if hasattr(scene, 'patterns') and scene.patterns else {},
            defs=scene.defs.copy() if hasattr(scene, 'defs') and scene.defs else [],
            special_elements=scene.special_elements.copy() if hasattr(scene, 'special_elements') else []
        )
        
        # Create id-to-object mapping for quick lookup
        object_map = {obj.id: obj for obj in refined_scene.objects}
        
        # Apply spatial relationships
        spatial_relationships = relationships.get("spatial_relationships", [])
        for rel in spatial_relationships:
            obj1_id = rel.get("object1_id")
            obj2_id = rel.get("object2_id")
            relationship = rel.get("relationship")
            offset = rel.get("offset", [0.0, 0.0])
            z_index_adjustment = rel.get("z_index_adjustment", 0)
            
            # Find objects by id
            obj1 = object_map.get(obj1_id)
            obj2 = object_map.get(obj2_id)
            
            if obj1 and obj2:
                try:
                    # Apply relationship based on type
                    if relationship == "ABOVE":
                        obj1.position = (obj1.position[0], obj2.position[1] - obj1.size - offset[1])
                        obj1.z_index = obj2.z_index + 1 + z_index_adjustment
                        
                    elif relationship == "BELOW":
                        obj1.position = (obj1.position[0], obj2.position[1] + obj2.size + offset[1])
                        obj1.z_index = obj2.z_index - 1 + z_index_adjustment
                        
                    elif relationship == "LEFT_OF":
                        obj1.position = (obj2.position[0] - obj2.size - obj1.size - offset[0], obj1.position[1])
                        
                    elif relationship == "RIGHT_OF":
                        obj1.position = (obj2.position[0] + obj2.size + obj1.size + offset[0], obj1.position[1])
                        
                    elif relationship == "INSIDE":
                        obj1.position = obj2.position
                        obj1.size = obj2.size * 0.7  # Make contained object smaller
                        obj1.z_index = obj2.z_index + 1 + z_index_adjustment
                        
                    elif relationship == "CONTAINING":
                        obj1.position = obj2.position
                        obj1.size = obj2.size * 1.3  # Make containing object larger
                        obj1.z_index = obj2.z_index - 1 + z_index_adjustment
                        
                    elif relationship == "OVERLAPPING":
                        # Partial overlap
                        offset_x = min(obj1.size, obj2.size) * 0.3
                        offset_y = min(obj1.size, obj2.size) * 0.3
                        obj1.position = (obj2.position[0] - offset_x, obj2.position[1] - offset_y)
                        obj1.z_index = obj2.z_index + 1 + z_index_adjustment
                        
                    elif relationship == "NEAR":
                        # Position near but not touching
                        distance = (obj1.size + obj2.size) * 0.7
                        angle = math.atan2(obj1.position[1] - obj2.position[1], 
                                        obj1.position[0] - obj2.position[0])
                        obj1.position = (
                            obj2.position[0] + distance * math.cos(angle) + offset[0],
                            obj2.position[1] + distance * math.sin(angle) + offset[1]
                        )
                        
                    elif relationship == "ALIGNED_WITH":
                        # Align either horizontally or vertically
                        if abs(obj1.position[0] - obj2.position[0]) < abs(obj1.position[1] - obj2.position[1]):
                            # Align horizontally
                            obj1.position = (obj2.position[0] + offset[0], obj1.position[1])
                        else:
                            # Align vertically
                            obj1.position = (obj1.position[0], obj2.position[1] + offset[1])
                            
                    elif relationship == "FACING":
                        # Make objects face each other
                        # Calculate direction from obj1 to obj2
                        dx = obj2.position[0] - obj1.position[0]
                        dy = obj2.position[1] - obj1.position[1]
                        angle = math.atan2(dy, dx)
                        
                        # Update positions to face each other
                        distance = (obj1.size + obj2.size) * 1.5
                        obj1.position = (
                            0.5 - distance/2 * math.cos(angle) + offset[0],
                            0.5 - distance/2 * math.sin(angle) + offset[1]
                        )
                        obj2.position = (
                            0.5 + distance/2 * math.cos(angle),
                            0.5 + distance/2 * math.sin(angle)
                        )
                        
                        # If objects have shapes, rotate them to face each other
                        if obj1.shapes and len(obj1.shapes) > 0:
                            obj1.shapes[0].rotation = angle * 180 / math.pi
                        
                        if obj2.shapes and len(obj2.shapes) > 0:
                            obj2.shapes[0].rotation = (angle + math.pi) * 180 / math.pi
                except Exception as e:
                    logger.warning(f"Error applying spatial relationship {relationship}: {e}")
        
        return refined_scene
    
    def _add_elements_to_scene(self, scene: Scene, missing_elements: Dict[str, Any]) -> Scene:
        """
        Add missing visual elements to a scene.
        
        Args:
            scene: Original scene
            missing_elements: Missing element data
            
        Returns:
            Enhanced scene
        """
        # Create a copy of the scene to avoid modifying the original
        enhanced_scene = Scene(
            id=scene.id,
            prompt=scene.prompt,
            background_color=scene.background_color,
            objects=scene.objects.copy(),
            width=scene.width,
            height=scene.height,
            patterns=scene.patterns.copy() if hasattr(scene, 'patterns') and scene.patterns else {},
            defs=scene.defs.copy() if hasattr(scene, 'defs') and scene.defs else [],
            special_elements=scene.special_elements.copy() if hasattr(scene, 'special_elements') else []
        )
        
        # Add missing elements
        elements = missing_elements.get("missing_elements", [])
        for i, element in enumerate(elements):
            try:
                # Only add elements with importance > 5 (more important elements)
                if element.get("importance", 0) <= 5:
                    continue
                    
                # Determine object type
                obj_type = ObjectType.ABSTRACT
                obj_type_str = element.get("type", "").upper()
                try:
                    obj_type = ObjectType(obj_type_str.lower())
                except (ValueError, AttributeError):
                    # Default to abstract if type not recognized
                    pass
                
                # Create color
                color = None
                color_name = element.get("color")
                if color_name:
                    try:
                        color = Color(name=color_name)
                    except Exception:
                        # Default color if failed
                        color = Color(name="gray", hex_code="#808080")
                
                # Determine position based on description
                position = (0.5, 0.5)  # Default to center
                position_desc = element.get("position", "").lower()
                
                if "top" in position_desc and "left" in position_desc:
                    position = (0.25, 0.25)
                elif "top" in position_desc and "right" in position_desc:
                    position = (0.75, 0.25)
                elif "bottom" in position_desc and "left" in position_desc:
                    position = (0.25, 0.75)
                elif "bottom" in position_desc and "right" in position_desc:
                    position = (0.75, 0.75)
                elif "top" in position_desc:
                    position = (0.5, 0.25)
                elif "bottom" in position_desc:
                    position = (0.5, 0.75)
                elif "left" in position_desc:
                    position = (0.25, 0.5)
                elif "right" in position_desc:
                    position = (0.75, 0.5)
                    
                # Create a default shape
                shape = Shape(
                    shape_type="rectangle",
                    attributes=[
                        Attribute("fill", color.hex_code if color else "#808080"),
                        Attribute("stroke", "#000000"),
                        Attribute("stroke-width", 1)
                    ]
                )
                
                # Create the new object
                new_obj = VisualObject(
                    id=f"{scene.id}_added_{i}",
                    name=element.get("name", f"Added Element {i}"),
                    object_type=obj_type,
                    shapes=[shape],
                    color=color,
                    position=position,
                    size=0.15,  # Default size
                    z_index=len(enhanced_scene.objects) + i  # Place on top
                )
                
                # Add to scene objects
                enhanced_scene.objects.append(new_obj)
            except Exception as e:
                logger.warning(f"Error adding missing element: {e}")
        
        return enhanced_scene
    
    def _apply_coherence_adjustments(self, scene: Scene, adjustments: Dict[str, Any]) -> Scene:
        """
        Apply visual coherence adjustments to a scene.
        
        Args:
            scene: Original scene
            adjustments: Coherence adjustment data
            
        Returns:
            Coherent scene
        """
        # Create a copy of the scene to avoid modifying the original
        coherent_scene = Scene(
            id=scene.id,
            prompt=scene.prompt,
            background_color=scene.background_color,
            objects=scene.objects.copy(),
            width=scene.width,
            height=scene.height,
            patterns=scene.patterns.copy() if hasattr(scene, 'patterns') and scene.patterns else {},
            defs=scene.defs.copy() if hasattr(scene, 'defs') and scene.defs else [],
            special_elements=scene.special_elements.copy() if hasattr(scene, 'special_elements') else []
        )
        
        # Create id-to-object mapping for quick lookup
        object_map = {obj.id: obj for obj in coherent_scene.objects}
        
        # Apply color adjustments
        color_adjustments = adjustments.get("color_adjustments", [])
        for adj in color_adjustments:
            try:
                obj_id = adj.get("object_id")
                adjusted_color = adj.get("adjusted_color")
                
                if obj_id in object_map and adjusted_color:
                    obj = object_map[obj_id]
                    obj.color = Color(name=adjusted_color)
                    
                    # Update shape fill colors
                    if obj.shapes:
                        for shape in obj.shapes:
                            for attr in shape.attributes:
                                if attr.name == "fill":
                                    attr.value = obj.color.hex_code
            except Exception as e:
                logger.warning(f"Error applying color adjustment: {e}")
        
        # Apply material adjustments
        material_adjustments = adjustments.get("material_adjustments", [])
        for adj in material_adjustments:
            try:
                obj_id = adj.get("object_id")
                adjusted_material = adj.get("adjusted_material")
                
                if obj_id in object_map and adjusted_material:
                    obj = object_map[obj_id]
                    obj.material = Material(name=adjusted_material, texture=adjusted_material)
            except Exception as e:
                logger.warning(f"Error applying material adjustment: {e}")
        
        # Apply background adjustment
        background_adjustment = adjustments.get("background_adjustment")
        if background_adjustment and background_adjustment.get("suggested_color"):
            try:
                color = Color(name=background_adjustment["suggested_color"])
                coherent_scene.background_color = color.hex_code
            except Exception as e:
                logger.warning(f"Error applying background adjustment: {e}")
        
        return coherent_scene
    
    def _apply_clip_optimizations(self, scene: Scene, optimizations: Dict[str, Any]) -> Scene:
        """
        Apply CLIP optimizations to a scene.
        
        Args:
            scene: Original scene
            optimizations: CLIP optimization data
            
        Returns:
            Optimized scene
        """
        # Create a copy of the scene to avoid modifying the original
        optimized_scene = Scene(
            id=scene.id,
            prompt=scene.prompt,
            background_color=scene.background_color,
            objects=scene.objects.copy(),
            width=scene.width,
            height=scene.height,
            patterns=scene.patterns.copy() if hasattr(scene, 'patterns') and scene.patterns else {},
            defs=scene.defs.copy() if hasattr(scene, 'defs') and scene.defs else [],
            special_elements=scene.special_elements.copy() if hasattr(scene, 'special_elements') else []
        )
        
        # Apply semantic emphasis
        semantic_emphasis = optimizations.get("semantic_emphasis", [])
        for emphasis in semantic_emphasis:
            try:
                key_term = emphasis.get("key_term")
                emphasis_method = emphasis.get("emphasis_method")
                
                if key_term and emphasis_method:
                    # Find objects that match the key term
                    for obj in optimized_scene.objects:
                        if key_term.lower() in obj.name.lower():
                            # Apply emphasis method
                            if emphasis_method == "increase_size":
                                obj.size *= 1.2
                            elif emphasis_method == "increase_contrast":
                                if obj.color:
                                    # Make color more saturated/vivid
                                    pass  # Would implement color adjustment here
                            elif emphasis_method == "center":
                                obj.position = (0.5, 0.5)
                            elif emphasis_method == "bring_forward":
                                obj.z_index += 10
            except Exception as e:
                logger.warning(f"Error applying semantic emphasis: {e}")
        
        # Apply visual adjustments
        visual_adjustments = optimizations.get("visual_adjustments", [])
        for adjustment in visual_adjustments:
            try:
                element_type = adjustment.get("element_type")
                adjustment_type = adjustment.get("adjustment")
                
                # Apply to all objects of the specified type
                for obj in optimized_scene.objects:
                    if element_type.lower() in obj.name.lower() or element_type.lower() == "all":
                        if adjustment_type == "add_stroke":
                            # Add stroke to all shapes
                            for shape in obj.shapes:
                                # Find stroke attributes
                                has_stroke = False
                                for attr in shape.attributes:
                                    if attr.name == "stroke":
                                        has_stroke = True
                                        break
                                
                                if not has_stroke:
                                    shape.attributes.append(Attribute("stroke", "#000000"))
                                    shape.attributes.append(Attribute("stroke-width", 2))
                        
                        elif adjustment_type == "add_glow":
                            if hasattr(obj, 'add_visual_effect'):
                                obj.add_visual_effect("glow")
                            else:
                                # Add a glow filter
                                filter_id = f"glow_{obj.id}"
                                if hasattr(optimized_scene, 'defs'):
                                    glow_def = {
                                        "type": "filter",
                                        "id": filter_id,
                                        "elements": [{
                                            "type": "feGaussianBlur",
                                            "stdDeviation": 2,
                                            "result": "blur"
                                        }, {
                                            "type": "feFlood",
                                            "flood-color": "#ffff00",
                                            "flood-opacity": 0.3,
                                            "result": "glow_color"
                                        }, {
                                            "type": "feComposite",
                                            "in": "glow_color",
                                            "in2": "blur",
                                            "operator": "in",
                                            "result": "glow_blur"
                                        }, {
                                            "type": "feMerge",
                                            "children": [{
                                                "type": "feMergeNode",
                                                "in": "glow_blur"
                                            }, {
                                                "type": "feMergeNode",
                                                "in": "SourceGraphic"
                                            }]
                                        }]
                                    }
                                    optimized_scene.defs.append(glow_def)
                                    
                                    # Apply filter to shapes
                                    for shape in obj.shapes:
                                        has_filter = False
                                        for attr in shape.attributes:
                                            if attr.name == "filter":
                                                attr.value = f"url(#{filter_id})"
                                                has_filter = True
                                                break
                                        
                                        if not has_filter:
                                            shape.attributes.append(Attribute("filter", f"url(#{filter_id})"))
                        
                        elif adjustment_type == "sharpen":
                            # For SVG, sharpening can mean more defined edges
                            for shape in obj.shapes:
                                for attr in shape.attributes:
                                    if attr.name == "stroke-width" and isinstance(attr.value, (int, float, str)):
                                        try:
                                            attr.value = float(attr.value) + 0.5
                                        except ValueError:
                                            pass
            except Exception as e:
                logger.warning(f"Error applying visual adjustment: {e}")
        
        # Apply metadata enhancements
        metadata_enhancements = optimizations.get("metadata_enhancements", [])
        for enhancement in metadata_enhancements:
            try:
                field = enhancement.get("field")
                content = enhancement.get("content")
                
                if field == "title":
                    # SVG title is handled in get_svg_code, so we need a way to pass it
                    optimized_scene.title_override = content
                elif field == "description":
                    optimized_scene.desc_override = content
            except Exception as e:
                logger.warning(f"Error applying metadata enhancement: {e}")
        
        return optimized_scene
    
    def _get_cache_key(self, scene: Scene) -> str:
        """
        Generate a cache key for the scene.
        
        Args:
            scene: Scene to generate key for
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create a deterministic hash of the scene
        scene_data = f"{scene.id}:{scene.prompt}"
        
        # Add object data to make the hash more specific
        for obj in scene.objects:
            scene_data += f":{obj.name}"
            if obj.color:
                scene_data += f":{obj.color.name}"
            
        return hashlib.md5(scene_data.encode()).hexdigest()
    
    def _cache_result(self, cache_key: str, scene: Scene) -> None:
        """Cache a scene elaboration result."""
        # Add to memory cache
        self._elaboration_cache[cache_key] = scene
        
        # Limit cache size
        if len(self._elaboration_cache) > self.max_cache_size:
            # Remove oldest item (arbitrary but consistent)
            oldest_key = next(iter(self._elaboration_cache))
            del self._elaboration_cache[oldest_key]
    
    def _load_reference_database(self) -> Optional[Dict[str, Any]]:
        """
        Load visual reference database if available.
        
        Returns:
            Reference database or None if not available
        """
        if not self.reference_database:
            return None
            
        try:
            with open(self.reference_database, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading reference database: {str(e)}")
            return None
    
    def shutdown(self) -> None:
        """Clean up resources on shutdown."""
        logger.info("Shutting down SceneElaborator")
        
        # Shutdown batch processor
        if hasattr(self, 'batch_processor'):
            self.batch_processor.stop(wait_complete=True)
        
        # Clear cache
        self._elaboration_cache.clear()
        
        # Force garbage collection
        gc.collect()