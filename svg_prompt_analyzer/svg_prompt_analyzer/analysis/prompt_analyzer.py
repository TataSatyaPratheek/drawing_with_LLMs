"""
Enhanced Prompt Analyzer Module
====================
This module contains the enhanced PromptAnalyzer class for analyzing text prompts.
"""

import logging
import re
from typing import List, Optional, Dict, Any, Tuple

from svg_prompt_analyzer.analysis.nlp_utils import (
    initialize_nlp, COLORS, MATERIALS, GEOMETRIC_SHAPES, 
    VISUAL_EFFECTS, CLOTHING_ITEMS, SCENE_TYPES, TIME_REFERENCES,
    COMPOUND_COLORS
)
from svg_prompt_analyzer.analysis.spatial_analyzer import SpatialAnalyzer
from svg_prompt_analyzer.models.scene import Scene
from svg_prompt_analyzer.models.visual_object import VisualObject, ObjectType, OBJECT_TYPE_KEYWORDS
from svg_prompt_analyzer.models.color import Color
from svg_prompt_analyzer.models.material import Material
from svg_prompt_analyzer.models.shape import Shape, Attribute

logger = logging.getLogger(__name__)


class PromptAnalyzer:
    """
    Enhanced class for analyzing text prompts to extract information for SVG generation.
    """
    
    def __init__(self):
        """Initialize the analyzer with necessary resources."""
        self.nlp = initialize_nlp()
        self.spatial_analyzer = SpatialAnalyzer()
        
    def analyze_prompt(self, prompt_id: str, prompt_text: str) -> Scene:
        """
        Analyze a prompt and extract information for generating an SVG.
        
        Args:
            prompt_id: ID of the prompt
            prompt_text: Text of the prompt
            
        Returns:
            Scene object containing all extracted information
        """
        logger.info(f"Analyzing prompt {prompt_id}: {prompt_text}")
        
        # Process text with spaCy
        doc = self.nlp(prompt_text)
        
        # Preprocess prompt to handle special cases
        preprocessed_info = self.preprocess_prompt(prompt_text)
        
        # Identify main objects and their attributes
        objects = self.extract_objects(doc, prompt_id, preprocessed_info)
        
        # Identify spatial relationships
        self.spatial_analyzer.extract_spatial_relationships(doc, objects)
        
        # Determine background
        background_color = self.determine_background(doc, prompt_text)
        
        # Create patterns if needed
        patterns = self.create_patterns_for_materials(objects)
        
        # Create scene
        scene = Scene(
            id=prompt_id,
            prompt=prompt_text,
            background_color=background_color,
            objects=objects,
            patterns=patterns
        )
        
        # Adjust layout based on spatial relationships and scene type
        self.spatial_analyzer.layout_objects(scene)
        
        return scene
    
    def preprocess_prompt(self, prompt_text: str) -> Dict[str, Any]:
        """
        Preprocess the prompt to identify special cases and complex structures.
        
        Args:
            prompt_text: The raw prompt text
            
        Returns:
            Dictionary containing preprocessed information
        """
        info = {}
        
        # Check for specific complex patterns in the test dataset
        
        # Handle "X and Y" color pairs
        color_pair_match = re.search(r'(\w+)(?:\s+and\s+|\s*&\s*)(\w+)\s+(\w+)', prompt_text.lower())
        if color_pair_match:
            color1, color2, item = color_pair_match.groups()
            if color1 in COLORS and color2 in COLORS:
                info['color_pair'] = (color1, color2, item)
        
        # Check for specific scene types
        for scene_type in SCENE_TYPES:
            if scene_type in prompt_text.lower():
                info['scene_type'] = scene_type
                break
        
        # Check for time references
        for time_ref in TIME_REFERENCES:
            if time_ref in prompt_text.lower():
                info['time_of_day'] = time_ref
                break
        
        # Check for special visual effects
        for effect in VISUAL_EFFECTS:
            if effect in prompt_text.lower():
                if 'visual_effects' not in info:
                    info['visual_effects'] = []
                info['visual_effects'].append(effect)
        
        # Check for specific patterns like "X forming Y" or "X connected by Y"
        forming_match = re.search(r'(\w+)\s+(\w+)(?:s)?\s+forming\s+a(?:n)?\s+(\w+)', prompt_text.lower())
        if forming_match:
            info['forming_pattern'] = forming_match.groups()
        
        connected_match = re.search(r'(\w+)(?:\s+\w+)?\s+connected\s+by\s+(\w+)', prompt_text.lower())
        if connected_match:
            info['connected_pattern'] = connected_match.groups()
        
        # Check for "X facing Y" pattern
        facing_match = re.search(r'(\w+)(?:\s+\w+)?\s+facing\s+(?:the\s+)?(\w+)', prompt_text.lower())
        if facing_match:
            info['facing_pattern'] = facing_match.groups()
        
        # Handle "as X falls" patterns (like "evening falls")
        falls_match = re.search(r'as\s+(\w+)\s+falls', prompt_text.lower())
        if falls_match:
            info['time_transition'] = falls_match.group(1)
        
        return info
    
    def extract_objects(self, doc, prompt_id: str, preprocessed_info: Dict[str, Any] = None) -> List[VisualObject]:
        """
        Extract objects from the prompt with enhanced detection.
        
        Args:
            doc: spaCy processed document
            prompt_id: Prompt identifier
            preprocessed_info: Optional preprocessed information
            
        Returns:
            List of VisualObject instances
        """
        objects = []
        preprocessed_info = preprocessed_info or {}
        
        # Extract noun phrases
        noun_phrases = [chunk for chunk in doc.noun_chunks]
        
        # Handle special case: color pairs (like "ivory and ebony harlequin trousers")
        if 'color_pair' in preprocessed_info:
            color1, color2, item = preprocessed_info['color_pair']
            
            # Create a special object for the color pair item
            obj = self.create_color_pair_object(prompt_id, color1, color2, item)
            if obj:
                objects.append(obj)
                # Skip normal noun phrase processing if we've created a special object
                # that covers the entire prompt
                if len(noun_phrases) <= 3:  # Simple heuristic
                    return objects
        
        # Handle special case: forming patterns (like "crimson rectangles forming a chaotic grid")
        if 'forming_pattern' in preprocessed_info:
            color_or_material, shape, pattern = preprocessed_info['forming_pattern']
            obj = self.create_forming_pattern_object(prompt_id, color_or_material, shape, pattern)
            if obj:
                objects.append(obj)
                # Skip normal noun phrase processing if we've created a special object
                if len(noun_phrases) <= 3:  # Simple heuristic
                    return objects
        
        # Handle connected pattern (like "a maroon dodecahedron interwoven with teal threads")
        if 'connected_pattern' in preprocessed_info:
            main_obj, connector = preprocessed_info['connected_pattern']
            # This will be handled during normal processing, but we'll flag it
        
        # Process noun phrases
        for i, np in enumerate(noun_phrases):
            # Skip if the noun phrase is likely part of a spatial relationship
            if self.spatial_analyzer.is_spatial_phrase(np.text):
                continue
                
            # Determine object type
            object_type = self.determine_object_type(np.text)
            
            # Extract color
            color_info = self.extract_color(np.text, doc.text)
            
            # Extract material
            material_info = self.extract_material(np.text)
            
            # Extract shapes
            shapes = self.extract_shapes(np.text, color_info)
            
            # Extract visual effects
            visual_effects = self.extract_visual_effects(np.text, preprocessed_info)
            
            # Create visual object
            obj = VisualObject(
                id=f"{prompt_id}_{i}",
                name=np.text,
                object_type=object_type,
                shapes=shapes,
                color=color_info,
                material=material_info,
                z_index=i,  # Simple initial z-ordering
                visual_effects=visual_effects
            )
            
            # Add additional attributes
            for token in np:
                if token.pos_ == "ADJ" and token.text.lower() not in COLORS:
                    obj.add_attribute("modifier", token.text)
            
            objects.append(obj)
        
        # If no objects found, try to create a default one
        if not objects:
            obj = VisualObject(
                id=f"{prompt_id}_default",
                name=doc.text,
                object_type=ObjectType.ABSTRACT,
                z_index=0
            )
            objects.append(obj)
        
        # Post-process objects for special relationships
        self.process_special_relationships(objects, preprocessed_info, doc.text)
        
        return objects
    
    def create_color_pair_object(self, prompt_id: str, color1: str, color2: str, item: str) -> Optional[VisualObject]:
        """
        Create a special object for a color pair pattern (like "ivory and ebony trousers").
        
        Args:
            prompt_id: The prompt identifier
            color1: First color name
            color2: Second color name
            item: The item being described
            
        Returns:
            VisualObject instance or None if not applicable
        """
        # Check if the item is a recognized object type
        object_type = None
        for ot, keywords in OBJECT_TYPE_KEYWORDS.items():
            if item in keywords or f"{item}s" in keywords:
                object_type = ot
                break
        
        if not object_type:
            # Check if it's a clothing item
            if item in CLOTHING_ITEMS or f"{item}s" in CLOTHING_ITEMS:
                object_type = ObjectType.CLOTHING
            else:
                object_type = ObjectType.ABSTRACT
        
        # Create the object
        obj = VisualObject(
            id=f"{prompt_id}_duo",
            name=f"{color1} and {color2} {item}",
            object_type=object_type,
            z_index=0
        )
        
        # Set primary color (arbitrarily the first one)
        obj.color = Color(name=color1)
        
        # Add secondary color as an attribute
        obj.add_attribute("secondary_color", color2)
        
        # Note the pattern type
        if "harlequin" in item:
            obj.add_visual_effect("harlequin")
            shape = Shape(
                shape_type="rectangle",  # Base shape for harlequin pattern
                attributes=[
                    Attribute("fill", obj.color.hex_code),
                    Attribute("stroke", "#000000"),
                    Attribute("stroke-width", 1)
                ],
                visual_effects={"harlequin": True}
            )
            obj.shapes.append(shape)
        elif "checkered" in item:
            obj.add_visual_effect("checkered")
            shape = Shape(
                shape_type="rectangle",  # Base shape for checkered pattern
                attributes=[
                    Attribute("fill", obj.color.hex_code),
                    Attribute("stroke", "#000000"),
                    Attribute("stroke-width", 1)
                ],
                visual_effects={"checkered": True}
            )
            obj.shapes.append(shape)
        
        return obj
    
    def create_forming_pattern_object(self, prompt_id: str, color_or_material: str, shape: str, pattern: str) -> Optional[VisualObject]:
        """
        Create a special object for a forming pattern (like "crimson rectangles forming a chaotic grid").
        
        Args:
            prompt_id: The prompt identifier
            color_or_material: Color or material description
            shape: Shape being repeated
            pattern: Pattern being formed
            
        Returns:
            VisualObject instance or None if not applicable
        """
        # Determine if the first word is a color or material
        color_info = None
        material_info = None
        
        if color_or_material in COLORS:
            color_info = Color(name=color_or_material)
        elif color_or_material in MATERIALS:
            material_info = Material(name=color_or_material, texture=color_or_material)
        
        # Create the object
        obj = VisualObject(
            id=f"{prompt_id}_pattern",
            name=f"{color_or_material} {shape} forming {pattern}",
            object_type=ObjectType.GEOMETRIC,
            color=color_info,
            material=material_info,
            z_index=0
        )
        
        # Add shape
        if shape in GEOMETRIC_SHAPES or f"{shape}s" in GEOMETRIC_SHAPES:
            shape_type = shape.rstrip('s')  # Remove plural 's' if present
            
            shape_obj = Shape(
                shape_type=shape_type,
                attributes=[
                    Attribute("fill", color_info.hex_code if color_info else "#808080"),
                    Attribute("stroke", "#000000"),
                    Attribute("stroke-width", 1)
                ]
            )
            obj.shapes.append(shape_obj)
        
        # Add pattern effect
        if "grid" in pattern:
            obj.add_visual_effect("grid")
        elif "chaotic" in pattern or "disordered" in pattern:
            obj.add_visual_effect("disordered")
        elif "spiral" in pattern:
            obj.add_visual_effect("spiral")
        elif "array" in pattern:
            obj.add_visual_effect("array")
        
        return obj
    
    def process_special_relationships(self, objects: List[VisualObject], preprocessed_info: Dict[str, Any], prompt_text: str) -> None:
        """
        Process special relationships between objects.
        
        Args:
            objects: List of visual objects
            preprocessed_info: Preprocessed information
            prompt_text: Original prompt text
        """
        # Handle connected patterns
        if 'connected_pattern' in preprocessed_info:
            main_obj_name, connector_name = preprocessed_info['connected_pattern']
            
            # Find main object and connector
            main_obj = None
            connector_obj = None
            
            for obj in objects:
                obj_name_lower = obj.name.lower()
                if main_obj_name in obj_name_lower:
                    main_obj = obj
                if connector_name in obj_name_lower:
                    connector_obj = obj
            
            # If both found, establish connection
            if main_obj and connector_obj:
                connector_obj.connected_to = main_obj
                
                # Set color for connection lines
                if connector_obj.color:
                    connector_obj.connection_color = connector_obj.color.hex_code
        
        # Handle "X circling Y" pattern
        if "circling" in prompt_text.lower():
            # Find objects involved in the circling relationship
            for i, obj1 in enumerate(objects):
                if "circling" in obj1.name.lower():
                    # Find object being circled
                    for j, obj2 in enumerate(objects):
                        if i != j:
                            # Assume the other object is being circled
                            obj1.add_visual_effect("circling")
                            obj1.add_attribute("circling_target", obj2.id)
                            break
    
    def determine_object_type(self, text: str) -> ObjectType:
        """
        Determine the type of object from text with enhanced recognition.
        
        Args:
            text: Text to analyze
            
        Returns:
            ObjectType enum value
        """
        text_lower = text.lower()
        
        # Check each object type's keywords
        for obj_type, keywords in OBJECT_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return obj_type
        
        # Additional checks for clothing items
        for item in CLOTHING_ITEMS:
            if item in text_lower:
                return ObjectType.CLOTHING
        
        # Default to abstract if no match
        return ObjectType.ABSTRACT
    
    def extract_color(self, noun_phrase: str, full_prompt: str = "") -> Optional[Color]:
        """
        Extract color information from text with enhanced detection.
        
        Args:
            noun_phrase: Noun phrase to analyze
            full_prompt: Optional full prompt text for context
            
        Returns:
            Color instance or None if no color found
        """
        text_lower = noun_phrase.lower()
        
        # Check for compound colors first
        for compound, hex_code in COMPOUND_COLORS.items():
            if compound in text_lower or compound in full_prompt.lower():
                return Color(name=compound, hex_code=hex_code)
        
        # Check for single colors
        for color in COLORS:
            if color in text_lower:
                return Color(name=color)
        
        return None
    
    def extract_material(self, text: str) -> Optional[Material]:
        """
        Extract material information from text with enhanced detection.
        
        Args:
            text: Text to analyze
            
        Returns:
            Material instance or None if no material found
        """
        text_lower = text.lower()
        
        # Check for material words
        for material in MATERIALS:
            if material in text_lower:
                # Check for transparency
                transparency = 0.0
                if "translucent" in text_lower:
                    transparency = 0.3
                elif "transparent" in text_lower:
                    transparency = 0.7
                elif "shimmering" in text_lower:
                    transparency = 0.1
                
                return Material(name=material, texture=material, transparency=transparency)
        
        return None
    
    def extract_shapes(self, text: str, color_info: Optional[Color] = None) -> List[Shape]:
        """
        Extract shape information from text with enhanced shape detection.
        
        Args:
            text: Text to analyze
            color_info: Optional color information
            
        Returns:
            List of Shape instances
        """
        shapes = []
        text_lower = text.lower()
        
        # Check for specific shapes
        for shape_word in GEOMETRIC_SHAPES:
            if shape_word in text_lower:
                # Get color for the shape
                fill_color = color_info.hex_code if color_info else "#808080"
                
                # Create shape with attributes
                shape = Shape(
                    shape_type=shape_word.rstrip('s'),  # Remove plural 's' if present
                    attributes=[
                        Attribute(name="fill", value=fill_color),
                        Attribute(name="stroke", value="#000000"),
                        Attribute(name="stroke-width", value=1)
                    ]
                )
                shapes.append(shape)
        
        return shapes
    
    def extract_visual_effects(self, text: str, preprocessed_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract visual effects from text.
        
        Args:
            text: Text to analyze
            preprocessed_info: Preprocessed information
            
        Returns:
            Dictionary of visual effects
        """
        effects = {}
        text_lower = text.lower()
        
        # Check for effects in text
        for effect in VISUAL_EFFECTS:
            if effect in text_lower:
                effects[effect] = True
        
        # Add effects from preprocessed info
        if 'visual_effects' in preprocessed_info:
            for effect in preprocessed_info['visual_effects']:
                if effect in text_lower:
                    effects[effect] = True
        
        return effects
    
    def determine_background(self, doc, prompt_text: str) -> str:
        """
        Determine background color or pattern based on prompt with enhanced scene detection.
        
        Args:
            doc: spaCy processed document
            prompt_text: Original prompt text
            
        Returns:
            Background color hex code
        """
        # Check for explicit background references
        background_colors = {
            "night": "#191970",     # Midnight blue
            "snow": "#FFFAFA",      # Snow white
            "forest": "#228B22",    # Forest green
            "ocean": "#1E90FF",     # Dodger blue
            "sea": "#006994",       # Sea blue
            "sky": "#87CEEB",       # Sky blue
            "cloudy": "#708090",    # Slate gray
            "overcast": "#708090",  # Slate gray
            "dusk": "#4B0082",      # Indigo
            "evening": "#4B0082",   # Indigo
            "plain": "#F5F5F5",     # White smoke
            "desert": "#EDC9AF",    # Desert sand
            "white desert": "#F5F5F5"  # White desert
        }
        
        # Start with default
        background = "#F5F5F5"  # Light gray background instead of white
        
        # Check for scene setting words
        scene_words = list(background_colors.keys())
        
        for word in scene_words:
            if word in prompt_text.lower():
                background = background_colors.get(word, background)
                break
        
        # Check for time of day
        if "evening falls" in prompt_text.lower():
            background = "#4B0082"  # Evening/dusk indigo
        
        # Specific scene type checks
        if "expanse" in prompt_text.lower() and "white" in prompt_text.lower() and "desert" in prompt_text.lower():
            background = "#F5F5F5"  # White desert
        
        return background
    
    def create_patterns_for_materials(self, objects: List[VisualObject]) -> Dict[str, str]:
        """
        Create SVG patterns for materials used in objects.
        
        Args:
            objects: List of visual objects
            
        Returns:
            Dictionary mapping pattern IDs to pattern definitions
        """
        patterns = {}
        
        # Create patterns for different materials
        materials_in_use = set()
        for obj in objects:
            if obj.material and obj.material.texture:
                materials_in_use.add(obj.material.texture)
        
        # Create pattern definitions
        for material in materials_in_use:
            if material == "silk":
                patterns["patternSilk"] = '''
<pattern id="patternSilk" patternUnits="userSpaceOnUse" width="20" height="20">
    <rect width="20" height="20" fill="#FFFFFF" opacity="0.9"/>
    <path d="M0,0 L20,20 M20,0 L0,20" stroke="#F5F5F5" stroke-width="0.5"/>
</pattern>'''
            
            elif material == "wool":
                patterns["patternWool"] = '''
<pattern id="patternWool" patternUnits="userSpaceOnUse" width="10" height="10">
    <rect width="10" height="10" fill="#FFFFFF"/>
    <path d="M0,0 Q2.5,2.5 5,0 Q7.5,2.5 10,0 M0,5 Q2.5,7.5 5,5 Q7.5,7.5 10,5" 
          stroke="#EFEFEF" stroke-width="1" fill="none"/>
</pattern>'''
            
            elif material == "corduroy":
                patterns["patternCorduroy"] = '''
<pattern id="patternCorduroy" patternUnits="userSpaceOnUse" width="8" height="8">
    <rect width="8" height="8" fill="#FFFFFF"/>
    <rect x="0" y="0" width="8" height="1" fill="#F0F0F0"/>
    <rect x="0" y="3" width="8" height="1" fill="#F0F0F0"/>
    <rect x="0" y="6" width="8" height="1" fill="#F0F0F0"/>
</pattern>'''
            
            elif material == "fur":
                patterns["patternFur"] = '''
<pattern id="patternFur" patternUnits="userSpaceOnUse" width="20" height="20">
    <rect width="20" height="20" fill="#FFFFFF"/>
    <path d="M5,0 L5,8 M10,0 L10,10 M15,0 L15,7 M0,5 L8,5 M0,10 L10,10 M0,15 L7,15" 
          stroke="#F8F8F8" stroke-width="1.5" stroke-linecap="round"/>
</pattern>'''
            
            elif material == "cashmere":
                patterns["patternCashmere"] = '''
<pattern id="patternCashmere" patternUnits="userSpaceOnUse" width="20" height="20">
    <rect width="20" height="20" fill="#FFFFFF"/>
    <path d="M0,0 Q5,5 10,0 Q15,5 20,0 M0,10 Q5,15 10,10 Q15,15 20,10 M0,20 Q5,15 10,20 Q15,15 20,20" 
          stroke="#F0F0F0" stroke-width="1" fill="none"/>
</pattern>'''
            
            elif material == "satin":
                patterns["patternSatin"] = '''
<pattern id="patternSatin" patternUnits="userSpaceOnUse" width="20" height="20">
    <rect width="20" height="20" fill="#FFFFFF" opacity="0.9"/>
    <path d="M0,0 L20,20 M20,0 L0,20" stroke="#FFFFFF" stroke-width="1.5" opacity="0.7"/>
</pattern>'''
        
        # Check for visual effects that need patterns
        effects_in_use = set()
        for obj in objects:
            if hasattr(obj, 'visual_effects'):
                for effect in obj.visual_effects:
                    effects_in_use.add(effect)
        
        # Create patterns for effects
        for effect in effects_in_use:
            if effect == "shimmering":
                patterns["patternShimmering"] = '''
<pattern id="patternShimmering" patternUnits="userSpaceOnUse" width="40" height="40">
    <rect width="40" height="40" fill="#FFFFFF" opacity="0.6"/>
    <path d="M0,0 L40,40 M40,0 L0,40" stroke="white" stroke-width="0.5" opacity="0.7"/>
    <rect width="40" height="40" fill="url(#shimmerGradient)" opacity="0.5"/>
</pattern>
<linearGradient id="shimmerGradient" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="white" stop-opacity="0.3"/>
    <stop offset="50%" stop-color="white" stop-opacity="0.5"/>
    <stop offset="100%" stop-color="white" stop-opacity="0.3"/>
</linearGradient>'''
            
            elif effect == "ribbed":
                patterns["patternRibbed"] = '''
<pattern id="patternRibbed" patternUnits="userSpaceOnUse" width="10" height="10">
    <rect width="10" height="10" fill="#FFFFFF"/>
    <rect x="0" y="0" width="10" height="2" fill="#F0F0F0" opacity="0.8"/>
    <rect x="0" y="4" width="10" height="2" fill="#F0F0F0" opacity="0.8"/>
    <rect x="0" y="8" width="10" height="2" fill="#F0F0F0" opacity="0.8"/>
</pattern>'''
        
        return patterns