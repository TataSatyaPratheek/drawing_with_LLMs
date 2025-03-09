"""
Enhanced Visual Object Model Module
========================
This module defines the enhanced VisualObject class for representing visual elements.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

from svg_prompt_analyzer.models.color import Color
from svg_prompt_analyzer.models.material import Material
from svg_prompt_analyzer.models.shape import Shape, Attribute


class ObjectType(Enum):
    """Enumeration of object types that can be recognized in prompts."""
    GEOMETRIC = "geometric"
    NATURE = "nature"
    CLOTHING = "clothing"
    ARCHITECTURE = "architecture"
    ABSTRACT = "abstract"
    LANDSCAPE = "landscape"
    WEATHER = "weather"
    CELESTIAL = "celestial"
    TIME = "time"
    OTHER = "other"


# Dictionary of keywords for each object type - this will be imported from nlp_utils in the actual code
OBJECT_TYPE_KEYWORDS = {
    ObjectType.GEOMETRIC: [
        "circle", "square", "rectangle", "triangle", "polygon", "hexagon", 
        "octagon", "pentagon", "star", "oval", "line", "curve", "spiral",
        "grid", "pattern", "trapezoid", "rhombus", "dodecahedron", "pyramid", 
        "cone", "crescent", "parallelogram", "prism", "arc", "spire"
    ],
    ObjectType.NATURE: [
        "tree", "flower", "leaf", "mountain", "river", "lake", "ocean", 
        "forest", "plant", "rock", "stone", "animal", "bird", "fish", "lagoon",
        "wood", "peak", "vistas", "sea", "desert", "expanse"
    ],
    ObjectType.CLOTHING: [
        "shirt", "pants", "dress", "skirt", "hat", "coat", "jacket", "scarf",
        "shoe", "boot", "glove", "sock", "belt", "tie", "button", "zipper",
        "collar", "pocket", "overalls", "trim", "tassel", "dungarees", 
        "neckerchief", "trousers", "overcoat", "lining", "fringed", "clasps",
        "harlequin", "ribbed"
    ],
    ObjectType.ARCHITECTURE: [
        "house", "building", "tower", "bridge", "wall", "window", "door", 
        "roof", "floor", "ceiling", "arch", "column", "stair", "lighthouse",
        "beacon"
    ],
    ObjectType.LANDSCAPE: [
        "mountain", "hill", "valley", "plain", "desert", "island", "beach",
        "cliff", "plateau", "canyon", "field", "meadow", "forest", "peak",
        "vistas", "expanse"
    ],
    ObjectType.WEATHER: [
        "rain", "snow", "cloud", "storm", "wind", "fog", "mist", "hail",
        "lightning", "thunder", "hurricane", "tornado", "cyclone", "rainbow",
        "snowy", "cloudy", "overcast"
    ],
    ObjectType.CELESTIAL: [
        "star", "moon", "sun", "planet", "galaxy", "comet", "asteroid", 
        "meteor", "constellation", "nebula", "starlit", "night", "dusk", "dawn",
        "evening"
    ]
}


@dataclass
class VisualObject:
    """Enhanced class representing a visual object extracted from the prompt."""
    id: str
    name: str
    object_type: ObjectType
    shapes: List[Shape] = field(default_factory=list)
    color: Optional[Color] = None
    material: Optional[Material] = None
    attributes: List[Attribute] = field(default_factory=list)
    position: Tuple[float, float] = (0.5, 0.5)  # Normalized position (x, y)
    size: float = 0.2  # Normalized size
    z_index: int = 0  # Layering
    
    # Enhanced properties for special relationships
    connected_to: Optional['VisualObject'] = None  # For objects connected to others
    connection_color: Optional[str] = None  # Color for connection lines
    visual_effects: Dict[str, Any] = field(default_factory=dict)  # Visual effects to apply
    
    def get_svg_elements(self, canvas_width: int, canvas_height: int) -> List[str]:
        """Generate SVG elements for this object with enhanced features."""
        elements = []
        
        # Check for special visual effects
        has_shimmering = "shimmering" in self.name.lower() or self.visual_effects.get("shimmering", False)
        has_ribbed = "ribbed" in self.name.lower() or self.visual_effects.get("ribbed", False)
        
        # Calculate absolute position and size
        x = self.position[0] * canvas_width
        y = self.position[1] * canvas_height
        size = self.size * min(canvas_width, canvas_height)
        
        # Add shapes
        if self.shapes:
            # Apply visual effects to shapes if needed
            if has_shimmering and len(self.shapes) > 0:
                self.shapes[0].visual_effects["shimmering"] = True
            
            if has_ribbed and len(self.shapes) > 0:
                self.shapes[0].visual_effects["ribbed"] = True
            
            for shape in self.shapes:
                elements.append(shape.get_svg_element(x, y, size))
        else:
            # Create default shape if none specified
            default_shape = Shape(
                shape_type="rectangle",
                attributes=[
                    Attribute("fill", self.color.hex_code if self.color else "#808080"),
                    Attribute("stroke", "#000000"),
                    Attribute("stroke-width", 1)
                ]
            )
            
            # Apply visual effects to default shape
            if has_shimmering:
                default_shape.visual_effects["shimmering"] = True
            
            if has_ribbed:
                default_shape.visual_effects["ribbed"] = True
            
            elements.append(default_shape.get_svg_element(x, y, size))
        
        # Handle connection lines if this object is connected to another
        if self.connected_to:
            target = self.connected_to
            connection_color = self.connection_color or "#888888"
            
            # Draw a connecting line
            target_x = target.position[0] * canvas_width
            target_y = target.position[1] * canvas_height
            
            # Create a curved path for the connection
            # Calculate control point for curve
            cx = (x + target_x) / 2
            cy = (y + target_y) / 2 - 20  # Curve upward slightly
            
            connection_line = f'<path d="M{x},{y} Q{cx},{cy} {target_x},{target_y}" ' + \
                              f'stroke="{connection_color}" stroke-width="2" fill="none" />'
            
            elements.append(connection_line)
        
        return elements
    
    def add_attribute(self, name: str, value: Any) -> None:
        """Add a new attribute to the object."""
        self.attributes.append(Attribute(name=name, value=value))
    
    def get_attribute(self, name: str) -> Optional[Any]:
        """Get attribute value by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr.value
        return None
    
    def add_visual_effect(self, effect_name: str, value: Any = True) -> None:
        """Add a visual effect to the object."""
        self.visual_effects[effect_name] = value
    
    def has_visual_effect(self, effect_name: str) -> bool:
        """Check if the object has a specific visual effect."""
        return self.visual_effects.get(effect_name, False)