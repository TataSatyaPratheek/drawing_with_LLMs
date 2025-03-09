"""
Enhanced Shape Model Module
================
This module defines enhanced Shape and Attribute classes for representing geometric shapes.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional


@dataclass
class Attribute:
    """Class representing object attributes."""
    name: str
    value: Any
    
    def __str__(self) -> str:
        return f"{self.name}: {self.value}"


@dataclass
class Shape:
    """Class representing a geometric shape with enhanced capabilities."""
    shape_type: str  # circle, rectangle, triangle, etc.
    attributes: List[Attribute] = field(default_factory=list)
    visual_effects: Dict[str, bool] = field(default_factory=dict)
    rotation: Optional[float] = None  # Rotation angle in degrees
    
    def get_svg_element(self, x: int, y: int, size: int) -> str:
        """Generate enhanced SVG element for this shape."""
        svg_code = ""
        
        # Get attributes as dictionary for easier access
        attr_dict = {attr.name: attr.value for attr in self.attributes}
        
        # Apply rotation if specified
        rotation_transform = ""
        if self.rotation is not None:
            rotation_transform = f' transform="rotate({self.rotation} {x} {y})"'
        
        # Configure shape based on type with enhanced shapes
        if self.shape_type == "circle":
            radius = size / 2
            svg_code = f'<circle cx="{x}" cy="{y}" r="{radius}" '
        
        elif self.shape_type == "rectangle":
            width = size
            height = size * 0.8  # Slightly shorter than width by default
            if "width" in attr_dict:
                width = attr_dict["width"]
            if "height" in attr_dict:
                height = attr_dict["height"]
            svg_code = f'<rect x="{x - width/2}" y="{y - height/2}" width="{width}" height="{height}" '
            
        elif self.shape_type == "triangle":
            # Equilateral triangle
            points = [
                (x, y - size/2),  # Top
                (x - size/2, y + size/2),  # Bottom left
                (x + size/2, y + size/2)   # Bottom right
            ]
            points_str = " ".join([f"{px},{py}" for px, py in points])
            svg_code = f'<polygon points="{points_str}" '
            
        elif self.shape_type == "polygon" or self.shape_type == "polygons":
            # Random polygon with 5-8 sides
            sides = random.randint(5, 8)
            radius = size / 2
            points = []
            for i in range(sides):
                angle = 2 * math.pi * i / sides
                px = x + radius * math.cos(angle)
                py = y + radius * math.sin(angle)
                points.append((px, py))
            points_str = " ".join([f"{px},{py}" for px, py in points])
            svg_code = f'<polygon points="{points_str}" '
            
        elif self.shape_type == "pyramid" or self.shape_type == "pyramids":
            # Simplified 2D pyramid (triangle)
            height = size
            base_width = size * 1.2
            points = [
                (x, y - height/2),  # Top
                (x - base_width/2, y + height/2),  # Bottom left
                (x + base_width/2, y + height/2)   # Bottom right
            ]
            points_str = " ".join([f"{px},{py}" for px, py in points])
            svg_code = f'<polygon points="{points_str}" '
            
        elif self.shape_type == "dodecahedron" or self.shape_type == "12-sided":
            # 12-sided polygon
            radius = size / 2
            points = []
            for i in range(12):
                angle = 2 * math.pi * i / 12
                px = x + radius * math.cos(angle)
                py = y + radius * math.sin(angle)
                points.append((px, py))
            points_str = " ".join([f"{px},{py}" for px, py in points])
            svg_code = f'<polygon points="{points_str}" '
            
        elif self.shape_type == "cone":
            # Simplified 2D cone
            height = size
            base_width = size * 0.8
            points = [
                (x, y - height/2),  # Top
                (x - base_width/2, y + height/2),  # Bottom left
                (x + base_width/2, y + height/2)   # Bottom right
            ]
            points_str = " ".join([f"{px},{py}" for px, py in points])
            svg_code = f'<polygon points="{points_str}" '
            
        elif self.shape_type == "trapezoid":
            top_width = size * 0.7
            bottom_width = size
            height = size * 0.6
            points = [
                (x - top_width/2, y - height/2),    # Top left
                (x + top_width/2, y - height/2),    # Top right
                (x + bottom_width/2, y + height/2), # Bottom right
                (x - bottom_width/2, y + height/2)  # Bottom left
            ]
            points_str = " ".join([f"{px},{py}" for px, py in points])
            svg_code = f'<polygon points="{points_str}" '
            
        elif self.shape_type == "parallelogram" or self.shape_type == "parallelograms":
            # Parallelogram
            width = size
            height = size * 0.6
            skew = size * 0.3
            points = [
                (x - width/2 + skew, y - height/2),  # Top left
                (x + width/2 + skew, y - height/2),  # Top right
                (x + width/2 - skew, y + height/2),  # Bottom right
                (x - width/2 - skew, y + height/2)   # Bottom left
            ]
            points_str = " ".join([f"{px},{py}" for px, py in points])
            svg_code = f'<polygon points="{points_str}" '
            
        elif self.shape_type == "crescent":
            # Crescent approximated with path
            radius = size / 2
            svg_code = f'''<path d="
                M {x + radius * 0.7} {y}
                a {radius} {radius} 0 1 1 0 0.1
                a {radius * 0.7} {radius * 0.7} 0 1 0 0 -0.1
            " '''
            
        elif self.shape_type == "arc" or self.shape_type == "arcs":
            # Arc
            radius = size / 2
            svg_code = f'''<path d="
                M {x - radius} {y}
                A {radius} {radius} 0 0 1 {x + radius} {y}
            " fill="none" '''
            
        elif self.shape_type == "prism" or self.shape_type == "prisms":
            # Simplified 3D prism effect using a hexagon with inner lines
            svg_code = f'<g>'
            radius = size / 2
            
            # Create hexagon points
            points = []
            for i in range(6):
                angle = 2 * math.pi * i / 6
                px = x + radius * math.cos(angle)
                py = y + radius * math.sin(angle)
                points.append((px, py))
            
            # Draw the hexagon
            points_str = " ".join([f"{px},{py}" for px, py in points])
            svg_code += f'<polygon points="{points_str}" '
            
            # Add fill, stroke attributes
            fill = attr_dict.get("fill", "#808080")
            stroke = attr_dict.get("stroke", "#000000")
            stroke_width = attr_dict.get("stroke-width", 1)
            
            svg_code += f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />'
            
            # Add inner lines for 3D effect
            for i in range(0, 6, 2):
                svg_code += f'<line x1="{points[i][0]}" y1="{points[i][1]}" x2="{points[(i+3)%6][0]}" y2="{points[(i+3)%6][1]}" stroke="{stroke}" stroke-width="{stroke_width/2}" stroke-opacity="0.5" />'
            
            svg_code += '</g>'
            return svg_code  # Return early since we've completed the SVG code
            
        elif self.shape_type == "spire":
            # Tall pointed structure
            width = size * 0.4
            height = size * 1.5
            points = [
                (x, y - height/2),  # Top
                (x - width/2, y + height/4),  # Middle left
                (x - width/3, y + height/2),  # Bottom left
                (x + width/3, y + height/2),  # Bottom right
                (x + width/2, y + height/4)   # Middle right
            ]
            points_str = " ".join([f"{px},{py}" for px, py in points])
            svg_code = f'<polygon points="{points_str}" '
            
        elif self.shape_type == "tower" or self.shape_type == "beacon":
            # Beacon tower
            base_width = size * 0.6
            top_width = size * 0.4
            height = size * 1.2
            
            # Tower body
            svg_code = f'<g>'
            svg_code += f'<rect x="{x - base_width/2}" y="{y - height/2 + base_width/2}" width="{base_width}" height="{height}" '
            
            # Add fill, stroke attributes for the body
            fill = attr_dict.get("fill", "#808080")
            stroke = attr_dict.get("stroke", "#000000")
            stroke_width = attr_dict.get("stroke-width", 1)
            
            svg_code += f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />'
            
            # Add a top for the beacon
            svg_code += f'<polygon points="{x - top_width/2},{y - height/2 + base_width/2} {x + top_width/2},{y - height/2 + base_width/2} {x},{y - height/2}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />'
            
            # Add a light at the top
            light_color = "#FFFF00"  # Yellow light
            svg_code += f'<circle cx="{x}" cy="{y - height/2 + base_width/4}" r="{base_width/4}" fill="{light_color}" stroke="{stroke}" stroke-width="{stroke_width/2}" />'
            
            svg_code += '</g>'
            return svg_code  # Return early since we've completed the SVG code
            
        else:
            # Default to rectangle if shape not recognized
            svg_code = f'<rect x="{x - size/2}" y="{y - size/2}" width="{size}" height="{size}" '
        
        # Add style attributes
        fill = attr_dict.get("fill", "#808080")  # Default gray
        stroke = attr_dict.get("stroke", "#000000")  # Default black
        stroke_width = attr_dict.get("stroke-width", 1)
        
        # Add visual effects
        if self.visual_effects.get("shimmering", False):
            # Shimmering effect with gradient
            gradient_id = f"shimmer{random.randint(1000, 9999)}"
            svg_code = f'''<defs>
                <linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="{fill}" stop-opacity="0.8" />
                    <stop offset="50%" stop-color="#FFFFFF" stop-opacity="0.9" />
                    <stop offset="100%" stop-color="{fill}" stop-opacity="0.8" />
                </linearGradient>
            </defs>
            {svg_code}'''
            fill = f"url(#{gradient_id})"
        
        if self.visual_effects.get("ribbed", False):
            # Apply ribbed pattern
            pattern_id = f"ribbed{random.randint(1000, 9999)}"
            svg_code = f'''<defs>
                <pattern id="{pattern_id}" patternUnits="userSpaceOnUse" width="10" height="10">
                    <rect width="10" height="10" fill="{fill}"/>
                    <rect x="0" y="0" width="10" height="2" fill="{fill}" fill-opacity="0.7"/>
                    <rect x="0" y="4" width="10" height="2" fill="{fill}" fill-opacity="0.7"/>
                    <rect x="0" y="8" width="10" height="2" fill="{fill}" fill-opacity="0.7"/>
                </pattern>
            </defs>
            {svg_code}'''
            fill = f"url(#{pattern_id})"
        
        # Complete the SVG element
        svg_code += f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{rotation_transform} />'
        
        return svg_code