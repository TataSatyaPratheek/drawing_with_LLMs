"""
Enhanced Scene Model Module
================
This module defines the enhanced Scene class for representing a complete visual scene.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from svg_prompt_analyzer.models.visual_object import VisualObject


@dataclass
class Scene:
    """Enhanced class representing the entire scene described in the prompt."""
    id: str
    prompt: str
    background_color: str = "#FFFFFF"  # Default white
    objects: List[VisualObject] = field(default_factory=list)
    width: int = 800
    height: int = 600
    patterns: Dict[str, str] = field(default_factory=dict)
    defs: List[str] = field(default_factory=list)  # Added for additional SVG definitions
    special_elements: List[str] = field(default_factory=list)  # For special rendering elements
    
    def get_svg_code(self) -> str:
        """Generate complete SVG code for the scene with enhanced features."""
        # SVG header
        svg_code = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}"
    xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>SVG illustration of {self.prompt}</title>
    <desc>Generated from prompt: {self.prompt}</desc>
    
    <!-- Definitions for patterns, gradients, etc. -->
    <defs>
'''
        
        # Add patterns
        for pattern_id, pattern_code in self.patterns.items():
            svg_code += f"        {pattern_code}\n"
        
        # Add custom defs
        for def_code in self.defs:
            svg_code += f"        {def_code}\n"
        
        # Close defs section
        svg_code += '''    </defs>
    
    <!-- Background -->
'''
        
        # Add background
        svg_code += f'    <rect width="100%" height="100%" fill="{self.background_color}" />\n'
        
        # Sort objects by z-index
        sorted_objects = sorted(self.objects, key=lambda obj: obj.z_index)
        
        # Add objects
        for obj in sorted_objects:
            svg_code += f'\n    <!-- {obj.name} -->\n'
            for element in obj.get_svg_elements(self.width, self.height):
                svg_code += f'    {element}\n'
        
        # Add special elements
        if hasattr(self, 'special_elements') and self.special_elements:
            svg_code += '\n    <!-- Special Elements -->\n'
            for element in self.special_elements:
                svg_code += f'    {element}\n'
        
        # SVG footer
        svg_code += '</svg>'
        
        return svg_code