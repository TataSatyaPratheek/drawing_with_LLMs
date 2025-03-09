"""
Material Model Module
====================
This module defines the Material class for representing material information.
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class Material:
    """Class representing material information."""
    name: str
    texture: Optional[str] = None
    transparency: float = 0.0  # 0.0 = opaque, 1.0 = fully transparent
    
    def get_svg_attributes(self) -> Dict[str, str]:
        """Return SVG attributes based on material properties."""
        attributes = {}
        
        # Handle transparency
        if self.transparency > 0:
            attributes["opacity"] = str(1 - self.transparency)
            
        # Handle patterns based on material texture
        texture_patterns = {
            "silk": "url(#patternSilk)",
            "wool": "url(#patternWool)",
            "corduroy": "url(#patternCorduroy)",
            "fur": "url(#patternFur)",
        }
        
        if self.texture in texture_patterns:
            attributes["fill"] = texture_patterns[self.texture]
        
        return attributes