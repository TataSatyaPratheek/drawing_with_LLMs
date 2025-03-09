"""
Enhanced Pattern Factory Module
===================
This module provides a factory for creating SVG patterns and textures with enhanced capabilities.
"""

import logging
import random
import re
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class PatternFactory:
    """Factory class for creating SVG patterns and textures."""
    
    @staticmethod
    def create_pattern(pattern_type: str, color: Optional[str] = None) -> Dict[str, str]:
        """
        Create an SVG pattern definition.
        
        Args:
            pattern_type: Type of pattern to create (e.g., "silk", "wool")
            color: Optional base color for the pattern
            
        Returns:
            Dictionary mapping pattern IDs to pattern code
        """
        if not color:
            color = "#FFFFFF"  # Default to white
            
        patterns = {}
        pattern_id = f"pattern{pattern_type.capitalize()}"
        
        # Standard patterns
        if pattern_type == "silk":
            patterns[pattern_id] = PatternFactory._create_silk_pattern(color)
        elif pattern_type == "wool":
            patterns[pattern_id] = PatternFactory._create_wool_pattern(color)
        elif pattern_type == "corduroy":
            patterns[pattern_id] = PatternFactory._create_corduroy_pattern(color)
        elif pattern_type == "fur":
            patterns[pattern_id] = PatternFactory._create_fur_pattern(color)
        elif pattern_type == "checkered":
            patterns[pattern_id] = PatternFactory._create_checkered_pattern(color)
        elif pattern_type == "grid":
            patterns[pattern_id] = PatternFactory._create_grid_pattern(color)
            
        # Enhanced patterns for test dataset
        elif pattern_type == "ribbed":
            patterns[pattern_id] = PatternFactory._create_ribbed_pattern(color)
        elif pattern_type == "shimmering":
            patterns[pattern_id] = PatternFactory._create_shimmering_pattern(color)
        elif pattern_type == "harlequin":
            patterns[pattern_id] = PatternFactory._create_harlequin_pattern(color)
        elif pattern_type == "disordered":
            patterns[pattern_id] = PatternFactory._create_disordered_pattern(color)
        elif pattern_type == "desert":
            patterns[pattern_id] = PatternFactory._create_desert_pattern(color)
        elif pattern_type == "mountain":
            patterns[pattern_id] = PatternFactory._create_mountain_pattern(color)
        elif pattern_type == "fringed":
            patterns[pattern_id] = PatternFactory._create_fringed_pattern(color)
        elif pattern_type == "overcast":
            patterns[pattern_id] = PatternFactory._create_overcast_pattern(color)
        elif pattern_type == "metallic" or pattern_type == "copper":
            patterns[pattern_id] = PatternFactory._create_metallic_pattern(color)
        elif pattern_type == "satin":
            patterns[pattern_id] = PatternFactory._create_satin_pattern(color)
        elif pattern_type == "scarlet_squares":
            patterns[pattern_id] = PatternFactory._create_scarlet_squares_pattern(color)
        elif pattern_type == "cashmere":
            patterns[pattern_id] = PatternFactory._create_cashmere_pattern(color)
        elif pattern_type == "wood":
            patterns[pattern_id] = PatternFactory._create_wood_pattern(color)
        else:
            # Default to a simple pattern if type not recognized
            logger.info(f"No specific pattern for '{pattern_type}', using default pattern")
            patterns[pattern_id] = PatternFactory._create_default_pattern(color)
            
        return patterns
    
    @staticmethod
    def _create_silk_pattern(color: str) -> str:
        """Create a silk-like pattern."""
        return f'''
<pattern id="patternSilk" patternUnits="userSpaceOnUse" width="20" height="20">
    <rect width="20" height="20" fill="{color}" opacity="0.9"/>
    <path d="M0,0 L20,20 M20,0 L0,20" stroke="{PatternFactory.lighten_color(color)}" stroke-width="0.5"/>
</pattern>'''
    
    @staticmethod
    def _create_wool_pattern(color: str) -> str:
        """Create a wool-like pattern."""
        return f'''
<pattern id="patternWool" patternUnits="userSpaceOnUse" width="10" height="10">
    <rect width="10" height="10" fill="{color}"/>
    <path d="M0,0 Q2.5,2.5 5,0 Q7.5,2.5 10,0 M0,5 Q2.5,7.5 5,5 Q7.5,7.5 10,5" 
          stroke="{PatternFactory.lighten_color(color)}" stroke-width="1" fill="none"/>
</pattern>'''
    
    @staticmethod
    def _create_corduroy_pattern(color: str) -> str:
        """Create a corduroy-like pattern."""
        lighter_color = PatternFactory.lighten_color(color)
        
        return f'''
<pattern id="patternCorduroy" patternUnits="userSpaceOnUse" width="8" height="8">
    <rect width="8" height="8" fill="{color}"/>
    <rect x="0" y="0" width="8" height="1" fill="{lighter_color}"/>
    <rect x="0" y="3" width="8" height="1" fill="{lighter_color}"/>
    <rect x="0" y="6" width="8" height="1" fill="{lighter_color}"/>
</pattern>'''
    
    @staticmethod
    def _create_fur_pattern(color: str) -> str:
        """Create a fur-like pattern."""
        lighter_color = PatternFactory.lighten_color(color)
        
        return f'''
<pattern id="patternFur" patternUnits="userSpaceOnUse" width="20" height="20">
    <rect width="20" height="20" fill="{color}"/>
    <path d="M5,0 L5,8 M10,0 L10,10 M15,0 L15,7 M0,5 L8,5 M0,10 L10,10 M0,15 L7,15" 
          stroke="{lighter_color}" stroke-width="1.5" stroke-linecap="round"/>
</pattern>'''
    
    @staticmethod
    def _create_checkered_pattern(color: str) -> str:
        """Create a checkered pattern."""
        # For a checkered pattern, we need a contrasting color
        contrast_color = PatternFactory.get_contrast_color(color)
        
        return f'''
<pattern id="patternCheckered" patternUnits="userSpaceOnUse" width="20" height="20">
    <rect width="10" height="10" fill="{color}"/>
    <rect x="10" y="0" width="10" height="10" fill="{contrast_color}"/>
    <rect x="0" y="10" width="10" height="10" fill="{contrast_color}"/>
    <rect x="10" y="10" width="10" height="10" fill="{color}"/>
</pattern>'''
    
    @staticmethod
    def _create_grid_pattern(color: str) -> str:
        """Create a grid pattern."""
        line_color = PatternFactory.darken_color(color, 0.3)
        
        return f'''
<pattern id="patternGrid" patternUnits="userSpaceOnUse" width="20" height="20">
    <rect width="20" height="20" fill="{color}"/>
    <path d="M0,0 L0,20 M20,0 L20,20 M0,0 L20,0 M0,20 L20,20" 
          stroke="{line_color}" stroke-width="0.5"/>
</pattern>'''
    
    @staticmethod
    def _create_ribbed_pattern(color: str) -> str:
        """Create a ribbed pattern for clothing."""
        darker_color = PatternFactory.darken_color(color, 0.2)
        
        return f'''
<pattern id="patternRibbed" patternUnits="userSpaceOnUse" width="10" height="10">
    <rect width="10" height="10" fill="{color}"/>
    <rect x="0" y="0" width="10" height="2" fill="{darker_color}" opacity="0.8"/>
    <rect x="0" y="4" width="10" height="2" fill="{darker_color}" opacity="0.8"/>
    <rect x="0" y="8" width="10" height="2" fill="{darker_color}" opacity="0.8"/>
</pattern>'''
    
    @staticmethod
    def _create_shimmering_pattern(color: str) -> str:
        """Create a shimmering effect pattern."""
        # Create a gradient pattern to simulate shimmering
        lighter_color = PatternFactory.lighten_color(color, 0.4)
        
        return f'''
<pattern id="patternShimmering" patternUnits="userSpaceOnUse" width="40" height="40">
    <rect width="40" height="40" fill="{color}"/>
    <path d="M0,0 L40,40 M40,0 L0,40" stroke="{lighter_color}" stroke-width="0.5" opacity="0.6"/>
    <path d="M20,0 L20,40 M0,20 L40,20" stroke="{lighter_color}" stroke-width="0.3" opacity="0.4"/>
    <rect width="40" height="40" fill="url(#shimmerGradient)"/>
</pattern>
<linearGradient id="shimmerGradient" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" stop-color="{lighter_color}" stop-opacity="0.2"/>
    <stop offset="50%" stop-color="{lighter_color}" stop-opacity="0.1"/>
    <stop offset="100%" stop-color="{lighter_color}" stop-opacity="0.2"/>
</linearGradient>'''
    
    @staticmethod
    def _create_harlequin_pattern(color: str) -> str:
        """Create a harlequin pattern (diamond check pattern)."""
        # For harlequin, we need a contrasting color
        contrast_color = PatternFactory.get_contrast_color(color)
        
        # For ivory and ebony specifically
        if color.lower() == "#fffff0":  # Ivory
            contrast_color = "#3D2B1F"  # Ebony
        elif color.lower() == "#3d2b1f":  # Ebony
            contrast_color = "#FFFFF0"  # Ivory
        
        return f'''
<pattern id="patternHarlequin" patternUnits="userSpaceOnUse" width="40" height="40">
    <rect width="40" height="40" fill="{color}"/>
    <polygon points="0,20 20,0 40,20 20,40" fill="{contrast_color}"/>
    <polygon points="20,0 40,20 60,0" fill="{contrast_color}"/>
    <polygon points="20,40 40,60 0,60" fill="{contrast_color}"/>
    <polygon points="-20,20 0,0 0,40" fill="{contrast_color}"/>
    <polygon points="40,0 40,40 60,20" fill="{contrast_color}"/>
</pattern>'''
    
    @staticmethod
    def _create_disordered_pattern(color: str) -> str:
        """Create a disordered pattern for arrays."""
        random_elements = []
        
        # Generate random shapes
        for i in range(10):
            x = random.randint(0, 40)
            y = random.randint(0, 40)
            size = random.randint(3, 8)
            opacity = random.uniform(0.3, 0.9)
            
            # Randomly choose between circle, square, and rectangle
            shape_type = random.choice(["circle", "rect", "rect"])
            
            if shape_type == "circle":
                random_elements.append(f'<circle cx="{x}" cy="{y}" r="{size}" fill="{color}" opacity="{opacity}" />')
            else:
                width = size * random.uniform(1, 2) if shape_type == "rect" else size
                height = size * random.uniform(1, 2) if shape_type == "rect" else size
                random_elements.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{color}" opacity="{opacity}" />')
        
        random_elements_str = "\n    ".join(random_elements)
        
        return f'''
<pattern id="patternDisordered" patternUnits="userSpaceOnUse" width="50" height="50">
    <rect width="50" height="50" fill="transparent"/>
    {random_elements_str}
</pattern>'''
    
    @staticmethod
    def _create_desert_pattern(color: str) -> str:
        """Create a desert sand pattern."""
        # Sand colors
        sand_color = "#EDC9AF" if color == "#FFFFFF" else color
        lighter_sand = PatternFactory.lighten_color(sand_color)
        darker_sand = PatternFactory.darken_color(sand_color)
        
        return f'''
<pattern id="patternDesert" patternUnits="userSpaceOnUse" width="100" height="100">
    <rect width="100" height="100" fill="{sand_color}"/>
    <!-- Sand dune shapes -->
    <path d="M0,70 Q25,50 50,70 Q75,90 100,70 L100,100 L0,100 Z" 
          fill="{lighter_sand}" opacity="0.7"/>
    <path d="M0,80 Q40,70 70,80 Q90,85 100,80 L100,100 L0,100 Z" 
          fill="{darker_sand}" opacity="0.5"/>
    <!-- Sand texture dots -->
    <circle cx="10" cy="75" r="0.5" fill="#D2B48C" opacity="0.4"/>
    <circle cx="25" cy="82" r="0.5" fill="#D2B48C" opacity="0.4"/>
    <circle cx="40" cy="77" r="0.5" fill="#D2B48C" opacity="0.4"/>
    <circle cx="55" cy="85" r="0.5" fill="#D2B48C" opacity="0.4"/>
    <circle cx="70" cy="80" r="0.5" fill="#D2B48C" opacity="0.4"/>
    <circle cx="85" cy="78" r="0.5" fill="#D2B48C" opacity="0.4"/>
    <circle cx="95" cy="83" r="0.5" fill="#D2B48C" opacity="0.4"/>
</pattern>'''
    
    @staticmethod
    def _create_mountain_pattern(color: str) -> str:
        """Create a mountain pattern for vistas."""
        mountain_color = "#808080" if color == "#FFFFFF" else color
        lighter_mountain = PatternFactory.lighten_color(mountain_color)
        darker_mountain = PatternFactory.darken_color(mountain_color)
        snow_color = "#FFFFFF"
        
        return f'''
<pattern id="patternMountain" patternUnits="userSpaceOnUse" width="200" height="100">
    <rect width="200" height="100" fill="#87CEEB"/> <!-- Sky background -->
    
    <!-- Distant mountains (background) -->
    <path d="M0,60 L30,40 L60,55 L90,35 L120,50 L150,30 L180,45 L200,40 L200,100 L0,100 Z" 
          fill="{lighter_mountain}" opacity="0.6"/>
    
    <!-- Middle mountains -->
    <path d="M0,70 L40,40 L80,65 L120,35 L160,60 L200,45 L200,100 L0,100 Z" 
          fill="{mountain_color}" opacity="0.8"/>
    
    <!-- Snow caps on middle mountains -->
    <path d="M40,40 L45,42 L50,39 L55,43 L60,41 L65,44 L80,65" 
          fill="{snow_color}" opacity="0.9"/>
    <path d="M120,35 L125,37 L130,34 L135,38 L140,36 L145,39 L160,60" 
          fill="{snow_color}" opacity="0.9"/>
    
    <!-- Foreground mountains -->
    <path d="M-30,100 L20,50 L70,75 L120,45 L170,70 L220,50 L230,100 Z" 
          fill="{darker_mountain}" opacity="1"/>
    
    <!-- Snow caps on foreground mountains -->
    <path d="M20,50 L25,52 L30,49 L35,53 L40,51 L45,54 L50,52 L55,55 L70,75" 
          fill="{snow_color}" opacity="0.9"/>
    <path d="M120,45 L125,47 L130,44 L135,48 L140,46 L145,49 L150,47 L155,50 L170,70" 
          fill="{snow_color}" opacity="0.9"/>
</pattern>'''
    
    @staticmethod
    def _create_fringed_pattern(color: str) -> str:
        """Create a fringed pattern for edges."""
        return f'''
<pattern id="patternFringed" patternUnits="userSpaceOnUse" width="20" height="10">
    <rect width="20" height="1" fill="{color}"/>
    <!-- Fringe lines -->
    <line x1="2" y1="1" x2="2" y2="8" stroke="{color}" stroke-width="0.5"/>
    <line x1="6" y1="1" x2="6" y2="10" stroke="{color}" stroke-width="0.5"/>
    <line x1="10" y1="1" x2="10" y2="7" stroke="{color}" stroke-width="0.5"/>
    <line x1="14" y1="1" x2="14" y2="9" stroke="{color}" stroke-width="0.5"/>
    <line x1="18" y1="1" x2="18" y2="8" stroke="{color}" stroke-width="0.5"/>
</pattern>'''
    
    @staticmethod
    def _create_overcast_pattern(color: str) -> str:
        """Create an overcast sky pattern."""
        # Use a light gray if no color specified
        sky_color = "#708090" if color == "#FFFFFF" else color
        cloud_color = PatternFactory.lighten_color(sky_color)
        darker_cloud = PatternFactory.darken_color(sky_color, 0.1)
        
        return f'''
<pattern id="patternOvercast" patternUnits="userSpaceOnUse" width="100" height="60">
    <rect width="100" height="60" fill="{sky_color}"/>
    <!-- Cloud formations -->
    <path d="M0,20 Q10,10 20,15 Q30,5 40,10 Q50,0 60,5 Q70,10 80,5 Q90,15 100,10 L100,30 L0,30 Z" 
          fill="{cloud_color}" opacity="0.5"/>
    <path d="M0,40 Q15,30 30,35 Q45,25 60,30 Q75,35 90,30 L100,50 L0,50 Z" 
          fill="{darker_cloud}" opacity="0.4"/>
    <path d="M10,15 Q25,5 40,10 Q55,0 70,5 Q85,15 100,5 L100,25 L10,25 Z" 
          fill="{cloud_color}" opacity="0.3"/>
</pattern>'''
    
    @staticmethod
    def _create_metallic_pattern(color: str) -> str:
        """Create a metallic/copper pattern."""
        # Default to copper color if none specified
        metal_color = "#B87333" if color == "#FFFFFF" else color
        highlight = PatternFactory.lighten_color(metal_color, 0.3)
        shadow = PatternFactory.darken_color(metal_color, 0.2)
        
        return f'''
<pattern id="patternMetallic" patternUnits="userSpaceOnUse" width="40" height="40">
    <rect width="40" height="40" fill="{metal_color}"/>
    <!-- Metallic sheen -->
    <path d="M0,0 L40,40" stroke="{highlight}" stroke-width="1" opacity="0.7"/>
    <path d="M0,10 L30,40" stroke="{highlight}" stroke-width="1" opacity="0.5"/>
    <path d="M10,0 L40,30" stroke="{highlight}" stroke-width="1" opacity="0.5"/>
    <path d="M0,20 L20,40" stroke="{shadow}" stroke-width="1" opacity="0.5"/>
    <path d="M20,0 L40,20" stroke="{shadow}" stroke-width="1" opacity="0.5"/>
    <!-- Circular highlight -->
    <circle cx="30" cy="10" r="5" fill="none" stroke="{highlight}" stroke-width="1" opacity="0.8"/>
</pattern>'''
    
    @staticmethod
    def _create_satin_pattern(color: str) -> str:
        """Create a satin fabric pattern."""
        highlight = PatternFactory.lighten_color(color, 0.3)
        
        return f'''
<pattern id="patternSatin" patternUnits="userSpaceOnUse" width="20" height="20">
    <rect width="20" height="20" fill="{color}" opacity="0.9"/>
    <path d="M0,0 L20,20 M20,0 L0,20" stroke="{highlight}" stroke-width="1" opacity="0.7"/>
    <rect width="20" height="20" fill="{color}" opacity="0.3"/>
</pattern>'''
    
    @staticmethod
    def _create_scarlet_squares_pattern(color: str) -> str:
        """Create a pattern of scarlet squares in a disordered array."""
        # Use scarlet color by default
        square_color = "#FF2400" if color == "#FFFFFF" else color
        
        # Create squares in various positions
        squares = []
        for i in range(15):
            x = random.randint(0, 80)
            y = random.randint(0, 80)
            size = random.randint(5, 15)
            rotation = random.randint(0, 45)
            opacity = random.uniform(0.7, 1.0)
            
            # Add slightly rotated square
            squares.append(
                f'<rect x="{x}" y="{y}" width="{size}" height="{size}" ' +
                f'fill="{square_color}" opacity="{opacity}" ' +
                f'transform="rotate({rotation} {x + size/2} {y + size/2})" />'
            )
        
        squares_str = "\n    ".join(squares)
        
        return f'''
<pattern id="patternScarletSquares" patternUnits="userSpaceOnUse" width="100" height="100">
    <rect width="100" height="100" fill="#FFFFFF" opacity="0.1"/>
    {squares_str}
</pattern>'''
    
    @staticmethod
    def _create_cashmere_pattern(color: str) -> str:
        """Create a cashmere fabric pattern."""
        lighter_color = PatternFactory.lighten_color(color, 0.2)
        
        return f'''
<pattern id="patternCashmere" patternUnits="userSpaceOnUse" width="20" height="20">
    <rect width="20" height="20" fill="{color}"/>
    <path d="M0,5 Q5,0 10,5 Q15,10 20,5 M0,15 Q5,10 10,15 Q15,20 20,15" 
          stroke="{lighter_color}" stroke-width="0.5" fill="none"/>
    <path d="M5,0 Q10,5 5,10 Q0,15 5,20 M15,0 Q20,5 15,10 Q10,15 15,20" 
          stroke="{lighter_color}" stroke-width="0.5" fill="none"/>
</pattern>'''
    
    @staticmethod
    def _create_wood_pattern(color: str) -> str:
        """Create a wood grain pattern."""
        # Default to a wood color if none specified
        wood_color = "#8B4513" if color == "#FFFFFF" else color
        darker_wood = PatternFactory.darken_color(wood_color, 0.2)
        lighter_wood = PatternFactory.lighten_color(wood_color, 0.1)
        
        return f'''
<pattern id="patternWood" patternUnits="userSpaceOnUse" width="100" height="20">
    <rect width="100" height="20" fill="{wood_color}"/>
    <!-- Wood grain lines -->
    <path d="M0,2 Q25,0 50,3 Q75,6 100,2" stroke="{darker_wood}" stroke-width="0.5" fill="none"/>
    <path d="M0,5 Q30,8 60,4 Q80,2 100,5" stroke="{darker_wood}" stroke-width="0.5" fill="none"/>
    <path d="M0,9 Q20,7 40,10 Q60,12 80,9 Q90,7 100,9" stroke="{darker_wood}" stroke-width="0.5" fill="none"/>
    <path d="M0,14 Q40,16 70,13 Q90,11 100,14" stroke="{darker_wood}" stroke-width="0.5" fill="none"/>
    <path d="M0,18 Q30,17 50,19 Q70,18 100,17" stroke="{darker_wood}" stroke-width="0.5" fill="none"/>
    <!-- Wood knots -->
    <circle cx="25" cy="10" r="3" fill="{darker_wood}"/>
    <circle cx="25" cy="10" r="1" fill="{lighter_wood}"/>
    <circle cx="75" cy="5" r="2" fill="{darker_wood}"/>
    <circle cx="75" cy="5" r="0.7" fill="{lighter_wood}"/>
</pattern>'''
    
    @staticmethod
    def _create_default_pattern(color: str) -> str:
        """Create a default pattern."""
        return f'''
<pattern id="patternDefault" patternUnits="userSpaceOnUse" width="10" height="10">
    <rect width="10" height="10" fill="{color}"/>
    <circle cx="5" cy="5" r="2" fill="{PatternFactory.darken_color(color)}"/>
</pattern>'''
    
    @staticmethod
    def lighten_color(hex_color: str, factor: float = 0.2) -> str:
        """
        Utility method to lighten a hex color.
        
        Args:
            hex_color: Hex color code (e.g. "#FF0000")
            factor: Factor by which to lighten (0.0 to 1.0)
            
        Returns:
            Lightened hex color
        """
        r, g, b = PatternFactory.hex_to_rgb(hex_color)
        # Lighten by moving each channel toward 255 by the factor amount
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return PatternFactory.rgb_to_hex(r, g, b)
        
    @staticmethod
    def darken_color(hex_color: str, factor: float = 0.2) -> str:
        """
        Utility method to darken a hex color.
        
        Args:
            hex_color: Hex color code (e.g. "#FF0000")
            factor: Factor by which to darken (0.0 to 1.0)
            
        Returns:
            Darkened hex color
        """
        r, g, b = PatternFactory.hex_to_rgb(hex_color)
        # Darken by moving each channel toward 0 by the factor amount
        r = int(r * (1 - factor))
        g = int(g * (1 - factor))
        b = int(b * (1 - factor))
        return PatternFactory.rgb_to_hex(r, g, b)
    
    @staticmethod
    def get_contrast_color(hex_color: str) -> str:
        """
        Get a contrasting color for the given color.
        
        Args:
            hex_color: Hex color code
            
        Returns:
            Contrasting hex color (either black or white based on brightness)
        """
        r, g, b = PatternFactory.hex_to_rgb(hex_color)
        # Calculate perceived brightness (YIQ formula)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return "#000000" if brightness > 128 else "#FFFFFF"
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """
        Convert hex color to RGB tuple.
        
        Args:
            hex_color: Hex color code (e.g. "#FF0000" or "#F00")
            
        Returns:
            Tuple of (r, g, b) values (0-255)
        """
        hex_color = hex_color.lstrip('#')
        
        # Handle shorthand hex (e.g. #F00)
        if len(hex_color) == 3:
            hex_color = ''.join([c + c for c in hex_color])
        
        # Parse the hex values
        try:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            # If invalid hex, return black
            logger.warning(f"Invalid hex color: {hex_color}, defaulting to black")
            return (0, 0, 0)
    
    @staticmethod
    def rgb_to_hex(r: int, g: int, b: int) -> str:
        """
        Convert RGB values to hex color.
        
        Args:
            r: Red value (0-255)
            g: Green value (0-255)
            b: Blue value (0-255)
            
        Returns:
            Hex color code (e.g. "#FF0000")
        """
        # Ensure values are within range
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        return f"#{r:02X}{g:02X}{b:02X}"