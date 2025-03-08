"""
A simple example model for SVG generation.
"""
import logging
import random
from typing import Dict, List, Tuple

from svg_gen_evaluator.models.base import BaseSVGModel

logger = logging.getLogger(__name__)


class SimpleModel(BaseSVGModel):
    """
    A simple SVG generation model that creates basic shapes based on keyword matching.
    
    This model serves as a basic example and can be used as a starting point
    for more sophisticated SVG generation approaches.
    """
    
    def __init__(self):
        """Initialize the model with color mappings and shapes."""
        self.colors = self._get_color_mappings()
    
    def _get_color_mappings(self) -> Dict[str, str]:
        """Define mappings from color names to hex values."""
        return {
            "red": "#FF0000", "blue": "#0000FF", "green": "#00FF00",
            "yellow": "#FFFF00", "purple": "#800080", "orange": "#FFA500",
            "black": "#000000", "white": "#FFFFFF", "gray": "#808080",
            "pink": "#FFC0CB", "brown": "#A52A2A", "cyan": "#00FFFF",
            "magenta": "#FF00FF", "lime": "#00FF00", "olive": "#808000",
            "navy": "#000080", "teal": "#008080", "maroon": "#800000",
            "crimson": "#DC143C", "emerald": "#50C878", "azure": "#F0FFFF",
            "silver": "#C0C0C0", "gold": "#FFD700", "bronze": "#CD7F32",
            "khaki": "#F0E68C", "sky-blue": "#87CEEB", "turquoise": "#40E0D0",
            "indigo": "#4B0082", "violet": "#EE82EE", "tan": "#D2B48C",
            "burgundy": "#800020", "aubergine": "#614051", "fuchsia": "#FF00FF",
            "chestnut": "#954535", "ginger": "#B06500", "copper": "#B87333",
            "pewter": "#8A9A5B", "ebony": "#555D50", "ivory": "#FFFFF0",
            "wine": "#722F37", "snow": "#FFFAFA", "charcoal": "#36454F"
        }
    
    def predict(self, description: str) -> str:
        """
        Generate an SVG based on the description.
        
        Args:
            description: Textual description of the image to generate
            
        Returns:
            SVG code as a string
        """
        # Extract colors from description
        primary_color, secondary_color = self._extract_colors(description)
        
        # Determine the type of SVG to generate based on keywords
        desc_lower = description.lower()
        
        if any(shape in desc_lower for shape in ["square", "rectangle", "grid"]):
            return self._create_rectangle_svg(primary_color, secondary_color, description)
        elif any(shape in desc_lower for shape in ["circle", "ellipse", "round"]):
            return self._create_circle_svg(primary_color, secondary_color, description)
        elif any(shape in desc_lower for shape in ["triangle", "polygon"]):
            return self._create_polygon_svg(primary_color, secondary_color, description)
        elif any(item in desc_lower for item in ["mountain", "peak", "hill"]):
            return self._create_landscape_svg(primary_color, secondary_color, description)
        elif any(item in desc_lower for item in ["ocean", "sea", "lake", "water", "lagoon"]):
            return self._create_water_scene_svg(primary_color, secondary_color, description)
        elif any(item in desc_lower for item in ["forest", "tree", "wood"]):
            return self._create_forest_svg(primary_color, secondary_color, description)
        elif any(item in desc_lower for item in ["pants", "trousers", "coat", "clothing", "scarf", "overalls"]):
            return self._create_clothing_svg(primary_color, secondary_color, description)
        else:
            return self._create_abstract_svg(primary_color, secondary_color, description)
    
    def _extract_colors(self, description: str) -> Tuple[str, str]:
        """
        Extract color information from the description.
        
        Args:
            description: Textual description
            
        Returns:
            Tuple of (primary_color, secondary_color) as hex values
        """
        primary_color = "#3498db"  # Default blue
        secondary_color = "#e74c3c"  # Default red
        
        # Extract colors from description
        desc_lower = description.lower()
        found_colors = []
        
        for color_name, hex_code in self.colors.items():
            if color_name in desc_lower:
                found_colors.append(hex_code)
                
        if found_colors:
            primary_color = found_colors[0]
            if len(found_colors) > 1:
                secondary_color = found_colors[1]
        
        return primary_color, secondary_color
    
    def _create_rectangle_svg(self, color1: str, color2: str, description: str) -> str:
        """Create SVG with rectangles."""
        desc_lower = description.lower()
        is_chaotic = "chaotic" in desc_lower or "disordered" in desc_lower
        is_checkered = "check" in desc_lower or "harlequin" in desc_lower
        
        if is_checkered:
            # Create a checkered pattern
            rects = ""
            for i in range(8):
                for j in range(8):
                    fill = color1 if (i + j) % 2 == 0 else color2
                    rects += f'<rect x="{i*50}" y="{j*50}" width="50" height="50" fill="{fill}" />\n'
            
            return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
                {rects}
            </svg>"""
        elif is_chaotic:
            # Create chaotic rectangles
            rects = ""
            for _ in range(20):
                x = random.randint(0, 350)
                y = random.randint(0, 350)
                width = random.randint(20, 100)
                height = random.randint(20, 100)
                rotation = random.randint(0, 360)
                opacity = random.uniform(0.3, 1.0)
                fill = color1 if random.random() > 0.5 else color2
                
                rects += f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
                rects += f'fill="{fill}" opacity="{opacity:.1f}" '
                rects += f'transform="rotate({rotation} {x+width/2} {y+height/2})" />\n'
            
            return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
                {rects}
            </svg>"""
        else:
            # Simple rectangles
            return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
                <rect x="50" y="50" width="300" height="300" fill="{color1}" />
                <rect x="100" y="100" width="200" height="200" fill="{color2}" />
            </svg>"""
    
    def _create_circle_svg(self, color1: str, color2: str, description: str) -> str:
        """Create SVG with circles."""
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
            <circle cx="200" cy="200" r="150" fill="{color1}" />
            <circle cx="200" cy="200" r="100" fill="{color2}" />
        </svg>"""
    
    def _create_polygon_svg(self, color1: str, color2: str, description: str) -> str:
        """Create SVG with polygons."""
        if "triangle" in description.lower():
            return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
                <polygon points="200,50 50,350 350,350" fill="{color1}" />
                <polygon points="200,100 100,300 300,300" fill="{color2}" />
            </svg>"""
        else:
            # Generic polygon or other shapes
            return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
                <polygon points="200,50 250,150 200,250 150,150" fill="{color1}" />
                <polygon points="300,200 350,300 250,300" fill="{color2}" />
                <polygon points="100,200 150,300 50,300" fill="{color2}" />
            </svg>"""
    
    def _create_landscape_svg(self, color1: str, color2: str, description: str) -> str:
        """Create a landscape with mountains."""
        desc_lower = description.lower()
        sky_color = "#87CEEB"  # Sky blue
        mountain_color = color1
        snow_color = "#FFFFFF"  # White for snow
        
        if "snow" in desc_lower or "snowy" in desc_lower:
            ground_color = "#FFFFFF"  # Snow
        else:
            ground_color = "#8B4513"  # Brown ground
            
        if "night" in desc_lower or "dusk" in desc_lower:
            sky_color = "#1A237E"  # Dark blue
            # Add stars if it's night
            stars = ""
            for _ in range(50):
                x = random.randint(0, 400)
                y = random.randint(0, 150)
                r = random.uniform(0.5, 2)
                stars += f'<circle cx="{x}" cy="{y}" r="{r}" fill="#FFFFFF" />\n'
        else:
            stars = ""
        
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
            <rect width="400" height="300" fill="{sky_color}" />
            {stars}
            <polygon points="0,150 100,50 200,120 300,30 400,130 400,300 0,300" fill="{mountain_color}" />
            <polygon points="100,50 130,65 150,40 170,60 200,120" fill="{snow_color}" />
            <polygon points="300,30 320,50 340,45 360,65 400,130" fill="{snow_color}" />
            <rect x="0" y="200" width="400" height="100" fill="{ground_color}" />
        </svg>"""
    
    def _create_water_scene_svg(self, color1: str, color2: str, description: str) -> str:
        """Create a scene with water."""
        desc_lower = description.lower()
        sky_color = "#87CEEB"  # Sky blue
        water_color = "#1E88E5"  # Water blue
        
        if "lagoon" in desc_lower and "green" in desc_lower:
            water_color = "#4CAF50"  # Green for lagoon
        
        if "cloudy" in desc_lower:
            clouds = """
                <circle cx="100" cy="80" r="30" fill="#FFFFFF" opacity="0.8" />
                <circle cx="130" cy="80" r="30" fill="#FFFFFF" opacity="0.8" />
                <circle cx="160" cy="80" r="30" fill="#FFFFFF" opacity="0.8" />
                <circle cx="250" cy="60" r="25" fill="#FFFFFF" opacity="0.8" />
                <circle cx="280" cy="60" r="25" fill="#FFFFFF" opacity="0.8" />
                <circle cx="310" cy="60" r="25" fill="#FFFFFF" opacity="0.8" />
            """
        else:
            clouds = ""
            
        if "lighthouse" in desc_lower:
            # Add a lighthouse
            lighthouse = """
                <rect x="180" y="120" width="40" height="120" fill="#EEEEEE" />
                <polygon points="170,120 230,120 200,80" fill="#FF5252" />
                <rect x="190" y="100" width="20" height="20" fill="#FFD600" />
                <rect x="170" y="240" width="60" height="20" fill="#EEEEEE" />
            """
        else:
            lighthouse = ""
        
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
            <rect width="400" height="150" fill="{sky_color}" />
            {clouds}
            <rect x="0" y="150" width="400" height="150" fill="{water_color}" />
            {lighthouse}
            
            <!-- Water waves -->
            <path d="M0,180 Q25,170 50,180 Q75,190 100,180 Q125,170 150,180 Q175,190 200,180 Q225,170 250,180 Q275,190 300,180 Q325,170 350,180 Q375,190 400,180 L400,300 L0,300 Z" fill="{water_color}" opacity="0.7" />
        </svg>"""
    
    def _create_forest_svg(self, color1: str, color2: str, description: str) -> str:
        """Create a forest scene."""
        desc_lower = description.lower()
        
        if "purple" in desc_lower or "violet" in desc_lower:
            sky_color = "#5E35B1"  # Purple sky
            ground_color = "#4A148C"  # Dark purple ground
            tree_color = "#6A1B9A"  # Tree color
        else:
            sky_color = "#2196F3"  # Blue sky
            ground_color = "#795548"  # Brown ground
            tree_color = "#2E7D32"  # Green trees
        
        if "dusk" in desc_lower:
            sky_color = "#5D4037"  # Brownish dusk sky
        
        trees = ""
        for i in range(10):
            x = i * 40 + 20
            y = 200 + random.randint(-20, 20)
            height = random.randint(80, 120)
            width = height / 3
            
            # Tree trunk
            trees += f'<rect x="{x-width/6}" y="{y-height}" width="{width/3}" height="{height}" fill="#5D4037" />\n'
            
            # Tree foliage (triangles for conifers)
            for j in range(3):
                tri_width = width * (3-j) / 3
                tri_height = height / 2
                tri_y = y - height + j * height / 6
                trees += f'<polygon points="{x-tri_width/2},{tri_y} {x+tri_width/2},{tri_y} {x},{tri_y-tri_height}" fill="{tree_color}" />\n'
        
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
            <rect width="400" height="300" fill="{sky_color}" />
            <rect x="0" y="200" width="400" height="100" fill="{ground_color}" />
            {trees}
        </svg>"""
    
    def _create_clothing_svg(self, color1: str, color2: str, description: str) -> str:
        """Create clothing items."""
        desc_lower = description.lower()
        
        if "pants" in desc_lower or "trousers" in desc_lower:
            if "cargo" in desc_lower:
                # Cargo pants with pockets
                return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 500">
                    <!-- Pants body -->
                    <path d="M100,50 L300,50 L320,450 L220,450 L200,300 L180,450 L80,450 Z" fill="{color1}" />
                    
                    <!-- Left cargo pocket -->
                    <rect x="110" y="200" width="60" height="70" fill="{color2}" opacity="0.9" />
                    <rect x="120" y="210" width="40" height="50" fill="{color1}" opacity="0.9" />
                    
                    <!-- Right cargo pocket -->
                    <rect x="230" y="200" width="60" height="70" fill="{color2}" opacity="0.9" />
                    <rect x="240" y="210" width="40" height="50" fill="{color1}" opacity="0.9" />
                    
                    <!-- Buttons -->
                    <circle cx="140" cy="190" r="5" fill="{color2}" />
                    <circle cx="260" cy="190" r="5" fill="{color2}" />
                </svg>"""
            elif "checkered" in desc_lower:
                # Checkered pants
                checkers = ""
                for i in range(10):
                    for j in range(20):
                        if (i + j) % 2 == 0:
                            x = 120 + i * 16
                            y = 70 + j * 20
                            checkers += f'<rect x="{x}" y="{y}" width="16" height="20" fill="{color2}" />\n'
                
                return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 500">
                    <!-- Pants outline -->
                    <path d="M100,50 L300,50 L320,450 L220,450 L200,300 L180,450 L80,450 Z" fill="{color1}" />
                    
                    <!-- Checkered pattern -->
                    {checkers}
                </svg>"""
            elif "corduroy" in desc_lower:
                # Corduroy pants with lines
                lines = ""
                for i in range(30):
                    y = 70 + i * 13
                    lines += f'<line x1="100" y1="{y}" x2="300" y2="{y}" stroke="{color2}" stroke-width="2" opacity="0.7" />\n'
                
                return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 500">
                    <!-- Pants body -->
                    <path d="M100,50 L300,50 L320,450 L220,450 L200,300 L180,450 L80,450 Z" fill="{color1}" />
                    
                    <!-- Corduroy lines -->
                    {lines}
                </svg>"""
            else:
                # Basic pants
                return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 500">
                    <path d="M100,50 L300,50 L320,450 L220,450 L200,300 L180,450 L80,450 Z" fill="{color1}" />
                    <path d="M100,50 L300,50 L280,100 L120,100 Z" fill="{color2}" />
                </svg>"""
                
        elif "overalls" in desc_lower:
            # Overalls
            return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 500">
                <!-- Pants part -->
                <path d="M100,150 L300,150 L320,450 L220,450 L200,300 L180,450 L80,450 Z" fill="{color1}" />
                
                <!-- Bib part -->
                <rect x="150" y="50" width="100" height="100" fill="{color1}" />
                
                <!-- Straps -->
                <path d="M150,50 L120,20 L100,150" fill="none" stroke="{color1}" stroke-width="10" />
                <path d="M250,50 L280,20 L300,150" fill="none" stroke="{color1}" stroke-width="10" />
                
                <!-- Pocket on bib -->
                <rect x="170" y="80" width="60" height="40" fill="{color2}" />
            </svg>"""
            
        elif "coat" in desc_lower:
            # Coat
            return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 500">
                <!-- Main coat -->
                <path d="M100,50 L300,50 L320,450 L200,400 L80,450 Z" fill="{color1}" />
                
                <!-- Collar -->
                <path d="M100,50 L150,100 L200,70 L250,100 L300,50 L250,170 L150,170 Z" fill="{color2}" />
                
                <!-- Left arm -->
                <path d="M100,50 L50,200 L80,210 L130,70" fill="{color1}" />
                
                <!-- Right arm -->
                <path d="M300,50 L350,200 L320,210 L270,70" fill="{color1}" />
                
                <!-- Buttons -->
                <circle cx="200" cy="200" r="5" fill="{color2}" />
                <circle cx="200" cy="230" r="5" fill="{color2}" />
                <circle cx="200" cy="260" r="5" fill="{color2}" />
            </svg>"""
            
        elif "scarf" in desc_lower:
            # Scarf
            return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
                <!-- Scarf body -->
                <path d="M50,50 C100,30 150,80 200,50 C250,20 300,70 350,50 L320,150 C270,170 220,120 170,150 C120,180 70,130 80,150 Z" fill="{color1}" />
                
                <!-- Tassels -->
                <line x1="80" y1="150" x2="70" y2="190" stroke="{color2}" stroke-width="3" />
                <line x1="90" y1="155" x2="80" y2="195" stroke="{color2}" stroke-width="3" />
                <line x1="100" y1="160" x2="90" y2="200" stroke="{color2}" stroke-width="3" />
                <line x1="300" y1="150" x2="290" y2="190" stroke="{color2}" stroke-width="3" />
                <line x1="310" y1="145" x2="300" y2="185" stroke="{color2}" stroke-width="3" />
                <line x1="320" y1="140" x2="310" y2="180" stroke="{color2}" stroke-width="3" />
            </svg>"""
            
        else:
            # Generic clothing outline
            return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
                <path d="M100,50 L300,50 L350,300 L250,400 L150,400 L50,300 Z" fill="{color1}" />
                <path d="M150,50 L250,50 L230,150 L170,150 Z" fill="{color2}" />
            </svg>"""
    
    def _create_abstract_svg(self, color1: str, color2: str, description: str) -> str:
        """Create an abstract SVG when no specific pattern is identified."""
        # Create a variety of random shapes
        shapes = []
        
        # Add some circles
        for _ in range(random.randint(3, 8)):
            cx = random.randint(50, 350)
            cy = random.randint(50, 350)
            r = random.randint(20, 80)
            opacity = random.uniform(0.3, 0.9)
            fill = color1 if random.random() > 0.5 else color2
            
            shapes.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" opacity="{opacity:.2f}" />')
        
        # Add some rectangles
        for _ in range(random.randint(3, 8)):
            x = random.randint(20, 300)
            y = random.randint(20, 300)
            width = random.randint(30, 150)
            height = random.randint(30, 150)
            opacity = random.uniform(0.3, 0.9)
            fill = color1 if random.random() > 0.5 else color2
            rotation = random.randint(0, 360)
            
            shapes.append(
                f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
                f'fill="{fill}" opacity="{opacity:.2f}" '
                f'transform="rotate({rotation} {x+width/2} {y+height/2})" />'
            )
        
        # Add some lines
        for _ in range(random.randint(5, 15)):
            x1 = random.randint(0, 400)
            y1 = random.randint(0, 400)
            x2 = random.randint(0, 400)
            y2 = random.randint(0, 400)
            stroke_width = random.randint(1, 8)
            stroke = color1 if random.random() > 0.5 else color2
            opacity = random.uniform(0.4, 1.0)
            
            shapes.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity:.2f}" />'
            )
        
        # Shuffle the shapes for a more random appearance
        random.shuffle(shapes)
        
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
            <rect width="400" height="400" fill="#F5F5F5" />
            {"".join(shapes)}
        </svg>"""