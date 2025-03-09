"""
Production-optimized color model for SVG generation and analysis.
Provides efficient color representation, manipulation, and conversion.
"""

import re
import math
import colorsys
from typing import Dict, List, Tuple, Union, Optional, Set, Any, NamedTuple
from enum import Enum, auto
import functools

# Import core optimizations
from core import CONFIG, memoize, jit, Profiler
from utils.logger import get_logger

# Configure logger
logger = get_logger(__name__)

# Type definitions
RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, float]
HSL = Tuple[float, float, float]
HSLA = Tuple[float, float, float, float]
ColorValue = Union[str, RGB, RGBA, HSL, HSLA]

# Constants
DEFAULT_ALPHA = 1.0
COLOR_PRECISION = 4  # Decimal places for color calculations

# Common color name mapping (most used web colors)
_COMMON_COLORS: Dict[str, RGB] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "silver": (192, 192, 192),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "purple": (128, 0, 128),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    # Add more as needed, but keep smaller for performance
}

# Memory-efficient color format enum
class ColorFormat(Enum):
    """Enum for different color formats."""
    HEX = auto()
    RGB = auto()
    RGBA = auto()
    HSL = auto()
    HSLA = auto()
    NAMED = auto()


class ColorError(Exception):
    """Custom exception for color-related errors."""
    pass


class Color:
    """
    Memory-efficient, immutable color representation with caching.
    
    This class provides methods for color manipulation, conversion,
    and analysis optimized for production SVG generation.
    """
    
    __slots__ = ('_r', '_g', '_b', '_a', '_format', '_original', '_hash')
    
    # Static cache for commonly used colors (improves memory usage)
    _cache: Dict[str, 'Color'] = {}
    
    def __init__(
        self,
        value: ColorValue,
        alpha: float = DEFAULT_ALPHA,
        format_hint: Optional[ColorFormat] = None
    ):
        """
        Initialize a color from various formats.
        
        Args:
            value: Color value (hex string, RGB tuple, HSL tuple, color name)
            alpha: Alpha value (0.0-1.0) for transparency
            format_hint: Optional hint for input format
        
        Raises:
            ColorError: If the color format is invalid or can't be parsed
        """
        # Store original value for potential serialization
        self._original = value
        self._format = format_hint
        
        # Parse and normalize color value
        try:
            # Handle string input
            if isinstance(value, str):
                r, g, b, a = self._parse_color_string(value)
                
            # Handle RGB/RGBA tuple
            elif isinstance(value, tuple) and len(value) in (3, 4):
                if len(value) == 3:
                    r, g, b = value
                    a = alpha
                else:
                    r, g, b, a = value
                    
                # Validate RGB values
                if not all(isinstance(c, (int, float)) for c in (r, g, b)):
                    raise ColorError(f"RGB values must be numbers, got {value}")
                    
                # Normalize RGB to 0-255 range
                r = min(255, max(0, int(r)))
                g = min(255, max(0, int(g)))
                b = min(255, max(0, int(b)))
                
            # Handle HSL/HSLA tuple with format hint
            elif isinstance(value, tuple) and len(value) in (3, 4) and format_hint in (ColorFormat.HSL, ColorFormat.HSLA):
                if len(value) == 3:
                    h, s, l = value
                    a = alpha
                else:
                    h, s, l, a = value
                    
                # Validate HSL values
                if not all(isinstance(c, (int, float)) for c in (h, s, l)):
                    raise ColorError(f"HSL values must be numbers, got {value}")
                    
                # Normalize HSL
                h = h % 360
                s = min(1.0, max(0.0, s))
                l = min(1.0, max(0.0, l))
                
                # Convert HSL to RGB
                r, g, b = self._hsl_to_rgb(h, s, l)
                
            else:
                raise ColorError(f"Unsupported color format: {value}")
                
            # Normalize alpha
            a = min(1.0, max(0.0, float(a)))
            
            # Round values for consistency
            r = round(r)
            g = round(g)
            b = round(b)
            a = round(a, COLOR_PRECISION)
            
            # Store components
            self._r = r
            self._g = g
            self._b = b
            self._a = a
            
            # Calculate hash once
            self._hash = hash((self._r, self._g, self._b, self._a))
            
        except Exception as e:
            if isinstance(e, ColorError):
                raise
            raise ColorError(f"Failed to parse color value {value}: {str(e)}")
    
    @classmethod
    def from_hex(cls, hex_string: str, alpha: float = DEFAULT_ALPHA) -> 'Color':
        """
        Create a color from a hex string.
        
        Args:
            hex_string: Hex color string (e.g., "#FF0000" or "FF0000")
            alpha: Alpha value (0.0-1.0)
            
        Returns:
            Color instance
        """
        # Check cache first
        cache_key = f"hex:{hex_string}:{alpha}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]
            
        # Create new color
        color = cls(hex_string, alpha, ColorFormat.HEX)
        
        # Cache common colors
        if len(cls._cache) < 1000:  # Limit cache size
            cls._cache[cache_key] = color
            
        return color
    
    @classmethod
    def from_rgb(cls, r: int, g: int, b: int, a: float = DEFAULT_ALPHA) -> 'Color':
        """
        Create a color from RGB values.
        
        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            a: Alpha value (0.0-1.0)
            
        Returns:
            Color instance
        """
        # Check cache first for common colors with full alpha
        if a == 1.0:
            cache_key = f"rgb:{r},{g},{b}"
            if cache_key in cls._cache:
                return cls._cache[cache_key]
                
        # Create new color
        color = cls((r, g, b, a), format_hint=ColorFormat.RGBA)
        
        # Cache common colors with full alpha
        if a == 1.0 and len(cls._cache) < 1000:
            cls._cache[cache_key] = color
            
        return color
    
    @classmethod
    def from_hsl(cls, h: float, s: float, l: float, a: float = DEFAULT_ALPHA) -> 'Color':
        """
        Create a color from HSL values.
        
        Args:
            h: Hue (0-360)
            s: Saturation (0.0-1.0)
            l: Lightness (0.0-1.0)
            a: Alpha value (0.0-1.0)
            
        Returns:
            Color instance
        """
        return cls((h, s, l, a), format_hint=ColorFormat.HSLA)
    
    @classmethod
    def from_name(cls, name: str, alpha: float = DEFAULT_ALPHA) -> 'Color':
        """
        Create a color from a predefined color name.
        
        Args:
            name: Color name (e.g., "red", "blue")
            alpha: Alpha value (0.0-1.0)
            
        Returns:
            Color instance
            
        Raises:
            ColorError: If the color name is not recognized
        """
        name = name.lower().strip()
        
        # Check cache first
        cache_key = f"name:{name}:{alpha}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]
            
        # Check if name exists in common colors
        if name in _COMMON_COLORS:
            rgb = _COMMON_COLORS[name]
            color = cls(rgb, alpha, ColorFormat.NAMED)
            
            # Cache the result
            if len(cls._cache) < 1000:
                cls._cache[cache_key] = color
                
            return color
            
        raise ColorError(f"Unknown color name: {name}")
    
    @classmethod
    @memoize
    def random(cls, saturation: float = 0.7, lightness: float = 0.5, alpha: float = 1.0) -> 'Color':
        """
        Generate a random color with controlled saturation and lightness.
        
        Args:
            saturation: Saturation value (0.0-1.0)
            lightness: Lightness value (0.0-1.0)
            alpha: Alpha value (0.0-1.0)
            
        Returns:
            Random color instance
        """
        import random
        h = random.uniform(0, 360)
        s = min(1.0, max(0.0, saturation))
        l = min(1.0, max(0.0, lightness))
        return cls.from_hsl(h, s, l, alpha)
    
    @property
    def red(self) -> int:
        """Get the red component (0-255)."""
        return self._r
        
    @property
    def green(self) -> int:
        """Get the green component (0-255)."""
        return self._g
        
    @property
    def blue(self) -> int:
        """Get the blue component (0-255)."""
        return self._b
        
    @property
    def alpha(self) -> float:
        """Get the alpha component (0.0-1.0)."""
        return self._a
        
    @property
    def rgb(self) -> RGB:
        """Get RGB tuple."""
        return (self._r, self._g, self._b)
        
    @property
    def rgba(self) -> RGBA:
        """Get RGBA tuple."""
        return (self._r, self._g, self._b, self._a)
    
    @property
    @memoize
    def hsl(self) -> HSL:
        """Get HSL tuple."""
        return self._rgb_to_hsl(self._r, self._g, self._b)
        
    @property
    def hsla(self) -> HSLA:
        """Get HSLA tuple."""
        h, s, l = self.hsl
        return (h, s, l, self._a)
    
    @property
    @memoize
    def hex(self) -> str:
        """Get hex color string without alpha."""
        return f"#{self._r:02x}{self._g:02x}{self._b:02x}"
    
    @property
    @memoize
    def hex_with_alpha(self) -> str:
        """Get hex color string with alpha."""
        alpha_hex = int(self._a * 255)
        return f"#{self._r:02x}{self._g:02x}{self._b:02x}{alpha_hex:02x}"
    
    @property
    @memoize
    def is_transparent(self) -> bool:
        """Check if color is fully transparent."""
        return self._a < 0.01
    
    @property
    @memoize
    def is_opaque(self) -> bool:
        """Check if color is fully opaque."""
        return self._a > 0.99
    
    @property
    @memoize
    def is_grayscale(self) -> bool:
        """Check if color is a shade of gray."""
        return abs(self._r - self._g) < 2 and abs(self._g - self._b) < 2
    
    @property
    @memoize
    def luminance(self) -> float:
        """
        Calculate relative luminance according to WCAG 2.0.
        
        Returns:
            Luminance value between 0 (black) and 1 (white)
        """
        # Convert sRGB to linear RGB
        r = self._r / 255
        g = self._g / 255
        b = self._b / 255
        
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        
        # Calculate luminance
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    def with_alpha(self, alpha: float) -> 'Color':
        """
        Create a new color with the specified alpha value.
        
        Args:
            alpha: New alpha value (0.0-1.0)
            
        Returns:
            New color instance with updated alpha
        """
        if abs(alpha - self._a) < 0.001:
            return self
        return Color((self._r, self._g, self._b), alpha)
    
    def lighten(self, amount: float = 0.1) -> 'Color':
        """
        Create a lighter version of this color.
        
        Args:
            amount: Amount to lighten by (0.0-1.0)
            
        Returns:
            Lightened color
        """
        h, s, l = self.hsl
        return Color.from_hsl(h, s, min(1.0, l + amount), self._a)
    
    def darken(self, amount: float = 0.1) -> 'Color':
        """
        Create a darker version of this color.
        
        Args:
            amount: Amount to darken by (0.0-1.0)
            
        Returns:
            Darkened color
        """
        h, s, l = self.hsl
        return Color.from_hsl(h, s, max(0.0, l - amount), self._a)
    
    def saturate(self, amount: float = 0.1) -> 'Color':
        """
        Create a more saturated version of this color.
        
        Args:
            amount: Amount to increase saturation by (0.0-1.0)
            
        Returns:
            Saturated color
        """
        h, s, l = self.hsl
        return Color.from_hsl(h, min(1.0, s + amount), l, self._a)
    
    def desaturate(self, amount: float = 0.1) -> 'Color':
        """
        Create a less saturated version of this color.
        
        Args:
            amount: Amount to decrease saturation by (0.0-1.0)
            
        Returns:
            Desaturated color
        """
        h, s, l = self.hsl
        return Color.from_hsl(h, max(0.0, s - amount), l, self._a)
    
    def grayscale(self) -> 'Color':
        """
        Convert color to grayscale.
        
        Returns:
            Grayscale version of color
        """
        # Use luminance-preserving method
        lum = int(0.299 * self._r + 0.587 * self._g + 0.114 * self._b)
        return Color.from_rgb(lum, lum, lum, self._a)
    
    def invert(self) -> 'Color':
        """
        Invert the color.
        
        Returns:
            Inverted color
        """
        return Color.from_rgb(255 - self._r, 255 - self._g, 255 - self._b, self._a)
    
    def complement(self) -> 'Color':
        """
        Get the complementary color.
        
        Returns:
            Complementary color
        """
        h, s, l = self.hsl
        return Color.from_hsl((h + 180) % 360, s, l, self._a)
    
    def contrast_color(self) -> 'Color':
        """
        Get a contrasting color (black or white) for text.
        
        Returns:
            Black or white color depending on luminance
        """
        # Use luminance to determine contrast color
        return Color.from_rgb(0, 0, 0) if self.luminance > 0.5 else Color.from_rgb(255, 255, 255)
    
    def blend(self, other: 'Color', ratio: float = 0.5) -> 'Color':
        """
        Blend with another color.
        
        Args:
            other: Color to blend with
            ratio: Blend ratio (0.0 = this color, 1.0 = other color)
            
        Returns:
            Blended color
        """
        ratio = min(1.0, max(0.0, ratio))
        r = int(self._r * (1 - ratio) + other._r * ratio)
        g = int(self._g * (1 - ratio) + other._g * ratio)
        b = int(self._b * (1 - ratio) + other._b * ratio)
        a = self._a * (1 - ratio) + other._a * ratio
        return Color.from_rgb(r, g, b, a)
    
    def contrast_ratio(self, other: 'Color') -> float:
        """
        Calculate contrast ratio between two colors according to WCAG 2.0.
        
        Args:
            other: Color to compare with
            
        Returns:
            Contrast ratio (1-21)
        """
        l1 = self.luminance
        l2 = other.luminance
        
        # Ensure the lighter color is always the first one
        if l1 < l2:
            l1, l2 = l2, l1
            
        # Calculate contrast ratio
        return (l1 + 0.05) / (l2 + 0.05)
    
    def distance(self, other: 'Color') -> float:
        """
        Calculate perceptual color distance (Euclidean distance in RGB space).
        
        Args:
            other: Color to compare with
            
        Returns:
            Distance value (0-441.67)
        """
        return math.sqrt(
            (self._r - other._r) ** 2 +
            (self._g - other._g) ** 2 +
            (self._b - other._b) ** 2
        )
    
    def to_svg_string(self) -> str:
        """
        Generate SVG-compatible color string.
        
        Returns:
            Color string suitable for SVG attributes
        """
        if self._a < 0.999:
            return f"rgba({self._r}, {self._g}, {self._b}, {self._a:.4f})"
        return self.hex
    
    def to_css_string(self) -> str:
        """
        Generate CSS-compatible color string.
        
        Returns:
            Color string suitable for CSS
        """
        if self._a < 0.999:
            return f"rgba({self._r}, {self._g}, {self._b}, {self._a:.4f})"
        return self.hex
    
    def __eq__(self, other: object) -> bool:
        """Compare colors for equality."""
        if not isinstance(other, Color):
            return False
        return (self._r, self._g, self._b, self._a) == (other._r, other._g, other._b, other._a)
    
    def __hash__(self) -> int:
        """Get hash of color for use in dictionaries and sets."""
        return self._hash
    
    def __str__(self) -> str:
        """Get string representation of color."""
        if self._a < 0.999:
            return f"rgba({self._r}, {self._g}, {self._b}, {self._a:.4f})"
        return self.hex
    
    def __repr__(self) -> str:
        """Get detailed string representation of color."""
        return f"Color(rgb=({self._r}, {self._g}, {self._b}), alpha={self._a:.4f})"
    
    @staticmethod
    def _parse_color_string(value: str) -> Tuple[int, int, int, float]:
        """
        Parse color string into RGBA components.
        
        Args:
            value: Color string (hex, rgb(), rgba(), etc.)
            
        Returns:
            Tuple of (r, g, b, a) components
            
        Raises:
            ColorError: If color string can't be parsed
        """
        value = value.strip().lower()
        
        # Named color
        if value in _COMMON_COLORS:
            r, g, b = _COMMON_COLORS[value]
            return r, g, b, DEFAULT_ALPHA
            
        # Hex color
        if value.startswith('#'):
            return Color._parse_hex(value)
            
        # RGB/RGBA color
        if value.startswith('rgb'):
            return Color._parse_rgb(value)
            
        # HSL/HSLA color
        if value.startswith('hsl'):
            return Color._parse_hsl(value)
            
        raise ColorError(f"Unsupported color format: {value}")
    
    @staticmethod
    def _parse_hex(hex_string: str) -> Tuple[int, int, int, float]:
        """
        Parse hex color string.
        
        Args:
            hex_string: Hex color string
            
        Returns:
            Tuple of (r, g, b, a) components
            
        Raises:
            ColorError: If hex string is invalid
        """
        # Remove # if present
        hex_string = hex_string.lstrip('#')
        
        # Handle different hex formats
        if len(hex_string) == 3:
            # Short form #RGB
            r = int(hex_string[0] + hex_string[0], 16)
            g = int(hex_string[1] + hex_string[1], 16)
            b = int(hex_string[2] + hex_string[2], 16)
            return r, g, b, DEFAULT_ALPHA
            
        elif len(hex_string) == 4:
            # Short form #RGBA
            r = int(hex_string[0] + hex_string[0], 16)
            g = int(hex_string[1] + hex_string[1], 16)
            b = int(hex_string[2] + hex_string[2], 16)
            a = int(hex_string[3] + hex_string[3], 16) / 255
            return r, g, b, a
            
        elif len(hex_string) == 6:
            # Standard form #RRGGBB
            r = int(hex_string[0:2], 16)
            g = int(hex_string[2:4], 16)
            b = int(hex_string[4:6], 16)
            return r, g, b, DEFAULT_ALPHA
            
        elif len(hex_string) == 8:
            # With alpha #RRGGBBAA
            r = int(hex_string[0:2], 16)
            g = int(hex_string[2:4], 16)
            b = int(hex_string[4:6], 16)
            a = int(hex_string[6:8], 16) / 255
            return r, g, b, a
            
        raise ColorError(f"Invalid hex color format: #{hex_string}")
    
    @staticmethod
    def _parse_rgb(rgb_string: str) -> Tuple[int, int, int, float]:
        """
        Parse RGB/RGBA color string.
        
        Args:
            rgb_string: RGB/RGBA color string
            
        Returns:
            Tuple of (r, g, b, a) components
            
        Raises:
            ColorError: If RGB string is invalid
        """
        # Extract values from rgb(r, g, b) or rgba(r, g, b, a)
        is_rgba = rgb_string.startswith('rgba')
        
        # Extract values
        values_str = rgb_string[rgb_string.find('(') + 1:rgb_string.find(')')].split(',')
        
        if is_rgba and len(values_str) != 4:
            raise ColorError(f"Invalid rgba format: {rgb_string}")
        elif not is_rgba and len(values_str) != 3:
            raise ColorError(f"Invalid rgb format: {rgb_string}")
            
        # Parse RGB values
        try:
            r, g, b = [int(v.strip()) for v in values_str[:3]]
        except ValueError:
            # Handle percentage values
            try:
                r, g, b = [int(float(v.strip().rstrip('%')) * 2.55) for v in values_str[:3]]
            except ValueError:
                raise ColorError(f"Invalid RGB values in: {rgb_string}")
                
        # Parse alpha if present
        a = DEFAULT_ALPHA
        if is_rgba:
            try:
                a = float(values_str[3].strip())
            except ValueError:
                raise ColorError(f"Invalid alpha value in: {rgb_string}")
                
        # Validate ranges
        r = min(255, max(0, r))
        g = min(255, max(0, g))
        b = min(255, max(0, b))
        a = min(1.0, max(0.0, a))
        
        return r, g, b, a
    
    @staticmethod
    def _parse_hsl(hsl_string: str) -> Tuple[int, int, int, float]:
        """
        Parse HSL/HSLA color string.
        
        Args:
            hsl_string: HSL/HSLA color string
            
        Returns:
            Tuple of (r, g, b, a) components (converted from HSL)
            
        Raises:
            ColorError: If HSL string is invalid
        """
        # Extract values from hsl(h, s%, l%) or hsla(h, s%, l%, a)
        is_hsla = hsl_string.startswith('hsla')
        
        # Extract values
        values_str = hsl_string[hsl_string.find('(') + 1:hsl_string.find(')')].split(',')
        
        if is_hsla and len(values_str) != 4:
            raise ColorError(f"Invalid hsla format: {hsl_string}")
        elif not is_hsla and len(values_str) != 3:
            raise ColorError(f"Invalid hsl format: {hsl_string}")
            
        # Parse HSL values
        try:
            h = float(values_str[0].strip())
            s = float(values_str[1].strip().rstrip('%')) / 100
            l = float(values_str[2].strip().rstrip('%')) / 100
        except ValueError:
            raise ColorError(f"Invalid HSL values in: {hsl_string}")
            
        # Parse alpha if present
        a = DEFAULT_ALPHA
        if is_hsla:
            try:
                a = float(values_str[3].strip())
            except ValueError:
                raise ColorError(f"Invalid alpha value in: {hsl_string}")
                
        # Validate ranges
        h = h % 360
        s = min(1.0, max(0.0, s))
        l = min(1.0, max(0.0, l))
        a = min(1.0, max(0.0, a))
        
        # Convert HSL to RGB
        r, g, b = Color._hsl_to_rgb(h, s, l)
        
        return r, g, b, a
    
    @staticmethod
    @jit
    def _hsl_to_rgb(h: float, s: float, l: float) -> RGB:
        """
        Convert HSL to RGB.
        
        Args:
            h: Hue (0-360)
            s: Saturation (0.0-1.0)
            l: Lightness (0.0-1.0)
            
        Returns:
            RGB tuple
        """
        # Implementation based on HSL to RGB conversion formula
        h = h / 360.0  # Normalize hue to 0-1
        
        if s == 0:
            # Achromatic (gray)
            r = g = b = l
        else:
            # Helper function for HSL conversion
            def hue_to_rgb(p, q, t):
                if t < 0:
                    t += 1
                if t > 1:
                    t -= 1
                if t < 1/6:
                    return p + (q - p) * 6 * t
                if t < 1/2:
                    return q
                if t < 2/3:
                    return p + (q - p) * (2/3 - t) * 6
                return p
                
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            
            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)
            
        # Convert to 0-255
        return (
            int(round(r * 255)),
            int(round(g * 255)),
            int(round(b * 255))
        )
    
    @staticmethod
    @jit
    def _rgb_to_hsl(r: int, g: int, b: int) -> HSL:
        """
        Convert RGB to HSL.
        
        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            
        Returns:
            HSL tuple
        """
        # Normalize RGB to 0-1
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
        
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        l = (max_val + min_val) / 2
        
        if max_val == min_val:
            # Achromatic (gray)
            h = 0
            s = 0
        else:
            d = max_val - min_val
            s = d / (2 - max_val - min_val) if l > 0.5 else d / (max_val + min_val)
            
            if max_val == r:
                h = (g - b) / d + (6 if g < b else 0)
            elif max_val == g:
                h = (b - r) / d + 2
            else:  # max_val == b
                h = (r - g) / d + 4
                
            h = h * 60
            
        return (
            round(h, COLOR_PRECISION),
            round(s, COLOR_PRECISION),
            round(l, COLOR_PRECISION)
        )


# Color palette for coordinated color schemes
class ColorPalette:
    """
    Memory-efficient color palette for coordinated color schemes.
    Optimized for SVG generation with caching of common operations.
    """
    
    __slots__ = ('_colors', '_name', '_hash')
    
    def __init__(self, colors: List[Color], name: Optional[str] = None):
        """
        Initialize a color palette.
        
        Args:
            colors: List of colors in the palette
            name: Optional name for the palette
        """
        self._colors = tuple(colors)  # Immutable for hashing
        self._name = name
        self._hash = hash(self._colors)
    
    @classmethod
    @memoize
    def from_base_color(
        cls,
        base_color: Color,
        variation_count: int = 5,
        include_complementary: bool = False,
        name: Optional[str] = None
    ) -> 'ColorPalette':
        """
        Create a palette from a base color.
        
        Args:
            base_color: Base color to build palette around
            variation_count: Number of variations to include
            include_complementary: Whether to include complementary colors
            name: Optional name for the palette
            
        Returns:
            ColorPalette instance
        """
        colors = [base_color]
        
        # Add variations (lighter/darker)
        step = 1.0 / (variation_count + 1)
        for i in range(1, variation_count + 1):
            # Add lighter variation
            colors.append(base_color.lighten(step * i))
            # Add darker variation
            colors.append(base_color.darken(step * i))
            
        # Add complementary if requested
        if include_complementary:
            comp_color = base_color.complement()
            colors.append(comp_color)
            # Add variations of complementary
            for i in range(1, min(3, variation_count)):
                colors.append(comp_color.lighten(step * i))
                colors.append(comp_color.darken(step * i))
                
        return cls(colors, name)
    
    @classmethod
    @memoize
    def analogous(
        cls,
        base_color: Color,
        count: int = 5,
        angle: float = 30,
        name: Optional[str] = None
    ) -> 'ColorPalette':
        """
        Create an analogous color palette.
        
        Args:
            base_color: Base color to build palette around
            count: Number of colors to include
            angle: Angle between colors in degrees
            name: Optional name for the palette
            
        Returns:
            ColorPalette instance
        """
        colors = [base_color]
        h, s, l = base_color.hsl
        
        # Generate colors on both sides of the base hue
        for i in range(1, count // 2 + 1):
            # Add color with higher hue
            h1 = (h + i * angle) % 360
            colors.append(Color.from_hsl(h1, s, l, base_color.alpha))
            
            # Add color with lower hue (if not at the count limit)
            if len(colors) < count:
                h2 = (h - i * angle) % 360
                colors.append(Color.from_hsl(h2, s, l, base_color.alpha))
                
        return cls(colors, name)
    
    @classmethod
    @memoize
    def monochromatic(
        cls,
        base_color: Color,
        count: int = 5,
        name: Optional[str] = None
    ) -> 'ColorPalette':
        """
        Create a monochromatic color palette.
        
        Args:
            base_color: Base color to build palette around
            count: Number of colors to include
            name: Optional name for the palette
            
        Returns:
            ColorPalette instance
        """
        colors = [base_color]
        h, s, l = base_color.hsl
        
        # Generate variations with different lightness
        l_step = min(0.8, max(0.1, 0.9 / count))
        l_values = []
        
        # Start from the middle lightness and work outward
        base_l_index = 0
        for i in range(1, count):
            if i % 2 == 1:
                # Go lighter
                base_l_index += 1
                new_l = min(0.95, l + base_l_index * l_step)
            else:
                # Go darker
                new_l = max(0.05, l - base_l_index * l_step)
                
            l_values.append(new_l)
            
        # Sort lightness values from darkest to lightest
        l_values.sort()
        
        # Create colors
        for new_l in l_values:
            if len(colors) < count:
                colors.append(Color.from_hsl(h, s, new_l, base_color.alpha))
                
        return cls(colors, name)
    
    @classmethod
    @memoize
    def triadic(
        cls,
        base_color: Color,
        variations_per_color: int = 2,
        name: Optional[str] = None
    ) -> 'ColorPalette':
        """
        Create a triadic color palette.
        
        Args:
            base_color: Base color to build palette around
            variations_per_color: Number of variations per base color
            name: Optional name for the palette
            
        Returns:
            ColorPalette instance
        """
        h, s, l = base_color.hsl
        
        # Create triadic base colors (120° apart)
        color1 = base_color
        color2 = Color.from_hsl((h + 120) % 360, s, l, base_color.alpha)
        color3 = Color.from_hsl((h + 240) % 360, s, l, base_color.alpha)
        
        colors = [color1, color2, color3]
        
        # Add variations for each base color
        for base in [color1, color2, color3]:
            for i in range(1, variations_per_color + 1):
                step = 0.1 * i
                colors.append(base.lighten(step))
                colors.append(base.darken(step))
                
        return cls(colors, name)
    
    @property
    def colors(self) -> List[Color]:
        """Get the list of colors in the palette."""
        return list(self._colors)
    
    @property
    def name(self) -> Optional[str]:
        """Get the palette name."""
        return self._name
    
    @property
    def count(self) -> int:
        """Get the number of colors in the palette."""
        return len(self._colors)
    
    def get_color(self, index: int) -> Color:
        """
        Get color at a specific index.
        
        Args:
            index: Index of the color to get
            
        Returns:
            Color at the specified index
            
        Raises:
            IndexError: If index is out of range
        """
        return self._colors[index % len(self._colors)]
    
    def get_random_color(self) -> Color:
        """
        Get a random color from the palette.
        
        Returns:
            Random color from the palette
        """
        import random
        return random.choice(self._colors)
    
    def to_hex_list(self) -> List[str]:
        """
        Get list of hex color codes.
        
        Returns:
            List of hex color strings
        """
        return [c.hex for c in self._colors]
    
    def to_rgb_list(self) -> List[RGB]:
        """
        Get list of RGB tuples.
        
        Returns:
            List of RGB tuples
        """
        return [c.rgb for c in self._colors]
    
    def to_css_variables(self, prefix: str = 'color') -> Dict[str, str]:
        """
        Generate CSS variable definitions for the palette.
        
        Args:
            prefix: Prefix for variable names
            
        Returns:
            Dictionary of CSS variable names to color values
        """
        variables = {}
        
        for i, color in enumerate(self._colors):
            variables[f'--{prefix}-{i}'] = color.to_css_string()
            
        return variables
    
    def __len__(self) -> int:
        """Get number of colors in the palette."""
        return len(self._colors)
    
    def __getitem__(self, index: int) -> Color:
        """Get color at index."""
        return self._colors[index % len(self._colors)]
    
    def __iter__(self):
        """Iterate over colors in the palette."""
        return iter(self._colors)
    
    def __eq__(self, other: object) -> bool:
        """Compare palettes for equality."""
        if not isinstance(other, ColorPalette):
            return False
        return self._colors == other._colors
    
    def __hash__(self) -> int:
        """Get hash of palette for use in dictionaries and sets."""
        return self._hash
    
    def __str__(self) -> str:
        """Get string representation of palette."""
        if self._name:
            return f"ColorPalette({self._name}, {len(self._colors)} colors)"
        return f"ColorPalette({len(self._colors)} colors)"
    
    def __repr__(self) -> str:
        """Get detailed string representation of palette."""
        colors_repr = ', '.join(str(c) for c in self._colors[:3])
        if len(self._colors) > 3:
            colors_repr += f", ... ({len(self._colors) - 3} more)"
        return f"ColorPalette([{colors_repr}], name={self._name!r})"


# Utility functions for color operations

@memoize
def parse_color(value: str) -> Color:
    """
    Parse color string into Color object.
    
    Args:
        value: Color string in various formats
        
    Returns:
        Color object
        
    Raises:
        ColorError: If color string can't be parsed
    """
    return Color(value)


@memoize
def blend_colors(colors: List[Color], weights: Optional[List[float]] = None) -> Color:
    """
    Blend multiple colors together.
    
    Args:
        colors: List of colors to blend
        weights: Optional list of weights for each color
        
    Returns:
        Blended color
        
    Raises:
        ValueError: If weights don't match colors length
    """
    if not colors:
        raise ValueError("No colors provided for blending")
        
    if len(colors) == 1:
        return colors[0]
        
    # Normalize weights
    if weights is None:
        weights = [1.0 / len(colors)] * len(colors)
    elif len(weights) != len(colors):
        raise ValueError("Number of weights must match number of colors")
        
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Calculate weighted average of components
    r = sum(c.red * w for c, w in zip(colors, weights))
    g = sum(c.green * w for c, w in zip(colors, weights))
    b = sum(c.blue * w for c, w in zip(colors, weights))
    a = sum(c.alpha * w for c, w in zip(colors, weights))
    
    return Color.from_rgb(int(r), int(g), int(b), a)


@memoize
def generate_gradient_colors(
    start_color: Color,
    end_color: Color,
    steps: int
) -> List[Color]:
    """
    Generate a list of colors representing a gradient.
    
    Args:
        start_color: Starting color
        end_color: Ending color
        steps: Number of colors to generate
        
    Returns:
        List of colors representing the gradient
    """
    if steps < 2:
        return [start_color]
        
    result = [start_color]
    
    for i in range(1, steps - 1):
        ratio = i / (steps - 1)
        result.append(start_color.blend(end_color, ratio))
        
    result.append(end_color)
    return result


@memoize
def get_contrast_color(
    color: Color,
    light_color: Color = None,
    dark_color: Color = None
) -> Color:
    """
    Get a contrasting color (light or dark) based on luminance.
    
    Args:
        color: Base color
        light_color: Light color to use (default: white)
        dark_color: Dark color to use (default: black)
        
    Returns:
        Contrasting color
    """
    if light_color is None:
        light_color = Color.from_rgb(255, 255, 255)
        
    if dark_color is None:
        dark_color = Color.from_rgb(0, 0, 0)
        
    return light_color if color.luminance < 0.5 else dark_color


@memoize
def find_nearest_color(target: Color, color_list: List[Color]) -> Color:
    """
    Find the nearest color in a list to the target color.
    
    Args:
        target: Target color to match
        color_list: List of colors to search
        
    Returns:
        Nearest color from the list
        
    Raises:
        ValueError: If color_list is empty
    """
    if not color_list:
        raise ValueError("Empty color list provided")
        
    return min(color_list, key=lambda c: target.distance(c))