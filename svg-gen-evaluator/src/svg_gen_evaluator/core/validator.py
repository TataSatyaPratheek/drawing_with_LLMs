"""
SVG validation utilities to ensure SVGs meet competition requirements.
"""
import logging
from typing import Dict, Set, Tuple, Union

logger = logging.getLogger(__name__)


class SVGValidator:
    """
    Validates SVGs against competition constraints.
    
    Ensures that SVGs meet size limits, use only allowed elements and attributes,
    and don't contain external or embedded resources.
    """
    
    def __init__(self, max_svg_size: int = 10000):
        """
        Initialize the SVG validator.
        
        Args:
            max_svg_size: Maximum allowed size of an SVG file in bytes
        """
        self.max_svg_size = max_svg_size
        self.allowed_elements = self._get_allowed_elements()
        
        # Verify defusedxml is installed
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if necessary dependencies are installed."""
        try:
            from defusedxml import ElementTree
            self.ElementTree = ElementTree
        except ImportError:
            logger.error("defusedxml package not installed. Installing...")
            import subprocess
            import sys
            
            subprocess.check_call([sys.executable, "-m", "pip", "install", "defusedxml"])
            from defusedxml import ElementTree
            self.ElementTree = ElementTree
    
    def _get_allowed_elements(self) -> Dict[str, Set[str]]:
        """
        Define allowed SVG elements and attributes.
        
        Returns:
            Dictionary mapping element names to sets of allowed attributes
        """
        # Common attributes allowed for all elements
        common_attrs = {
            'id', 'clip-path', 'clip-rule', 'color', 'color-interpolation',
            'color-interpolation-filters', 'color-rendering', 'display',
            'fill', 'fill-opacity', 'fill-rule', 'filter', 'flood-color',
            'flood-opacity', 'lighting-color', 'marker-end', 'marker-mid',
            'marker-start', 'mask', 'opacity', 'paint-order', 'stop-color',
            'stop-opacity', 'stroke', 'stroke-dasharray', 'stroke-dashoffset',
            'stroke-linecap', 'stroke-linejoin', 'stroke-miterlimit',
            'stroke-opacity', 'stroke-width', 'transform',
        }
        
        # Define element-specific attributes
        allowed = {
            'common': common_attrs,
            'svg': {'width', 'height', 'viewBox', 'preserveAspectRatio'},
            'g': {'viewBox'},
            'defs': set(),
            'symbol': {'viewBox', 'x', 'y', 'width', 'height'},
            'use': {'x', 'y', 'width', 'height', 'href'},
            'marker': {
                'viewBox', 'preserveAspectRatio', 'refX', 'refY',
                'markerUnits', 'markerWidth', 'markerHeight', 'orient',
            },
            'pattern': {
                'viewBox', 'preserveAspectRatio', 'x', 'y',
                'width', 'height', 'patternUnits', 'patternContentUnits',
                'patternTransform', 'href',
            },
            'linearGradient': {
                'x1', 'x2', 'y1', 'y2', 'gradientUnits',
                'gradientTransform', 'spreadMethod', 'href',
            },
            'radialGradient': {
                'cx', 'cy', 'r', 'fx', 'fy', 'fr',
                'gradientUnits', 'gradientTransform', 'spreadMethod', 'href',
            },
            'stop': {'offset'},
            'filter': {
                'x', 'y', 'width', 'height', 'filterUnits', 'primitiveUnits',
            },
            'feBlend': {'result', 'in', 'in2', 'mode'},
            'feColorMatrix': {'result', 'in', 'type', 'values'},
            'feComposite': {
                'result', 'style', 'in', 'in2', 'operator', 'k1', 'k2', 'k3', 'k4',
            },
            'feFlood': {'result'},
            'feGaussianBlur': {'result', 'in', 'stdDeviation', 'edgeMode'},
            'feMerge': {'result', 'x', 'y', 'width', 'height', 'result'},
            'feMergeNode': {'result', 'in'},
            'feOffset': {'result', 'in', 'dx', 'dy'},
            'feTurbulence': {
                'result', 'baseFrequency', 'numOctaves',
                'seed', 'stitchTiles', 'type',
            },
            'path': {'d'},
            'rect': {'x', 'y', 'width', 'height', 'rx', 'ry'},
            'circle': {'cx', 'cy', 'r'},
            'ellipse': {'cx', 'cy', 'rx', 'ry'},
            'line': {'x1', 'y1', 'x2', 'y2'},
            'polyline': {'points'},
            'polygon': {'points'},
        }
        
        return allowed
    
    def validate(self, svg_code: str) -> Tuple[bool, Union[str, None]]:
        """
        Validate an SVG string against competition constraints.
        
        Args:
            svg_code: The SVG string to validate
            
        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        # Check file size
        svg_size = len(svg_code.encode('utf-8'))
        if svg_size > self.max_svg_size:
            return False, f"SVG exceeds allowed size: {svg_size} bytes (max: {self.max_svg_size})"
        
        try:
            # Parse XML using defusedxml to prevent XXE attacks
            tree = self.ElementTree.fromstring(
                svg_code.encode('utf-8'),
                forbid_dtd=True,
                forbid_entities=True,
                forbid_external=True,
            )
        except Exception as e:
            return False, f"Invalid XML: {str(e)}"
        
        allowed_elements = set(self.allowed_elements.keys())
        
        # Check elements and attributes
        for element in tree.iter():
            # Check for disallowed elements
            tag_name = element.tag.split('}')[-1]
            if tag_name not in allowed_elements:
                return False, f"Disallowed element: {tag_name}"
            
            # Check attributes
            for attr, attr_value in element.attrib.items():
                # Check for disallowed attributes
                attr_name = attr.split('}')[-1]
                if (
                    attr_name not in self.allowed_elements[tag_name]
                    and attr_name not in self.allowed_elements['common']):
                    return False, f"Disallowed attribute: {attr_name} on element {tag_name}"
                
                # Check for embedded data
                if 'data:' in attr_value.lower():
                    return False, f"Embedded data not allowed in attribute: {attr_name}"
                if ';base64' in attr_value:
                    return False, f"Base64 encoded content not allowed in attribute: {attr_name}"
                
                # Check that href attributes are internal references
                if attr_name == 'href' and not attr_value.startswith('#'):
                    return False, f"Invalid href attribute in <{tag_name}>. Only internal references (starting with '#') are allowed."
        
        # If we've made it this far, the SVG is valid
        return True, None