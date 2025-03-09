"""
Production-grade shape models for SVG generation and manipulation.
Provides optimized implementations of SVG shapes with path generation
capabilities and transformation support.
"""

import re
import math
from enum import Enum, auto
from typing import Dict, List, Tuple, Union, Optional, Any, NamedTuple, Set, Callable
import xml.etree.ElementTree as ET
import hashlib
import threading

# Import core optimizations
from core import CONFIG, memoize, jit, Profiler
from utils.logger import get_logger
from models.color import Color

# Configure logger
logger = get_logger(__name__)

# Type definitions
Point = Tuple[float, float]
BoundingBox = Tuple[float, float, float, float]  # x, y, width, height
Matrix = Tuple[float, float, float, float, float, float]  # a, b, c, d, e, f (SVG transform)
PathData = str  # SVG path data string

# Constants
DEFAULT_STROKE_WIDTH = 1.0
DEFAULT_OPACITY = 1.0
DEFAULT_TRANSFORM = (1, 0, 0, 1, 0, 0)  # Identity matrix
IDENTITY_MATRIX = DEFAULT_TRANSFORM
PATH_PRECISION = 3  # Decimal places for path data
SHAPE_ID_PREFIX = "shape_"

# Thread-local storage for ID generation
_thread_local = threading.local()


class ShapeType(Enum):
    """Enum for different SVG shape types."""
    RECT = auto()
    CIRCLE = auto()
    ELLIPSE = auto()
    LINE = auto()
    POLYLINE = auto()
    POLYGON = auto()
    PATH = auto()
    TEXT = auto()
    GROUP = auto()
    CUSTOM = auto()


class ShapeError(Exception):
    """Custom exception for shape-related errors."""
    pass


class Transform:
    """
    Memory-efficient SVG transformation matrix with operation caching.
    
    Represents a 2D transformation matrix in the form:
    [a c e]
    [b d f]
    [0 0 1]
    
    Used for transforming shapes in SVG (scale, rotate, translate, etc.)
    """
    
    __slots__ = ('_matrix', '_hash')
    
    def __init__(self, matrix: Matrix = DEFAULT_TRANSFORM):
        """
        Initialize transformation matrix.
        
        Args:
            matrix: Transformation matrix (a, b, c, d, e, f)
        """
        self._matrix = matrix
        self._hash = hash(self._matrix)
    
    @classmethod
    def identity(cls) -> 'Transform':
        """Create identity transformation."""
        return cls(IDENTITY_MATRIX)
    
    @classmethod
    def translate(cls, tx: float, ty: float) -> 'Transform':
        """
        Create translation transformation.
        
        Args:
            tx: Translation in x direction
            ty: Translation in y direction
            
        Returns:
            Translation transform
        """
        return cls((1, 0, 0, 1, tx, ty))
    
    @classmethod
    def scale(cls, sx: float, sy: Optional[float] = None) -> 'Transform':
        """
        Create scaling transformation.
        
        Args:
            sx: Scale factor in x direction
            sy: Scale factor in y direction (defaults to sx)
            
        Returns:
            Scaling transform
        """
        if sy is None:
            sy = sx
        return cls((sx, 0, 0, sy, 0, 0))
    
    @classmethod
    def rotate(cls, angle: float, cx: float = 0, cy: float = 0) -> 'Transform':
        """
        Create rotation transformation.
        
        Args:
            angle: Rotation angle in degrees
            cx: Center of rotation, x-coordinate
            cy: Center of rotation, y-coordinate
            
        Returns:
            Rotation transform
        """
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Translate to origin, rotate, translate back
        if cx == 0 and cy == 0:
            return cls((cos_a, sin_a, -sin_a, cos_a, 0, 0))
        else:
            return (cls.translate(cx, cy)
                   .multiply(cls((cos_a, sin_a, -sin_a, cos_a, 0, 0)))
                   .multiply(cls.translate(-cx, -cy)))
    
    @classmethod
    def skewX(cls, angle: float) -> 'Transform':
        """
        Create horizontal skew transformation.
        
        Args:
            angle: Skew angle in degrees
            
        Returns:
            Horizontal skew transform
        """
        angle_rad = math.radians(angle)
        return cls((1, 0, math.tan(angle_rad), 1, 0, 0))
    
    @classmethod
    def skewY(cls, angle: float) -> 'Transform':
        """
        Create vertical skew transformation.
        
        Args:
            angle: Skew angle in degrees
            
        Returns:
            Vertical skew transform
        """
        angle_rad = math.radians(angle)
        return cls((1, math.tan(angle_rad), 0, 1, 0, 0))
    
    @classmethod
    def from_string(cls, transform_str: str) -> 'Transform':
        """
        Parse SVG transform string.
        
        Args:
            transform_str: SVG transform string (e.g., "translate(10,20) scale(2)")
            
        Returns:
            Transform instance
            
        Raises:
            ShapeError: If transform string is invalid
        """
        transform = cls.identity()
        
        if not transform_str or transform_str.strip() == "":
            return transform
            
        try:
            # Extract transform operations
            pattern = r'(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)'
            for op, args in re.findall(pattern, transform_str):
                # Parse arguments
                args = [float(x.strip()) for x in args.split(',')]
                
                # Apply transform operation
                if op == 'matrix' and len(args) == 6:
                    transform = transform.multiply(cls(tuple(args)))
                    
                elif op == 'translate':
                    if len(args) == 1:
                        args.append(0)
                    if len(args) == 2:
                        transform = transform.multiply(cls.translate(args[0], args[1]))
                        
                elif op == 'scale':
                    if len(args) == 1:
                        args.append(args[0])
                    if len(args) == 2:
                        transform = transform.multiply(cls.scale(args[0], args[1]))
                        
                elif op == 'rotate':
                    if len(args) == 1:
                        transform = transform.multiply(cls.rotate(args[0]))
                    elif len(args) == 3:
                        transform = transform.multiply(cls.rotate(args[0], args[1], args[2]))
                        
                elif op == 'skewX' and len(args) == 1:
                    transform = transform.multiply(cls.skewX(args[0]))
                    
                elif op == 'skewY' and len(args) == 1:
                    transform = transform.multiply(cls.skewY(args[0]))
                
            return transform
            
        except Exception as e:
            raise ShapeError(f"Invalid transform string: {transform_str} - {str(e)}")
    
    @property
    def matrix(self) -> Matrix:
        """Get the transformation matrix."""
        return self._matrix
    
    @property
    def is_identity(self) -> bool:
        """Check if this is an identity transform."""
        return self._matrix == IDENTITY_MATRIX
    
    @property
    @memoize
    def determinant(self) -> float:
        """Calculate the determinant of the transformation matrix."""
        a, b, c, d, _, _ = self._matrix
        return a * d - b * c
    
    @property
    @memoize
    def is_invertible(self) -> bool:
        """Check if the transformation is invertible."""
        return abs(self.determinant) > 1e-10
    
    @property
    @memoize
    def inverse(self) -> Optional['Transform']:
        """
        Get the inverse transformation if it exists.
        
        Returns:
            Inverse transform or None if not invertible
        """
        if not self.is_invertible:
            return None
            
        a, b, c, d, e, f = self._matrix
        det = self.determinant
        
        inv_a = d / det
        inv_b = -b / det
        inv_c = -c / det
        inv_d = a / det
        inv_e = (c * f - d * e) / det
        inv_f = (b * e - a * f) / det
        
        return Transform((inv_a, inv_b, inv_c, inv_d, inv_e, inv_f))
    
    def multiply(self, other: 'Transform') -> 'Transform':
        """
        Multiply with another transformation (this * other).
        
        Args:
            other: Transform to multiply with
            
        Returns:
            Combined transform
        """
        a1, b1, c1, d1, e1, f1 = self._matrix
        a2, b2, c2, d2, e2, f2 = other._matrix
        
        a = a1 * a2 + c1 * b2
        b = b1 * a2 + d1 * b2
        c = a1 * c2 + c1 * d2
        d = b1 * c2 + d1 * d2
        e = a1 * e2 + c1 * f2 + e1
        f = b1 * e2 + d1 * f2 + f1
        
        return Transform((a, b, c, d, e, f))
    
    def transform_point(self, point: Point) -> Point:
        """
        Transform a point.
        
        Args:
            point: Point to transform (x, y)
            
        Returns:
            Transformed point
        """
        x, y = point
        a, b, c, d, e, f = self._matrix
        
        return (
            a * x + c * y + e,
            b * x + d * y + f
        )
    
    def transform_points(self, points: List[Point]) -> List[Point]:
        """
        Transform multiple points.
        
        Args:
            points: List of points to transform
            
        Returns:
            List of transformed points
        """
        a, b, c, d, e, f = self._matrix
        
        return [
            (a * x + c * y + e, b * x + d * y + f)
            for x, y in points
        ]
    
    def transform_bounding_box(self, bbox: BoundingBox) -> BoundingBox:
        """
        Transform a bounding box.
        
        Args:
            bbox: Bounding box to transform (x, y, width, height)
            
        Returns:
            Transformed bounding box
        """
        x, y, width, height = bbox
        
        # Transform all four corners
        corners = [
            (x, y),
            (x + width, y),
            (x, y + height),
            (x + width, y + height)
        ]
        
        transformed_corners = self.transform_points(corners)
        
        # Calculate new bounds
        xs = [p[0] for p in transformed_corners]
        ys = [p[1] for p in transformed_corners]
        
        new_x = min(xs)
        new_y = min(ys)
        new_width = max(xs) - new_x
        new_height = max(ys) - new_y
        
        return (new_x, new_y, new_width, new_height)
    
    def to_svg_string(self) -> str:
        """
        Convert to SVG transform attribute string.
        
        Returns:
            SVG transform string
        """
        if self.is_identity:
            return ""
            
        a, b, c, d, e, f = self._matrix
        return f"matrix({a:.6g},{b:.6g},{c:.6g},{d:.6g},{e:.6g},{f:.6g})"
    
    def __eq__(self, other: object) -> bool:
        """Check if transforms are equal."""
        if not isinstance(other, Transform):
            return False
            
        # Compare each component with small epsilon for floating point error
        for a, b in zip(self._matrix, other._matrix):
            if abs(a - b) > 1e-10:
                return False
                
        return True
    
    def __hash__(self) -> int:
        """Hash for dictionary keys."""
        return self._hash
    
    def __str__(self) -> str:
        """String representation."""
        return self.to_svg_string()
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"Transform({self._matrix})"


class Shape:
    """
    Base class for SVG shapes with optimized memory usage.
    
    Provides common functionality for all shapes including
    transformations, styling, and SVG generation.
    """
    
    __slots__ = ('_id', '_type', '_fill', '_stroke', '_stroke_width', '_opacity',
                '_transform', '_attributes', '_class_name', '_data', '_hash',
                '_bounding_box_cache')
    
    def __init__(
        self,
        shape_id: Optional[str] = None,
        shape_type: ShapeType = ShapeType.CUSTOM,
        fill: Optional[Union[Color, str]] = None,
        stroke: Optional[Union[Color, str]] = None,
        stroke_width: float = DEFAULT_STROKE_WIDTH,
        opacity: float = DEFAULT_OPACITY,
        transform: Optional[Union[Transform, Matrix, str]] = None,
        class_name: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new shape.
        
        Args:
            shape_id: Unique ID for the shape (generated if None)
            shape_type: Type of shape
            fill: Fill color or None for no fill
            stroke: Stroke color or None for no stroke
            stroke_width: Width of stroke
            opacity: Opacity (0.0-1.0)
            transform: Transformation
            class_name: CSS class name
            attributes: Additional SVG attributes
            data: Custom data for the shape
        """
        # Generate ID if not provided
        self._id = shape_id or self._generate_id()
        self._type = shape_type
        
        # Process colors
        self._fill = self._process_color(fill)
        self._stroke = self._process_color(stroke)
        self._stroke_width = max(0, stroke_width)
        self._opacity = max(0, min(1, opacity))
        
        # Process transform
        if transform is None:
            self._transform = Transform.identity()
        elif isinstance(transform, Transform):
            self._transform = transform
        elif isinstance(transform, tuple) and len(transform) == 6:
            self._transform = Transform(transform)
        elif isinstance(transform, str):
            self._transform = Transform.from_string(transform)
        else:
            raise ShapeError(f"Invalid transform: {transform}")
            
        # Additional properties
        self._class_name = class_name
        self._attributes = attributes or {}
        self._data = data or {}
        
        # Cache for bounding box
        self._bounding_box_cache = None
        
        # Compute hash
        self._hash = None  # Computed lazily on first use
    
    @staticmethod
    def _generate_id() -> str:
        """Generate a unique ID for a shape."""
        # Use thread-local counter to avoid collisions
        if not hasattr(_thread_local, 'shape_id_counter'):
            _thread_local.shape_id_counter = 0
            
        _thread_local.shape_id_counter += 1
        
        # Include randomness for uniqueness
        import uuid
        random_part = uuid.uuid4().hex[:8]
        
        return f"{SHAPE_ID_PREFIX}{_thread_local.shape_id_counter}_{random_part}"
    
    @staticmethod
    def _process_color(color: Optional[Union[Color, str]]) -> Optional[Color]:
        """
        Process color input.
        
        Args:
            color: Color input (Color object, string, or None)
            
        Returns:
            Processed Color object or None
        """
        if color is None:
            return None
            
        if isinstance(color, Color):
            return color
            
        # Convert string to Color
        from models.color import Color
        try:
            return Color(color)
        except Exception as e:
            logger.warning(f"Invalid color: {color} - {str(e)}")
            return None
    
    @property
    def id(self) -> str:
        """Get shape ID."""
        return self._id
    
    @property
    def type(self) -> ShapeType:
        """Get shape type."""
        return self._type
    
    @property
    def fill(self) -> Optional[Color]:
        """Get fill color."""
        return self._fill
    
    @property
    def stroke(self) -> Optional[Color]:
        """Get stroke color."""
        return self._stroke
    
    @property
    def stroke_width(self) -> float:
        """Get stroke width."""
        return self._stroke_width
    
    @property
    def opacity(self) -> float:
        """Get opacity."""
        return self._opacity
    
    @property
    def transform(self) -> Transform:
        """Get transformation."""
        return self._transform
    
    @property
    def class_name(self) -> Optional[str]:
        """Get CSS class name."""
        return self._class_name
    
    @property
    def attributes(self) -> Dict[str, str]:
        """Get additional SVG attributes."""
        return self._attributes.copy()
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get custom data."""
        return self._data.copy()
    
    def with_id(self, shape_id: str) -> 'Shape':
        """
        Create a copy with a new ID.
        
        Args:
            shape_id: New shape ID
            
        Returns:
            New shape with updated ID
        """
        shape = self.copy()
        shape._id = shape_id
        shape._hash = None
        return shape
    
    def with_fill(self, fill: Optional[Union[Color, str]]) -> 'Shape':
        """
        Create a copy with a new fill color.
        
        Args:
            fill: New fill color or None for no fill
            
        Returns:
            New shape with updated fill
        """
        shape = self.copy()
        shape._fill = self._process_color(fill)
        shape._hash = None
        return shape
    
    def with_stroke(
        self,
        stroke: Optional[Union[Color, str]],
        stroke_width: Optional[float] = None
    ) -> 'Shape':
        """
        Create a copy with a new stroke.
        
        Args:
            stroke: New stroke color or None for no stroke
            stroke_width: New stroke width or None to keep current
            
        Returns:
            New shape with updated stroke
        """
        shape = self.copy()
        shape._stroke = self._process_color(stroke)
        
        if stroke_width is not None:
            shape._stroke_width = max(0, stroke_width)
            
        shape._hash = None
        return shape
    
    def with_opacity(self, opacity: float) -> 'Shape':
        """
        Create a copy with a new opacity.
        
        Args:
            opacity: New opacity (0.0-1.0)
            
        Returns:
            New shape with updated opacity
        """
        shape = self.copy()
        shape._opacity = max(0, min(1, opacity))
        shape._hash = None
        return shape
    
    def with_transform(
        self,
        transform: Optional[Union[Transform, Matrix, str]]
    ) -> 'Shape':
        """
        Create a copy with a new transformation.
        
        Args:
            transform: New transformation
            
        Returns:
            New shape with updated transformation
        """
        shape = self.copy()
        
        if transform is None:
            shape._transform = Transform.identity()
        elif isinstance(transform, Transform):
            shape._transform = transform
        elif isinstance(transform, tuple) and len(transform) == 6:
            shape._transform = Transform(transform)
        elif isinstance(transform, str):
            shape._transform = Transform.from_string(transform)
        else:
            raise ShapeError(f"Invalid transform: {transform}")
            
        shape._hash = None
        shape._bounding_box_cache = None
        return shape
    
    def with_class(self, class_name: Optional[str]) -> 'Shape':
        """
        Create a copy with a new CSS class name.
        
        Args:
            class_name: New CSS class name
            
        Returns:
            New shape with updated class
        """
        shape = self.copy()
        shape._class_name = class_name
        shape._hash = None
        return shape
    
    def with_attributes(
        self,
        attributes: Dict[str, str],
        replace: bool = False
    ) -> 'Shape':
        """
        Create a copy with updated attributes.
        
        Args:
            attributes: New attributes to add
            replace: Whether to replace all attributes or merge
            
        Returns:
            New shape with updated attributes
        """
        shape = self.copy()
        
        if replace:
            shape._attributes = attributes.copy()
        else:
            new_attrs = shape._attributes.copy()
            new_attrs.update(attributes)
            shape._attributes = new_attrs
            
        shape._hash = None
        return shape
    
    def apply_transform(self, transform: Transform) -> 'Shape':
        """
        Apply an additional transformation.
        
        Args:
            transform: Transform to apply
            
        Returns:
            New shape with combined transformation
        """
        # Skip identity transforms
        if transform.is_identity:
            return self
            
        # Combine with existing transform
        new_transform = self._transform.multiply(transform)
        
        # Create copy with new transform
        shape = self.copy()
        shape._transform = new_transform
        shape._hash = None
        shape._bounding_box_cache = None
        
        return shape
    
    def translate(self, tx: float, ty: float) -> 'Shape':
        """
        Apply translation.
        
        Args:
            tx: Translation in x direction
            ty: Translation in y direction
            
        Returns:
            New shape with translation applied
        """
        return self.apply_transform(Transform.translate(tx, ty))
    
    def scale(self, sx: float, sy: Optional[float] = None) -> 'Shape':
        """
        Apply scaling.
        
        Args:
            sx: Scale factor in x direction
            sy: Scale factor in y direction (defaults to sx)
            
        Returns:
            New shape with scaling applied
        """
        return self.apply_transform(Transform.scale(sx, sy))
    
    def rotate(self, angle: float, cx: float = 0, cy: float = 0) -> 'Shape':
        """
        Apply rotation.
        
        Args:
            angle: Rotation angle in degrees
            cx: Center of rotation, x-coordinate
            cy: Center of rotation, y-coordinate
            
        Returns:
            New shape with rotation applied
        """
        return self.apply_transform(Transform.rotate(angle, cx, cy))
    
    def skew_x(self, angle: float) -> 'Shape':
        """
        Apply horizontal skew.
        
        Args:
            angle: Skew angle in degrees
            
        Returns:
            New shape with horizontal skew applied
        """
        return self.apply_transform(Transform.skewX(angle))
    
    def skew_y(self, angle: float) -> 'Shape':
        """
        Apply vertical skew.
        
        Args:
            angle: Skew angle in degrees
            
        Returns:
            New shape with vertical skew applied
        """
        return self.apply_transform(Transform.skewY(angle))
    
    def get_bounding_box(self) -> BoundingBox:
        """
        Get the bounding box of the shape.
        
        Returns:
            Bounding box (x, y, width, height)
        """
        # Use cached value if available
        if self._bounding_box_cache is not None:
            return self._bounding_box_cache
            
        # Default implementation should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement get_bounding_box")
    
    def contains_point(self, point: Point) -> bool:
        """
        Check if the shape contains a point.
        
        Args:
            point: Point to check (x, y)
            
        Returns:
            True if the shape contains the point
        """
        # Default implementation uses bounding box
        # Subclasses should override with more precise checks
        x, y = point
        bbox_x, bbox_y, bbox_width, bbox_height = self.get_bounding_box()
        
        return (
            x >= bbox_x and
            y >= bbox_y and
            x <= bbox_x + bbox_width and
            y <= bbox_y + bbox_height
        )
    
    def to_path_data(self) -> PathData:
        """
        Convert shape to SVG path data.
        
        Returns:
            SVG path data string
        """
        # Default implementation should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement to_path_data")
    
    def to_svg_element(self) -> ET.Element:
        """
        Convert shape to SVG element.
        
        Returns:
            XML element representing the shape
        """
        # Default implementation should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement to_svg_element")
    
    def _add_common_attributes(self, element: ET.Element) -> None:
        """
        Add common attributes to an SVG element.
        
        Args:
            element: XML element to add attributes to
        """
        # Add ID if not empty
        if self._id:
            element.set('id', self._id)
            
        # Add class if not empty
        if self._class_name:
            element.set('class', self._class_name)
            
        # Add fill
        if self._fill is None:
            element.set('fill', 'none')
        else:
            element.set('fill', self._fill.to_svg_string())
            
        # Add stroke
        if self._stroke is not None:
            element.set('stroke', self._stroke.to_svg_string())
            element.set('stroke-width', str(self._stroke_width))
            
        # Add opacity if not default
        if self._opacity < 0.999:
            element.set('opacity', f"{self._opacity:.6g}")
            
        # Add transform if not identity
        if not self._transform.is_identity:
            element.set('transform', self._transform.to_svg_string())
            
        # Add additional attributes
        for name, value in self._attributes.items():
            element.set(name, value)
    
    def to_svg_string(self) -> str:
        """
        Convert shape to SVG string.
        
        Returns:
            SVG string representation
        """
        element = self.to_svg_element()
        
        # Use ElementTree to generate SVG string
        from xml.etree.ElementTree import tostring
        return tostring(element, encoding='unicode')
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert shape to dictionary representation.
        
        Returns:
            Dictionary with shape data
        """
        return {
            'id': self._id,
            'type': self._type.name,
            'fill': str(self._fill) if self._fill else None,
            'stroke': str(self._stroke) if self._stroke else None,
            'stroke_width': self._stroke_width,
            'opacity': self._opacity,
            'transform': str(self._transform) if not self._transform.is_identity else None,
            'class_name': self._class_name,
            'attributes': self._attributes,
            'data': self._data
        }
    
    def copy(self) -> 'Shape':
        """
        Create a deep copy of the shape.
        
        Returns:
            Copied shape
        """
        # Default implementation - subclasses should override
        # for more efficient copying of specific attributes
        return self.__class__(
            shape_id=self._id,
            shape_type=self._type,
            fill=self._fill,
            stroke=self._stroke,
            stroke_width=self._stroke_width,
            opacity=self._opacity,
            transform=self._transform,
            class_name=self._class_name,
            attributes=self._attributes.copy(),
            data=self._data.copy()
        )
    
    def __eq__(self, other: object) -> bool:
        """Check if shapes are equal."""
        if not isinstance(other, Shape):
            return False
            
        return hash(self) == hash(other)
    
    def __hash__(self) -> int:
        """Hash for dictionary keys."""
        if self._hash is None:
            # Compute hash based on essential properties
            shape_hash = hashlib.md5()
            
            # Add type and ID
            shape_hash.update(str(self._type).encode())
            shape_hash.update(str(self._id).encode())
            
            # Add core visual properties
            if self._fill:
                shape_hash.update(str(self._fill).encode())
            if self._stroke:
                shape_hash.update(str(self._stroke).encode())
                shape_hash.update(str(self._stroke_width).encode())
                
            shape_hash.update(str(self._opacity).encode())
            shape_hash.update(str(self._transform).encode())
            
            # Add class and attributes
            if self._class_name:
                shape_hash.update(str(self._class_name).encode())
                
            for k, v in sorted(self._attributes.items()):
                shape_hash.update(f"{k}:{v}".encode())
                
            # Store hash as integer
            self._hash = int(shape_hash.hexdigest(), 16) % (2**32)
            
        return self._hash
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self._type.name}(id={self._id})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"{self.__class__.__name__}(id={self._id}, type={self._type.name})"


class Rect(Shape):
    """
    Rectangle shape with optimized memory usage.
    """
    
    __slots__ = ('_x', '_y', '_width', '_height', '_rx', '_ry')
    
    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        rx: Optional[float] = None,
        ry: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize a rectangle.
        
        Args:
            x: X-coordinate of top-left corner
            y: Y-coordinate of top-left corner
            width: Width of rectangle
            height: Height of rectangle
            rx: X-axis corner radius
            ry: Y-axis corner radius
            **kwargs: Additional shape parameters
        """
        super().__init__(shape_type=ShapeType.RECT, **kwargs)
        
        self._x = x
        self._y = y
        self._width = max(0, width)
        self._height = max(0, height)
        self._rx = rx
        self._ry = ry
        
        # Clear cached hash
        self._hash = None
    
    @property
    def x(self) -> float:
        """Get x-coordinate."""
        return self._x
    
    @property
    def y(self) -> float:
        """Get y-coordinate."""
        return self._y
    
    @property
    def width(self) -> float:
        """Get width."""
        return self._width
    
    @property
    def height(self) -> float:
        """Get height."""
        return self._height
    
    @property
    def rx(self) -> Optional[float]:
        """Get x-axis corner radius."""
        return self._rx
    
    @property
    def ry(self) -> Optional[float]:
        """Get y-axis corner radius."""
        return self._ry
    
    def with_position(self, x: float, y: float) -> 'Rect':
        """
        Create a copy with a new position.
        
        Args:
            x: New x-coordinate
            y: New y-coordinate
            
        Returns:
            New rectangle with updated position
        """
        rect = self.copy()
        rect._x = x
        rect._y = y
        rect._hash = None
        rect._bounding_box_cache = None
        return rect
    
    def with_size(self, width: float, height: float) -> 'Rect':
        """
        Create a copy with a new size.
        
        Args:
            width: New width
            height: New height
            
        Returns:
            New rectangle with updated size
        """
        rect = self.copy()
        rect._width = max(0, width)
        rect._height = max(0, height)
        rect._hash = None
        rect._bounding_box_cache = None
        return rect
    
    def with_corner_radius(
        self,
        rx: Optional[float],
        ry: Optional[float] = None
    ) -> 'Rect':
        """
        Create a copy with new corner radii.
        
        Args:
            rx: New x-axis corner radius
            ry: New y-axis corner radius (defaults to rx)
            
        Returns:
            New rectangle with updated corner radii
        """
        rect = self.copy()
        rect._rx = rx
        
        if ry is None and rx is not None:
            rect._ry = rx
        else:
            rect._ry = ry
            
        rect._hash = None
        return rect
    
    def get_bounding_box(self) -> BoundingBox:
        """
        Get the bounding box of the rectangle.
        
        Returns:
            Bounding box (x, y, width, height)
        """
        # Use cached value if available
        if self._bounding_box_cache is not None:
            return self._bounding_box_cache
            
        # Apply transformation to the bounding box
        bbox = (self._x, self._y, self._width, self._height)
        
        if not self._transform.is_identity:
            bbox = self._transform.transform_bounding_box(bbox)
            
        self._bounding_box_cache = bbox
        return bbox
    
    def contains_point(self, point: Point) -> bool:
        """
        Check if the rectangle contains a point.
        
        Args:
            point: Point to check (x, y)
            
        Returns:
            True if the rectangle contains the point
        """
        # If transformed, convert to path and use path check
        if not self._transform.is_identity:
            # Create a path object for this rectangle
            from models.path import Path
            path_data = self.to_path_data()
            path = Path(path_data, transform=self._transform)
            return path.contains_point(point)
            
        # Simple check for untransformed rectangle
        x, y = point
        
        # Check if point is inside rectangle bounds
        if (
            x < self._x or
            y < self._y or
            x > self._x + self._width or
            y > self._y + self._height
        ):
            return False
            
        # If no corner radius, we're done
        if (self._rx is None or self._rx <= 0) and (self._ry is None or self._ry <= 0):
            return True
            
        # Handle corner radius (more complex check)
        rx = self._rx or 0
        ry = self._ry or rx
        
        # Check if point is in a corner region
        if x <= self._x + rx and y <= self._y + ry:
            # Top-left corner
            return ((x - (self._x + rx))**2 / rx**2 + 
                    (y - (self._y + ry))**2 / ry**2) <= 1
                    
        elif x >= self._x + self._width - rx and y <= self._y + ry:
            # Top-right corner
            return ((x - (self._x + self._width - rx))**2 / rx**2 + 
                    (y - (self._y + ry))**2 / ry**2) <= 1
                    
        elif x <= self._x + rx and y >= self._y + self._height - ry:
            # Bottom-left corner
            return ((x - (self._x + rx))**2 / rx**2 + 
                    (y - (self._y + self._height - ry))**2 / ry**2) <= 1
                    
        elif x >= self._x + self._width - rx and y >= self._y + self._height - ry:
            # Bottom-right corner
            return ((x - (self._x + self._width - rx))**2 / rx**2 + 
                    (y - (self._y + self._height - ry))**2 / ry**2) <= 1
                    
        # Point is in the non-corner region
        return True
    
    def to_path_data(self) -> PathData:
        """
        Convert rectangle to SVG path data.
        
        Returns:
            SVG path data string
        """
        # Simple rectangle (no corner radius)
        if (self._rx is None or self._rx <= 0) and (self._ry is None or self._ry <= 0):
            return (
                f"M{self._x:.{PATH_PRECISION}g},{self._y:.{PATH_PRECISION}g} "
                f"h{self._width:.{PATH_PRECISION}g} "
                f"v{self._height:.{PATH_PRECISION}g} "
                f"h{-self._width:.{PATH_PRECISION}g} "
                f"Z"
            )
            
        # Rectangle with corner radius
        rx = self._rx or 0
        ry = self._ry or rx
        
        # Ensure radius is not too large
        rx = min(rx, self._width / 2)
        ry = min(ry, self._height / 2)
        
        # Create path with rounded corners
        return (
            f"M{(self._x + rx):.{PATH_PRECISION}g},{self._y:.{PATH_PRECISION}g} "
            f"h{(self._width - 2*rx):.{PATH_PRECISION}g} "
            f"a{rx:.{PATH_PRECISION}g},{ry:.{PATH_PRECISION}g} 0 0 1 {rx:.{PATH_PRECISION}g},{ry:.{PATH_PRECISION}g} "
            f"v{(self._height - 2*ry):.{PATH_PRECISION}g} "
            f"a{rx:.{PATH_PRECISION}g},{ry:.{PATH_PRECISION}g} 0 0 1 {-rx:.{PATH_PRECISION}g},{ry:.{PATH_PRECISION}g} "
            f"h{-(self._width - 2*rx):.{PATH_PRECISION}g} "
            f"a{rx:.{PATH_PRECISION}g},{ry:.{PATH_PRECISION}g} 0 0 1 {-rx:.{PATH_PRECISION}g},{-ry:.{PATH_PRECISION}g} "
            f"v{-(self._height - 2*ry):.{PATH_PRECISION}g} "
            f"a{rx:.{PATH_PRECISION}g},{ry:.{PATH_PRECISION}g} 0 0 1 {rx:.{PATH_PRECISION}g},{-ry:.{PATH_PRECISION}g} "
            f"Z"
        )
    
    def to_svg_element(self) -> ET.Element:
        """
        Convert rectangle to SVG element.
        
        Returns:
            XML element representing the rectangle
        """
        # Create rect element
        rect = ET.Element('rect')
        rect.set('x', f"{self._x:.{PATH_PRECISION}g}")
        rect.set('y', f"{self._y:.{PATH_PRECISION}g}")
        rect.set('width', f"{self._width:.{PATH_PRECISION}g}")
        rect.set('height', f"{self._height:.{PATH_PRECISION}g}")
        
        # Add corner radius if specified
        if self._rx is not None and self._rx > 0:
            rect.set('rx', f"{self._rx:.{PATH_PRECISION}g}")
        if self._ry is not None and self._ry > 0:
            rect.set('ry', f"{self._ry:.{PATH_PRECISION}g}")
            
        # Add common attributes
        self._add_common_attributes(rect)
        
        return rect
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert rectangle to dictionary representation.
        
        Returns:
            Dictionary with rectangle data
        """
        data = super().to_dict()
        data.update({
            'x': self._x,
            'y': self._y,
            'width': self._width,
            'height': self._height,
            'rx': self._rx,
            'ry': self._ry
        })
        return data
    
    def copy(self) -> 'Rect':
        """
        Create a deep copy of the rectangle.
        
        Returns:
            Copied rectangle
        """
        return Rect(
            x=self._x,
            y=self._y,
            width=self._width,
            height=self._height,
            rx=self._rx,
            ry=self._ry,
            shape_id=self._id,
            fill=self._fill,
            stroke=self._stroke,
            stroke_width=self._stroke_width,
            opacity=self._opacity,
            transform=self._transform,
            class_name=self._class_name,
            attributes=self._attributes.copy(),
            data=self._data.copy()
        )
    
    def __eq__(self, other: object) -> bool:
        """Check if rectangles are equal."""
        if not isinstance(other, Rect):
            return False
            
        return (
            super().__eq__(other) and
            self._x == other._x and
            self._y == other._y and
            self._width == other._width and
            self._height == other._height and
            self._rx == other._rx and
            self._ry == other._ry
        )
    
    def __hash__(self) -> int:
        """Hash for dictionary keys."""
        if self._hash is None:
            # Start with base hash
            hash_value = super().__hash__()
            
            # Add rectangle-specific properties
            shape_hash = hashlib.md5()
            shape_hash.update(f"{hash_value}".encode())
            shape_hash.update(f"{self._x}:{self._y}:{self._width}:{self._height}".encode())
            
            if self._rx is not None:
                shape_hash.update(f"{self._rx}".encode())
            if self._ry is not None:
                shape_hash.update(f"{self._ry}".encode())
                
            self._hash = int(shape_hash.hexdigest(), 16) % (2**32)
            
        return self._hash


class Circle(Shape):
    """
    Circle shape with optimized memory usage.
    """
    
    __slots__ = ('_cx', '_cy', '_radius')
    
    def __init__(
        self,
        cx: float,
        cy: float,
        radius: float,
        **kwargs
    ):
        """
        Initialize a circle.
        
        Args:
            cx: X-coordinate of center
            cy: Y-coordinate of center
            radius: Radius of circle
            **kwargs: Additional shape parameters
        """
        super().__init__(shape_type=ShapeType.CIRCLE, **kwargs)
        
        self._cx = cx
        self._cy = cy
        self._radius = max(0, radius)
        
        # Clear cached hash
        self._hash = None
    
    @property
    def cx(self) -> float:
        """Get center x-coordinate."""
        return self._cx
    
    @property
    def cy(self) -> float:
        """Get center y-coordinate."""
        return self._cy
    
    @property
    def radius(self) -> float:
        """Get radius."""
        return self._radius
    
    def with_center(self, cx: float, cy: float) -> 'Circle':
        """
        Create a copy with a new center position.
        
        Args:
            cx: New center x-coordinate
            cy: New center y-coordinate
            
        Returns:
            New circle with updated center
        """
        circle = self.copy()
        circle._cx = cx
        circle._cy = cy
        circle._hash = None
        circle._bounding_box_cache = None
        return circle
    
    def with_radius(self, radius: float) -> 'Circle':
        """
        Create a copy with a new radius.
        
        Args:
            radius: New radius
            
        Returns:
            New circle with updated radius
        """
        circle = self.copy()
        circle._radius = max(0, radius)
        circle._hash = None
        circle._bounding_box_cache = None
        return circle
    
    def get_bounding_box(self) -> BoundingBox:
        """
        Get the bounding box of the circle.
        
        Returns:
            Bounding box (x, y, width, height)
        """
        # Use cached value if available
        if self._bounding_box_cache is not None:
            return self._bounding_box_cache
            
        # Compute bounding box
        bbox = (
            self._cx - self._radius,
            self._cy - self._radius,
            2 * self._radius,
            2 * self._radius
        )
        
        # Apply transformation if needed
        if not self._transform.is_identity:
            bbox = self._transform.transform_bounding_box(bbox)
            
        self._bounding_box_cache = bbox
        return bbox
    
    def contains_point(self, point: Point) -> bool:
        """
        Check if the circle contains a point.
        
        Args:
            point: Point to check (x, y)
            
        Returns:
            True if the circle contains the point
        """
        # If transformed, need more complex handling
        if not self._transform.is_identity:
            # Inverse transform the point to the circle's original space
            inverse = self._transform.inverse
            if inverse is None:
                # Non-invertible transform - fall back to path check
                from models.path import Path
                path_data = self.to_path_data()
                path = Path(path_data, transform=self._transform)
                return path.contains_point(point)
                
            # Transform point to circle's space
            x, y = inverse.transform_point(point)
        else:
            x, y = point
            
        # Check if point is within circle
        return ((x - self._cx)**2 + (y - self._cy)**2) <= self._radius**2
    
    def to_path_data(self) -> PathData:
        """
        Convert circle to SVG path data.
        
        Returns:
            SVG path data string
        """
        # Create path using arcs
        r = self._radius
        return (
            f"M{(self._cx - r):.{PATH_PRECISION}g},{self._cy:.{PATH_PRECISION}g} "
            f"a{r:.{PATH_PRECISION}g},{r:.{PATH_PRECISION}g} 0 1 0 {(2*r):.{PATH_PRECISION}g},0 "
            f"a{r:.{PATH_PRECISION}g},{r:.{PATH_PRECISION}g} 0 1 0 {(-2*r):.{PATH_PRECISION}g},0 "
            f"Z"
        )
    
    def to_svg_element(self) -> ET.Element:
        """
        Convert circle to SVG element.
        
        Returns:
            XML element representing the circle
        """
        # Create circle element
        circle = ET.Element('circle')
        circle.set('cx', f"{self._cx:.{PATH_PRECISION}g}")
        circle.set('cy', f"{self._cy:.{PATH_PRECISION}g}")
        circle.set('r', f"{self._radius:.{PATH_PRECISION}g}")
        
        # Add common attributes
        self._add_common_attributes(circle)
        
        return circle
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert circle to dictionary representation.
        
        Returns:
            Dictionary with circle data
        """
        data = super().to_dict()
        data.update({
            'cx': self._cx,
            'cy': self._cy,
            'radius': self._radius
        })
        return data
    
    def copy(self) -> 'Circle':
        """
        Create a deep copy of the circle.
        
        Returns:
            Copied circle
        """
        return Circle(
            cx=self._cx,
            cy=self._cy,
            radius=self._radius,
            shape_id=self._id,
            fill=self._fill,
            stroke=self._stroke,
            stroke_width=self._stroke_width,
            opacity=self._opacity,
            transform=self._transform,
            class_name=self._class_name,
            attributes=self._attributes.copy(),
            data=self._data.copy()
        )
    
    def __eq__(self, other: object) -> bool:
        """Check if circles are equal."""
        if not isinstance(other, Circle):
            return False
            
        return (
            super().__eq__(other) and
            self._cx == other._cx and
            self._cy == other._cy and
            self._radius == other._radius
        )
    
    def __hash__(self) -> int:
        """Hash for dictionary keys."""
        if self._hash is None:
            # Start with base hash
            hash_value = super().__hash__()
            
            # Add circle-specific properties
            shape_hash = hashlib.md5()
            shape_hash.update(f"{hash_value}".encode())
            shape_hash.update(f"{self._cx}:{self._cy}:{self._radius}".encode())
            
            self._hash = int(shape_hash.hexdigest(), 16) % (2**32)
            
        return self._hash


class Ellipse(Shape):
    """
    Ellipse shape with optimized memory usage.
    """
    
    __slots__ = ('_cx', '_cy', '_rx', '_ry')
    
    def __init__(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        **kwargs
    ):
        """
        Initialize an ellipse.
        
        Args:
            cx: X-coordinate of center
            cy: Y-coordinate of center
            rx: X-axis radius
            ry: Y-axis radius
            **kwargs: Additional shape parameters
        """
        super().__init__(shape_type=ShapeType.ELLIPSE, **kwargs)
        
        self._cx = cx
        self._cy = cy
        self._rx = max(0, rx)
        self._ry = max(0, ry)
        
        # Clear cached hash
        self._hash = None
    
    @property
    def cx(self) -> float:
        """Get center x-coordinate."""
        return self._cx
    
    @property
    def cy(self) -> float:
        """Get center y-coordinate."""
        return self._cy
    
    @property
    def rx(self) -> float:
        """Get x-axis radius."""
        return self._rx
    
    @property
    def ry(self) -> float:
        """Get y-axis radius."""
        return self._ry
    
    def with_center(self, cx: float, cy: float) -> 'Ellipse':
        """
        Create a copy with a new center position.
        
        Args:
            cx: New center x-coordinate
            cy: New center y-coordinate
            
        Returns:
            New ellipse with updated center
        """
        ellipse = self.copy()
        ellipse._cx = cx
        ellipse._cy = cy
        ellipse._hash = None
        ellipse._bounding_box_cache = None
        return ellipse
    
    def with_radii(self, rx: float, ry: float) -> 'Ellipse':
        """
        Create a copy with new radii.
        
        Args:
            rx: New x-axis radius
            ry: New y-axis radius
            
        Returns:
            New ellipse with updated radii
        """
        ellipse = self.copy()
        ellipse._rx = max(0, rx)
        ellipse._ry = max(0, ry)
        ellipse._hash = None
        ellipse._bounding_box_cache = None
        return ellipse
    
    def get_bounding_box(self) -> BoundingBox:
        """
        Get the bounding box of the ellipse.
        
        Returns:
            Bounding box (x, y, width, height)
        """
        # Use cached value if available
        if self._bounding_box_cache is not None:
            return self._bounding_box_cache
            
        # Compute bounding box
        bbox = (
            self._cx - self._rx,
            self._cy - self._ry,
            2 * self._rx,
            2 * self._ry
        )
        
        # Apply transformation if needed
        if not self._transform.is_identity:
            bbox = self._transform.transform_bounding_box(bbox)
            
        self._bounding_box_cache = bbox
        return bbox
    
    def contains_point(self, point: Point) -> bool:
        """
        Check if the ellipse contains a point.
        
        Args:
            point: Point to check (x, y)
            
        Returns:
            True if the ellipse contains the point
        """
        # If transformed, need more complex handling
        if not self._transform.is_identity:
            # Inverse transform the point to the ellipse's original space
            inverse = self._transform.inverse
            if inverse is None:
                # Non-invertible transform - fall back to path check
                from models.path import Path
                path_data = self.to_path_data()
                path = Path(path_data, transform=self._transform)
                return path.contains_point(point)
                
            # Transform point to ellipse's space
            x, y = inverse.transform_point(point)
        else:
            x, y = point
            
        # Check if point is within ellipse
        if self._rx <= 0 or self._ry <= 0:
            return False
            
        return (((x - self._cx) / self._rx)**2 + 
                ((y - self._cy) / self._ry)**2) <= 1
    
    def to_path_data(self) -> PathData:
        """
        Convert ellipse to SVG path data.
        
        Returns:
            SVG path data string
        """
        # Create path using arcs
        rx, ry = self._rx, self._ry
        return (
            f"M{(self._cx - rx):.{PATH_PRECISION}g},{self._cy:.{PATH_PRECISION}g} "
            f"a{rx:.{PATH_PRECISION}g},{ry:.{PATH_PRECISION}g} 0 1 0 {(2*rx):.{PATH_PRECISION}g},0 "
            f"a{rx:.{PATH_PRECISION}g},{ry:.{PATH_PRECISION}g} 0 1 0 {(-2*rx):.{PATH_PRECISION}g},0 "
            f"Z"
        )
    
    def to_svg_element(self) -> ET.Element:
        """
        Convert ellipse to SVG element.
        
        Returns:
            XML element representing the ellipse
        """
        # Create ellipse element
        ellipse = ET.Element('ellipse')
        ellipse.set('cx', f"{self._cx:.{PATH_PRECISION}g}")
        ellipse.set('cy', f"{self._cy:.{PATH_PRECISION}g}")
        ellipse.set('rx', f"{self._rx:.{PATH_PRECISION}g}")
        ellipse.set('ry', f"{self._ry:.{PATH_PRECISION}g}")
        
        # Add common attributes
        self._add_common_attributes(ellipse)
        
        return ellipse
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ellipse to dictionary representation.
        
        Returns:
            Dictionary with ellipse data
        """
        data = super().to_dict()
        data.update({
            'cx': self._cx,
            'cy': self._cy,
            'rx': self._rx,
            'ry': self._ry
        })
        return data
    
    def copy(self) -> 'Ellipse':
        """
        Create a deep copy of the ellipse.
        
        Returns:
            Copied ellipse
        """
        return Ellipse(
            cx=self._cx,
            cy=self._cy,
            rx=self._rx,
            ry=self._ry,
            shape_id=self._id,
            fill=self._fill,
            stroke=self._stroke,
            stroke_width=self._stroke_width,
            opacity=self._opacity,
            transform=self._transform,
            class_name=self._class_name,
            attributes=self._attributes.copy(),
            data=self._data.copy()
        )
    
    def __eq__(self, other: object) -> bool:
        """Check if ellipses are equal."""
        if not isinstance(other, Ellipse):
            return False
            
        return (
            super().__eq__(other) and
            self._cx == other._cx and
            self._cy == other._cy and
            self._rx == other._rx and
            self._ry == other._ry
        )
    
    def __hash__(self) -> int:
        """Hash for dictionary keys."""
        if self._hash is None:
            # Start with base hash
            hash_value = super().__hash__()
            
            # Add ellipse-specific properties
            shape_hash = hashlib.md5()
            shape_hash.update(f"{hash_value}".encode())
            shape_hash.update(f"{self._cx}:{self._cy}:{self._rx}:{self._ry}".encode())
            
            self._hash = int(shape_hash.hexdigest(), 16) % (2**32)
            
        return self._hash


class Line(Shape):
    """
    Line shape with optimized memory usage.
    """
    
    __slots__ = ('_x1', '_y1', '_x2', '_y2')
    
    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        **kwargs
    ):
        """
        Initialize a line.
        
        Args:
            x1: X-coordinate of start point
            y1: Y-coordinate of start point
            x2: X-coordinate of end point
            y2: Y-coordinate of end point
            **kwargs: Additional shape parameters
        """
        # Ensure shape has a stroke
        if 'stroke' not in kwargs:
            from models.color import Color
            kwargs['stroke'] = Color.from_rgb(0, 0, 0)
            
        # Lines have no fill by default
        if 'fill' not in kwargs:
            kwargs['fill'] = None
            
        super().__init__(shape_type=ShapeType.LINE, **kwargs)
        
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2
        
        # Clear cached hash
        self._hash = None
    
    @property
    def x1(self) -> float:
        """Get x-coordinate of start point."""
        return self._x1
    
    @property
    def y1(self) -> float:
        """Get y-coordinate of start point."""
        return self._y1
    
    @property
    def x2(self) -> float:
        """Get x-coordinate of end point."""
        return self._x2
    
    @property
    def y2(self) -> float:
        """Get y-coordinate of end point."""
        return self._y2
    
    @property
    def start_point(self) -> Point:
        """Get start point."""
        return (self._x1, self._y1)
    
    @property
    def end_point(self) -> Point:
        """Get end point."""
        return (self._x2, self._y2)
    
    @property
    def length(self) -> float:
        """Get length of the line."""
        return math.sqrt((self._x2 - self._x1)**2 + (self._y2 - self._y1)**2)
    
    @property
    def angle(self) -> float:
        """Get angle of the line in degrees."""
        return math.degrees(math.atan2(self._y2 - self._y1, self._x2 - self._x1))
    
    def with_points(self, x1: float, y1: float, x2: float, y2: float) -> 'Line':
        """
        Create a copy with new points.
        
        Args:
            x1: New x-coordinate of start point
            y1: New y-coordinate of start point
            x2: New x-coordinate of end point
            y2: New y-coordinate of end point
            
        Returns:
            New line with updated points
        """
        line = self.copy()
        line._x1 = x1
        line._y1 = y1
        line._x2 = x2
        line._y2 = y2
        line._hash = None
        line._bounding_box_cache = None
        return line
    
    def with_start_point(self, x1: float, y1: float) -> 'Line':
        """
        Create a copy with a new start point.
        
        Args:
            x1: New x-coordinate of start point
            y1: New y-coordinate of start point
            
        Returns:
            New line with updated start point
        """
        line = self.copy()
        line._x1 = x1
        line._y1 = y1
        line._hash = None
        line._bounding_box_cache = None
        return line
    
    def with_end_point(self, x2: float, y2: float) -> 'Line':
        """
        Create a copy with a new end point.
        
        Args:
            x2: New x-coordinate of end point
            y2: New y-coordinate of end point
            
        Returns:
            New line with updated end point
        """
        line = self.copy()
        line._x2 = x2
        line._y2 = y2
        line._hash = None
        line._bounding_box_cache = None
        return line
    
    def get_bounding_box(self) -> BoundingBox:
        """
        Get the bounding box of the line.
        
        Returns:
            Bounding box (x, y, width, height)
        """
        # Use cached value if available
        if self._bounding_box_cache is not None:
            return self._bounding_box_cache
            
        # Compute bounding box
        x_min = min(self._x1, self._x2)
        y_min = min(self._y1, self._y2)
        x_max = max(self._x1, self._x2)
        y_max = max(self._y1, self._y2)
        
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Apply transformation if needed
        if not self._transform.is_identity:
            bbox = self._transform.transform_bounding_box(bbox)
            
        self._bounding_box_cache = bbox
        return bbox
    
    def contains_point(self, point: Point) -> bool:
        """
        Check if the line contains a point.
        
        Args:
            point: Point to check (x, y)
            
        Returns:
            True if the point is on the line
        """
        # If transformed, transform the point to line's space
        if not self._transform.is_identity:
            inverse = self._transform.inverse
            if inverse is None:
                # Non-invertible transform - fall back to path check
                from models.path import Path
                path_data = self.to_path_data()
                path = Path(path_data, transform=self._transform)
                return path.contains_point(point)
                
            # Transform point to line's space
            x, y = inverse.transform_point(point)
        else:
            x, y = point
            
        # Check if point is on line (with some tolerance)
        # Line equation: (y - y1) / (x - x1) = (y2 - y1) / (x2 - x1)
        if abs(self._x2 - self._x1) < 1e-6:
            # Vertical line
            return (abs(x - self._x1) < 1e-6 and 
                    y >= min(self._y1, self._y2) - 1e-6 and
                    y <= max(self._y1, self._y2) + 1e-6)
                    
        if abs(self._y2 - self._y1) < 1e-6:
            # Horizontal line
            return (abs(y - self._y1) < 1e-6 and 
                    x >= min(self._x1, self._x2) - 1e-6 and
                    x <= max(self._x1, self._x2) + 1e-6)
                    
        # General case - check if point lies on line segment
        # Using parametric form: point = start + t * (end - start), 0 <= t <= 1
        dx = self._x2 - self._x1
        dy = self._y2 - self._y1
        
        if abs(dx) > abs(dy):
            # Use x for parameter
            t = (x - self._x1) / dx
        else:
            # Use y for parameter
            t = (y - self._y1) / dy
            
        # Check if t is within range and point is close to line
        if t < 0 or t > 1:
            return False
            
        # Calculate point on line at parameter t
        line_x = self._x1 + t * dx
        line_y = self._y1 + t * dy
        
        # Check distance from point to line point
        distance = math.sqrt((x - line_x)**2 + (y - line_y)**2)
        
        # Use stroke width as tolerance, default to small value
        tolerance = max(self._stroke_width / 2, 1e-6)
        return distance <= tolerance
    
    def to_path_data(self) -> PathData:
        """
        Convert line to SVG path data.
        
        Returns:
            SVG path data string
        """
        return (
            f"M{self._x1:.{PATH_PRECISION}g},{self._y1:.{PATH_PRECISION}g} "
            f"L{self._x2:.{PATH_PRECISION}g},{self._y2:.{PATH_PRECISION}g}"
        )
    
    def to_svg_element(self) -> ET.Element:
        """
        Convert line to SVG element.
        
        Returns:
            XML element representing the line
        """
        # Create line element
        line = ET.Element('line')
        line.set('x1', f"{self._x1:.{PATH_PRECISION}g}")
        line.set('y1', f"{self._y1:.{PATH_PRECISION}g}")
        line.set('x2', f"{self._x2:.{PATH_PRECISION}g}")
        line.set('y2', f"{self._y2:.{PATH_PRECISION}g}")
        
        # Add common attributes
        self._add_common_attributes(line)
        
        return line
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert line to dictionary representation.
        
        Returns:
            Dictionary with line data
        """
        data = super().to_dict()
        data.update({
            'x1': self._x1,
            'y1': self._y1,
            'x2': self._x2,
            'y2': self._y2,
            'length': self.length,
            'angle': self.angle
        })
        return data
    
    def copy(self) -> 'Line':
        """
        Create a deep copy of the line.
        
        Returns:
            Copied line
        """
        return Line(
            x1=self._x1,
            y1=self._y1,
            x2=self._x2,
            y2=self._y2,
            shape_id=self._id,
            fill=self._fill,
            stroke=self._stroke,
            stroke_width=self._stroke_width,
            opacity=self._opacity,
            transform=self._transform,
            class_name=self._class_name,
            attributes=self._attributes.copy(),
            data=self._data.copy()
        )
    
    def __eq__(self, other: object) -> bool:
        """Check if lines are equal."""
        if not isinstance(other, Line):
            return False
            
        return (
            super().__eq__(other) and
            self._x1 == other._x1 and
            self._y1 == other._y1 and
            self._x2 == other._x2 and
            self._y2 == other._y2
        )
    
    def __hash__(self) -> int:
        """Hash for dictionary keys."""
        if self._hash is None:
            # Start with base hash
            hash_value = super().__hash__()
            
            # Add line-specific properties
            shape_hash = hashlib.md5()
            shape_hash.update(f"{hash_value}".encode())
            shape_hash.update(f"{self._x1}:{self._y1}:{self._x2}:{self._y2}".encode())
            
            self._hash = int(shape_hash.hexdigest(), 16) % (2**32)
            
        return self._hash


class Group(Shape):
    """
    Group of shapes with optimized memory usage.
    """
    
    __slots__ = ('_shapes',)
    
    def __init__(
        self,
        shapes: Optional[List[Shape]] = None,
        **kwargs
    ):
        """
        Initialize a group of shapes.
        
        Args:
            shapes: List of shapes in the group
            **kwargs: Additional shape parameters
        """
        super().__init__(shape_type=ShapeType.GROUP, **kwargs)
        
        self._shapes = tuple(shapes) if shapes else tuple()
    
    @property
    def shapes(self) -> List[Shape]:
        """Get list of shapes in the group."""
        return list(self._shapes)
    
    @property
    def count(self) -> int:
        """Get number of shapes in the group."""
        return len(self._shapes)
    
    @property
    def is_empty(self) -> bool:
        """Check if group is empty."""
        return len(self._shapes) == 0
    
    def with_shapes(self, shapes: List[Shape]) -> 'Group':
        """
        Create a copy with new shapes.
        
        Args:
            shapes: New list of shapes
            
        Returns:
            New group with updated shapes
        """
        group = self.copy()
        group._shapes = tuple(shapes)
        group._hash = None
        group._bounding_box_cache = None
        return group
    
    def add_shape(self, shape: Shape) -> 'Group':
        """
        Create a copy with an additional shape.
        
        Args:
            shape: Shape to add
            
        Returns:
            New group with added shape
        """
        shapes = list(self._shapes)
        shapes.append(shape)
        
        group = self.copy()
        group._shapes = tuple(shapes)
        group._hash = None
        group._bounding_box_cache = None
        
        return group
    
    def add_shapes(self, shapes: List[Shape]) -> 'Group':
        """
        Create a copy with additional shapes.
        
        Args:
            shapes: Shapes to add
            
        Returns:
            New group with added shapes
        """
        if not shapes:
            return self
            
        all_shapes = list(self._shapes)
        all_shapes.extend(shapes)
        
        group = self.copy()
        group._shapes = tuple(all_shapes)
        group._hash = None
        group._bounding_box_cache = None
        
        return group
    
    def remove_shape(self, shape_id: str) -> 'Group':
        """
        Create a copy with a shape removed.
        
        Args:
            shape_id: ID of shape to remove
            
        Returns:
            New group with shape removed
        """
        if not self._shapes:
            return self
            
        shapes = [s for s in self._shapes if s.id != shape_id]
        
        if len(shapes) == len(self._shapes):
            return self
            
        group = self.copy()
        group._shapes = tuple(shapes)
        group._hash = None
        group._bounding_box_cache = None
        
        return group
    
    def get_shape_by_id(self, shape_id: str) -> Optional[Shape]:
        """
        Find a shape by ID.
        
        Args:
            shape_id: Shape ID to find
            
        Returns:
            Shape with matching ID or None if not found
        """
        for shape in self._shapes:
            if shape.id == shape_id:
                return shape
                
            # Recursively search in nested groups
            if isinstance(shape, Group):
                nested = shape.get_shape_by_id(shape_id)
                if nested is not None:
                    return nested
                    
        return None
    
    def get_bounding_box(self) -> BoundingBox:
        """
        Get the bounding box of the group.
        
        Returns:
            Bounding box (x, y, width, height)
        """
        # Use cached value if available
        if self._bounding_box_cache is not None:
            return self._bounding_box_cache
            
        # Handle empty group
        if not self._shapes:
            if self._transform.is_identity:
                return (0, 0, 0, 0)
            else:
                return self._transform.transform_bounding_box((0, 0, 0, 0))
                
        # Compute bounding box of all shapes
        bbox_list = [shape.get_bounding_box() for shape in self._shapes]
        
        # Find extremes
        x_min = min(bbox[0] for bbox in bbox_list)
        y_min = min(bbox[1] for bbox in bbox_list)
        x_max = max(bbox[0] + bbox[2] for bbox in bbox_list)
        y_max = max(bbox[1] + bbox[3] for bbox in bbox_list)
        
        # Create group bounding box
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Apply group transform if needed
        if not self._transform.is_identity:
            bbox = self._transform.transform_bounding_box(bbox)
            
        self._bounding_box_cache = bbox
        return bbox
    
    def contains_point(self, point: Point) -> bool:
        """
        Check if any shape in the group contains a point.
        
        Args:
            point: Point to check (x, y)
            
        Returns:
            True if any shape contains the point
        """
        # If transformed, transform the point to group's space
        if not self._transform.is_identity:
            inverse = self._transform.inverse
            if inverse is None:
                # Non-invertible transform
                return False
                
            # Transform point to group's space
            transformed_point = inverse.transform_point(point)
        else:
            transformed_point = point
            
        # Check if any shape contains the point
        return any(shape.contains_point(transformed_point) for shape in self._shapes)
    
    def to_path_data(self) -> PathData:
        """
        Convert group to SVG path data.
        
        Returns:
            SVG path data string combining all shapes
        """
        # Combine path data of all shapes
        path_data = []
        
        for shape in self._shapes:
            try:
                path_data.append(shape.to_path_data())
            except NotImplementedError:
                # Skip shapes that don't support path data
                pass
                
        return " ".join(path_data)
    
    def to_svg_element(self) -> ET.Element:
        """
        Convert group to SVG element.
        
        Returns:
            XML element representing the group
        """
        # Create group element
        group = ET.Element('g')
        
        # Add common attributes
        self._add_common_attributes(group)
        
        # Add child shapes
        for shape in self._shapes:
            shape_element = shape.to_svg_element()
            group.append(shape_element)
            
        return group
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert group to dictionary representation.
        
        Returns:
            Dictionary with group data
        """
        data = super().to_dict()
        data.update({
            'shapes': [shape.to_dict() for shape in self._shapes],
            'count': len(self._shapes)
        })
        return data
    
    def copy(self) -> 'Group':
        """
        Create a deep copy of the group.
        
        Returns:
            Copied group
        """
        return Group(
            shapes=self._shapes,  # Shapes are immutable
            shape_id=self._id,
            fill=self._fill,
            stroke=self._stroke,
            stroke_width=self._stroke_width,
            opacity=self._opacity,
            transform=self._transform,
            class_name=self._class_name,
            attributes=self._attributes.copy(),
            data=self._data.copy()
        )
    
    def __eq__(self, other: object) -> bool:
        """Check if groups are equal."""
        if not isinstance(other, Group):
            return False
            
        if len(self._shapes) != len(other._shapes):
            return False
            
        return (
            super().__eq__(other) and
            all(s1 == s2 for s1, s2 in zip(self._shapes, other._shapes))
        )
    
    def __hash__(self) -> int:
        """Hash for dictionary keys."""
        if self._hash is None:
            # Start with base hash
            hash_value = super().__hash__()
            
            # Add shape hashes
            shape_hash = hashlib.md5()
            shape_hash.update(f"{hash_value}".encode())
            
            for shape in self._shapes:
                shape_hash.update(f"{hash(shape)}".encode())
                
            self._hash = int(shape_hash.hexdigest(), 16) % (2**32)
            
        return self._hash
    
    def __len__(self) -> int:
        """Get number of shapes in the group."""
        return len(self._shapes)
    
    def __iter__(self):
        """Iterate over shapes in the group."""
        return iter(self._shapes)
    
    def __getitem__(self, index: int) -> Shape:
        """Get shape at index."""
        return self._shapes[index]


# Utility functions for shape operations

def create_shape_from_dict(shape_dict: Dict[str, Any]) -> Shape:
    """
    Create a shape from a dictionary representation.
    
    Args:
        shape_dict: Dictionary with shape data
        
    Returns:
        Created shape
        
    Raises:
        ShapeError: If shape type is unknown or data is invalid
    """
    shape_type = shape_dict.get('type')
    if not shape_type:
        raise ShapeError("Missing shape type in dictionary")
        
    # Common attributes
    common_args = {
        'shape_id': shape_dict.get('id'),
        'fill': shape_dict.get('fill'),
        'stroke': shape_dict.get('stroke'),
        'stroke_width': shape_dict.get('stroke_width', DEFAULT_STROKE_WIDTH),
        'opacity': shape_dict.get('opacity', DEFAULT_OPACITY),
        'transform': shape_dict.get('transform'),
        'class_name': shape_dict.get('class_name'),
        'attributes': shape_dict.get('attributes', {}),
        'data': shape_dict.get('data', {})
    }
    
    try:
        if shape_type == 'RECT':
            return Rect(
                x=shape_dict['x'],
                y=shape_dict['y'],
                width=shape_dict['width'],
                height=shape_dict['height'],
                rx=shape_dict.get('rx'),
                ry=shape_dict.get('ry'),
                **common_args
            )
            
        elif shape_type == 'CIRCLE':
            return Circle(
                cx=shape_dict['cx'],
                cy=shape_dict['cy'],
                radius=shape_dict['radius'],
                **common_args
            )
            
        elif shape_type == 'ELLIPSE':
            return Ellipse(
                cx=shape_dict['cx'],
                cy=shape_dict['cy'],
                rx=shape_dict['rx'],
                ry=shape_dict['ry'],
                **common_args
            )
            
        elif shape_type == 'LINE':
            return Line(
                x1=shape_dict['x1'],
                y1=shape_dict['y1'],
                x2=shape_dict['x2'],
                y2=shape_dict['y2'],
                **common_args
            )
            
        elif shape_type == 'GROUP':
            # Recursively create child shapes
            child_shapes = []
            for child_dict in shape_dict.get('shapes', []):
                child_shapes.append(create_shape_from_dict(child_dict))
                
            return Group(
                shapes=child_shapes,
                **common_args
            )
            
        elif shape_type == 'PATH':
            from models.path import Path
            return Path(
                data=shape_dict['data'],
                **common_args
            )
            
        elif shape_type == 'POLYGON' or shape_type == 'POLYLINE':
            from models.path import Polygon, Polyline
            points = shape_dict.get('points', [])
            
            if shape_type == 'POLYGON':
                return Polygon(
                    points=points,
                    **common_args
                )
            else:
                return Polyline(
                    points=points,
                    **common_args
                )
                
        elif shape_type == 'TEXT':
            from models.text import Text
            return Text(
                x=shape_dict['x'],
                y=shape_dict['y'],
                text=shape_dict['text'],
                font_family=shape_dict.get('font_family'),
                font_size=shape_dict.get('font_size'),
                font_weight=shape_dict.get('font_weight'),
                text_anchor=shape_dict.get('text_anchor'),
                **common_args
            )
            
        else:
            raise ShapeError(f"Unknown shape type: {shape_type}")
            
    except KeyError as e:
        raise ShapeError(f"Missing required field for {shape_type}: {e}")
    except Exception as e:
        raise ShapeError(f"Error creating {shape_type}: {e}")


def create_shape_from_svg(svg_element: ET.Element) -> Optional[Shape]:
    """
    Create a shape from an SVG element.
    
    Args:
        svg_element: SVG element to convert
        
    Returns:
        Created shape or None if element type is not supported
        
    Raises:
        ShapeError: If element data is invalid
    """
    # Extract tag name without namespace
    tag = svg_element.tag
    if '}' in tag:
        tag = tag.split('}', 1)[1]
        
    # Common attributes
    common_args = {
        'shape_id': svg_element.get('id'),
        'fill': svg_element.get('fill'),
        'stroke': svg_element.get('stroke'),
        'stroke_width': float(svg_element.get('stroke-width', DEFAULT_STROKE_WIDTH)),
        'opacity': float(svg_element.get('opacity', DEFAULT_OPACITY)),
        'transform': svg_element.get('transform'),
        'class_name': svg_element.get('class')
    }
    
    # Extract additional attributes
    attributes = {}
    for name, value in svg_element.attrib.items():
        if name not in ('id', 'fill', 'stroke', 'stroke-width', 'opacity', 
                       'transform', 'class', 'x', 'y', 'width', 'height',
                       'cx', 'cy', 'r', 'rx', 'ry', 'x1', 'y1', 'x2', 'y2',
                       'd', 'points'):
            attributes[name] = value
            
    if attributes:
        common_args['attributes'] = attributes
        
    try:
        if tag == 'rect':
            return Rect(
                x=float(svg_element.get('x', 0)),
                y=float(svg_element.get('y', 0)),
                width=float(svg_element.get('width', 0)),
                height=float(svg_element.get('height', 0)),
                rx=float(svg_element.get('rx')) if 'rx' in svg_element.attrib else None,
                ry=float(svg_element.get('ry')) if 'ry' in svg_element.attrib else None,
                **common_args
            )
            
        elif tag == 'circle':
            return Circle(
                cx=float(svg_element.get('cx', 0)),
                cy=float(svg_element.get('cy', 0)),
                radius=float(svg_element.get('r', 0)),
                **common_args
            )
            
        elif tag == 'ellipse':
            return Ellipse(
                cx=float(svg_element.get('cx', 0)),
                cy=float(svg_element.get('cy', 0)),
                rx=float(svg_element.get('rx', 0)),
                ry=float(svg_element.get('ry', 0)),
                **common_args
            )
            
        elif tag == 'line':
            return Line(
                x1=float(svg_element.get('x1', 0)),
                y1=float(svg_element.get('y1', 0)),
                x2=float(svg_element.get('x2', 0)),
                y2=float(svg_element.get('y2', 0)),
                **common_args
            )
            
        elif tag == 'g':
            # Recursively create child shapes
            child_shapes = []
            
            for child in svg_element:
                shape = create_shape_from_svg(child)
                if shape:
                    child_shapes.append(shape)
                    
            return Group(
                shapes=child_shapes,
                **common_args
            )
            
        elif tag == 'path':
            from models.path import Path
            return Path(
                data=svg_element.get('d', ''),
                **common_args
            )
            
        elif tag == 'polygon' or tag == 'polyline':
            from models.path import Polygon, Polyline
            
            # Parse points attribute
            points_str = svg_element.get('points', '')
            point_pairs = points_str.strip().split()
            
            points = []
            for pair in point_pairs:
                if ',' in pair:
                    x, y = pair.split(',')
                    points.append((float(x), float(y)))
                    
            if tag == 'polygon':
                return Polygon(
                    points=points,
                    **common_args
                )
            else:
                return Polyline(
                    points=points,
                    **common_args
                )
                
        elif tag == 'text':
            from models.text import Text
            
            # Extract text content
            text_content = svg_element.text or ""
            
            return Text(
                x=float(svg_element.get('x', 0)),
                y=float(svg_element.get('y', 0)),
                text=text_content,
                font_family=svg_element.get('font-family'),
                font_size=svg_element.get('font-size'),
                font_weight=svg_element.get('font-weight'),
                text_anchor=svg_element.get('text-anchor'),
                **common_args
            )
            
        # Other element types not supported
        return None
        
    except ValueError as e:
        raise ShapeError(f"Invalid value in {tag} element: {e}")
    except Exception as e:
        raise ShapeError(f"Error creating shape from {tag}: {e}")


def parse_svg_document(svg_content: str) -> Group:
    """
    Parse an SVG document into a group of shapes.
    
    Args:
        svg_content: SVG document content
        
    Returns:
        Group containing all shapes in the SVG
        
    Raises:
        ShapeError: If SVG content is invalid
    """
    try:
        # Parse SVG content
        root = ET.fromstring(svg_content)
        
        # Extract SVG namespace if present
        ns = ''
        if '}' in root.tag:
            ns = root.tag.split('}', 1)[0] + '}'
            
        # Extract viewBox and other SVG attributes
        view_box = root.get('viewBox')
        width = root.get('width')
        height = root.get('height')
        
        # Create group attributes
        group_attrs = {
            'shape_id': 'svg_root',
            'data': {
                'viewBox': view_box,
                'width': width,
                'height': height
            }
        }
        
        # Process all child elements
        shapes = []
        
        for child in root:
            shape = create_shape_from_svg(child)
            if shape:
                shapes.append(shape)
                
        # Create root group
        return Group(shapes=shapes, **group_attrs)
        
    except ET.ParseError as e:
        raise ShapeError(f"Invalid SVG content: {e}")
    except Exception as e:
        raise ShapeError(f"Error parsing SVG: {e}")