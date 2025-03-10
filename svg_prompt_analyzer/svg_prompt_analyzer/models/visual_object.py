"""
Production-grade visual object model for SVG generation.
Provides high-level abstractions for complex visual objects combining
multiple shapes with semantic meaning and properties.
"""

import math
import hashlib
import threading
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
from enum import Enum, auto

# Import core optimizations
from svg_prompt_analyzer.core import CONFIG, memoize, Profiler, get_thread_pool
from svg_prompt_analyzer.utils.logger import get_logger
from svg_prompt_analyzer.models.color import Color, ColorPalette
from svg_prompt_analyzer.models.shape import (
    Shape, Rect, Circle, Ellipse, Line, Group, Transform,
    ShapeType, ShapeError, BoundingBox, Point
)

# Configure logger
logger = get_logger(__name__)

# Type aliases
ObjectID = str
Dimension = Tuple[float, float]  # width, height
Position = Tuple[float, float]   # x, y
Style = Dict[str, Any]

# Constants
DEFAULT_STYLE = {
    'fill': '#3498db',  # Blue
    'stroke': '#2c3e50',  # Dark blue
    'stroke_width': 2.0,
    'opacity': 1.0
}
OBJECT_ID_PREFIX = "obj_"


class ObjectCategory(Enum):
    """Categories of visual objects."""
    GEOMETRIC = auto()
    ICON = auto()
    CHART = auto()
    DIAGRAM = auto()
    UI = auto()
    DECORATION = auto()
    CUSTOM = auto()


class VisualObjectError(Exception):
    """Custom exception for visual object errors."""
    pass


class VisualObject:
    """
    High-level visual object for SVG generation.
    
    A visual object combines multiple shapes with semantic meaning,
    properties, and behaviors. It handles layout, styling, and interaction
    in a unified way.
    """
    
    _id_counter = 0
    _id_lock = threading.Lock()
    
    def __init__(
        self,
        obj_id: Optional[ObjectID] = None,
        position: Optional[Position] = None,
        dimension: Optional[Dimension] = None,
        category: ObjectCategory = ObjectCategory.CUSTOM,
        style: Optional[Style] = None,
        properties: Optional[Dict[str, Any]] = None,
        parent: Optional['VisualObject'] = None,
        children: Optional[List['VisualObject']] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a visual object.
        
        Args:
            obj_id: Unique ID for the object (generated if None)
            position: Position (x, y) of the object
            dimension: Dimension (width, height) of the object
            category: Category of the object
            style: Visual style properties
            properties: Custom object properties
            parent: Parent object (if part of a hierarchy)
            children: Child objects
            metadata: Additional metadata
        """
        # Generate ID if not provided
        self._id = obj_id if obj_id else self._generate_id()
        
        # Basic properties
        self._position = position or (0.0, 0.0)
        self._dimension = dimension or (100.0, 100.0)
        self._category = category
        
        # Styling
        self._style = DEFAULT_STYLE.copy()
        if style:
            self._style.update(style)
            
        # Custom properties
        self._properties = properties or {}
        
        # Object hierarchy
        self._parent = parent
        self._children = list(children) if children else []
        
        # Shapes that make up this object
        self._shapes: Dict[str, Shape] = {}
        self._shape_order: List[str] = []
        
        # Metadata
        self._metadata = metadata or {}
        
        # Cached values
        self._group_cache = None
        self._hash_cache = None
        
        # Create default shapes
        self._create_shapes()
    
    @classmethod
    def _generate_id(cls) -> ObjectID:
        """Generate a unique ID for an object."""
        with cls._id_lock:
            cls._id_counter += 1
            return f"{OBJECT_ID_PREFIX}{cls._id_counter}"
    
    @property
    def id(self) -> ObjectID:
        """Get object ID."""
        return self._id
    
    @property
    def position(self) -> Position:
        """Get object position."""
        return self._position
    
    @property
    def dimension(self) -> Dimension:
        """Get object dimension."""
        return self._dimension
    
    @property
    def category(self) -> ObjectCategory:
        """Get object category."""
        return self._category
    
    @property
    def style(self) -> Style:
        """Get object style."""
        return self._style.copy()
    
    @property
    def properties(self) -> Dict[str, Any]:
        """Get object properties."""
        return self._properties.copy()
    
    @property
    def parent(self) -> Optional['VisualObject']:
        """Get parent object."""
        return self._parent
    
    @property
    def children(self) -> List['VisualObject']:
        """Get child objects."""
        return self._children.copy()
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get object metadata."""
        return self._metadata.copy()
    
    @property
    def bounding_box(self) -> BoundingBox:
        """
        Get the bounding box of the object.
        
        Returns:
            Bounding box (x, y, width, height)
        """
        x, y = self._position
        width, height = self._dimension
        return (x, y, width, height)
    
    @property
    def shapes(self) -> Dict[str, Shape]:
        """Get shapes that make up this object."""
        return self._shapes.copy()
    
    def set_position(self, x: float, y: float) -> 'VisualObject':
        """
        Set object position.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            
        Returns:
            Self for method chaining
        """
        old_x, old_y = self._position
        
        # Skip if unchanged
        if old_x == x and old_y == y:
            return self
            
        # Update position
        self._position = (x, y)
        
        # Update shapes
        dx, dy = x - old_x, y - old_y
        self._translate_shapes(dx, dy)
        
        # Clear caches
        self._group_cache = None
        
        return self
    
    def set_dimension(self, width: float, height: float) -> 'VisualObject':
        """
        Set object dimension.
        
        Args:
            width: Width
            height: Height
            
        Returns:
            Self for method chaining
        """
        old_width, old_height = self._dimension
        
        # Skip if unchanged
        if old_width == width and old_height == height:
            return self
            
        # Update dimension
        self._dimension = (width, height)
        
        # Scale shapes
        scale_x = width / max(0.1, old_width)  # Avoid division by zero
        scale_y = height / max(0.1, old_height)
        self._scale_shapes(scale_x, scale_y)
        
        # Clear caches
        self._group_cache = None
        
        return self
    
    def set_style(self, style: Style, merge: bool = True) -> 'VisualObject':
        """
        Set object style.
        
        Args:
            style: Style properties
            merge: Whether to merge with existing style or replace
            
        Returns:
            Self for method chaining
        """
        # Update style
        if merge:
            old_style = self._style.copy()
            old_style.update(style)
            self._style = old_style
        else:
            self._style = style.copy()
            
        # Apply style to shapes
        self._apply_style_to_shapes()
        
        # Clear caches
        self._group_cache = None
        
        return self
    
    def set_property(self, key: str, value: Any) -> 'VisualObject':
        """
        Set a custom property.
        
        Args:
            key: Property key
            value: Property value
            
        Returns:
            Self for method chaining
        """
        # Skip if unchanged
        if key in self._properties and self._properties[key] == value:
            return self
            
        # Update property
        self._properties[key] = value
        
        # Update visual representation if needed
        self._update_shapes_from_properties()
        
        # Clear caches
        self._group_cache = None
        
        return self
    
    def set_metadata(self, key: str, value: Any) -> 'VisualObject':
        """
        Set metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            Self for method chaining
        """
        self._metadata[key] = value
        return self
    
    def add_child(self, child: 'VisualObject') -> 'VisualObject':
        """
        Add a child object.
        
        Args:
            child: Child object to add
            
        Returns:
            Self for method chaining
            
        Raises:
            VisualObjectError: If child is already a child of this object
        """
        if child in self._children:
            raise VisualObjectError(f"Child already exists: {child.id}")
            
        # Add child
        self._children.append(child)
        
        # Set parent reference
        child._parent = self
        
        # Clear caches
        self._group_cache = None
        
        return self
    
    def remove_child(self, child_id: ObjectID) -> Optional['VisualObject']:
        """
        Remove a child object.
        
        Args:
            child_id: ID of child to remove
            
        Returns:
            Removed child or None if not found
        """
        # Find child
        for i, child in enumerate(self._children):
            if child.id == child_id:
                # Remove from children
                removed = self._children.pop(i)
                
                # Clear parent reference
                removed._parent = None
                
                # Clear caches
                self._group_cache = None
                
                return removed
                
        return None
    
    def clear_children(self) -> 'VisualObject':
        """
        Remove all child objects.
        
        Returns:
            Self for method chaining
        """
        if not self._children:
            return self
            
        # Clear parent references
        for child in self._children:
            child._parent = None
            
        # Clear children
        self._children = []
        
        # Clear caches
        self._group_cache = None
        
        return self
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get a custom property.
        
        Args:
            key: Property key
            default: Default value if key not found
            
        Returns:
            Property value or default
        """
        return self._properties.get(key, default)
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self._metadata.get(key, default)
    
    def add_shape(
        self,
        shape: Shape,
        shape_id: Optional[str] = None,
        update_group: bool = True
    ) -> 'VisualObject':
        """
        Add a shape to the object.
        
        Args:
            shape: Shape to add
            shape_id: Optional ID for the shape (defaults to shape.id)
            update_group: Whether to update the group cache
            
        Returns:
            Self for method chaining
        """
        # Use provided ID or shape's ID
        shape_key = shape_id or shape.id
        
        # Add shape
        self._shapes[shape_key] = shape
        if shape_key not in self._shape_order:
            self._shape_order.append(shape_key)
            
        # Clear caches if needed
        if update_group:
            self._group_cache = None
            
        return self
    
    def remove_shape(
        self,
        shape_id: str,
        update_group: bool = True
    ) -> 'VisualObject':
        """
        Remove a shape from the object.
        
        Args:
            shape_id: ID of shape to remove
            update_group: Whether to update the group cache
            
        Returns:
            Self for method chaining
        """
        if shape_id in self._shapes:
            # Remove shape
            del self._shapes[shape_id]
            if shape_id in self._shape_order:
                self._shape_order.remove(shape_id)
                
            # Clear caches if needed
            if update_group:
                self._group_cache = None
                
        return self
    
    def update_shape(
        self,
        shape_id: str,
        shape: Shape,
        update_group: bool = True
    ) -> 'VisualObject':
        """
        Update a shape in the object.
        
        Args:
            shape_id: ID of shape to update
            shape: New shape
            update_group: Whether to update the group cache
            
        Returns:
            Self for method chaining
        """
        if shape_id in self._shapes:
            # Update shape
            self._shapes[shape_id] = shape
            
            # Clear caches if needed
            if update_group:
                self._group_cache = None
                
        return self
    
    def clear_shapes(self) -> 'VisualObject':
        """
        Remove all shapes from the object.
        
        Returns:
            Self for method chaining
        """
        if not self._shapes:
            return self
            
        # Clear shapes
        self._shapes = {}
        self._shape_order = []
        
        # Clear caches
        self._group_cache = None
        
        return self
    
    def to_group(self) -> Group:
        """
        Convert object to a group of shapes.
        
        Returns:
            Group containing all shapes in the object and its children
        """
        # Use cached value if available
        if self._group_cache is not None:
            return self._group_cache
            
        with Profiler("visual_object_to_group"):
            # Collect all shapes from this object
            shapes = []
            
            # Add shapes from this object in order
            for shape_id in self._shape_order:
                if shape_id in self._shapes:
                    shapes.append(self._shapes[shape_id])
                    
            # Add shapes from children
            for child in self._children:
                child_group = child.to_group()
                if child_group.count > 0:
                    shapes.extend(child_group.shapes)
                    
            # Create group
            group = Group(
                shapes=shapes,
                shape_id=f"group_{self._id}"
            )
            
            # Cache result
            self._group_cache = group
            
            return group
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert object to dictionary representation.
        
        Returns:
            Dictionary with object data
        """
        return {
            'id': self._id,
            'position': self._position,
            'dimension': self._dimension,
            'category': self._category.name,
            'style': self._style,
            'properties': self._properties,
            'metadata': self._metadata,
            'children': [child.to_dict() for child in self._children],
            'shape_count': len(self._shapes)
        }
    
    def copy(self, new_id: bool = True) -> 'VisualObject':
        """
        Create a copy of the object.
        
        Args:
            new_id: Whether to generate a new ID for the copy
            
        Returns:
            Copied object
        """
        # Create new object
        copied = self.__class__(
            obj_id=None if new_id else self._id,
            position=self._position,
            dimension=self._dimension,
            category=self._category,
            style=self._style.copy(),
            properties=self._properties.copy(),
            metadata=self._metadata.copy()
        )
        
        # Copy shapes
        for shape_id, shape in self._shapes.items():
            copied.add_shape(shape.copy(), shape_id, update_group=False)
            
        # Set shape order
        copied._shape_order = self._shape_order.copy()
        
        # Copy children
        for child in self._children:
            child_copy = child.copy(new_id)
            copied.add_child(child_copy)
            
        return copied
    
    def _create_shapes(self) -> None:
        """
        Create the shapes that make up this object.
        
        This is a template method that should be overridden by subclasses.
        The default implementation creates a simple rectangle.
        """
        # Create a simple rectangle by default
        x, y = self._position
        width, height = self._dimension
        
        # Apply style properties
        fill = self._style.get('fill')
        stroke = self._style.get('stroke')
        stroke_width = self._style.get('stroke_width', 1.0)
        opacity = self._style.get('opacity', 1.0)
        
        # Create rectangle
        rect = Rect(
            x=x,
            y=y,
            width=width,
            height=height,
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            opacity=opacity,
            shape_id=f"rect_{self._id}"
        )
        
        # Add to shapes
        self.add_shape(rect, update_group=False)
    
    def _apply_style_to_shapes(self) -> None:
        """Apply current style to all shapes."""
        # Extract style properties
        fill = self._style.get('fill')
        stroke = self._style.get('stroke')
        stroke_width = self._style.get('stroke_width', 1.0)
        opacity = self._style.get('opacity', 1.0)
        
        # Apply to shapes
        updated_shapes = []
        
        for shape_id, shape in self._shapes.items():
            # Create updated shape
            updated = shape
            
            if fill is not None:
                updated = updated.with_fill(fill)
                
            if stroke is not None:
                updated = updated.with_stroke(stroke, stroke_width)
                
            if opacity is not None:
                updated = updated.with_opacity(opacity)
                
            # Add to update list if changed
            if updated != shape:
                updated_shapes.append((shape_id, updated))
                
        # Update shapes
        for shape_id, updated in updated_shapes:
            self.update_shape(shape_id, updated, update_group=False)
            
        # Clear caches
        self._group_cache = None
    
    def _translate_shapes(self, dx: float, dy: float) -> None:
        """
        Translate all shapes by a delta.
        
        Args:
            dx: X-axis delta
            dy: Y-axis delta
        """
        if dx == 0 and dy == 0:
            return
            
        # Apply translation to each shape
        updated_shapes = []
        
        for shape_id, shape in self._shapes.items():
            # Create translated shape
            translated = shape.translate(dx, dy)
            
            # Add to update list if changed
            if translated != shape:
                updated_shapes.append((shape_id, translated))
                
        # Update shapes
        for shape_id, translated in updated_shapes:
            self.update_shape(shape_id, translated, update_group=False)
            
        # Clear caches
        self._group_cache = None
    
    def _scale_shapes(self, scale_x: float, scale_y: float) -> None:
        """
        Scale all shapes.
        
        Args:
            scale_x: X-axis scale factor
            scale_y: Y-axis scale factor
        """
        if scale_x == 1.0 and scale_y == 1.0:
            return
            
        # Get object center for scaling
        x, y = self._position
        width, height = self._dimension
        center_x = x + width / 2
        center_y = y + height / 2
        
        # Apply scaling to each shape
        updated_shapes = []
        
        for shape_id, shape in self._shapes.items():
            # Create scaled shape (from center)
            scaled = shape.translate(-center_x, -center_y)
            scaled = scaled.scale(scale_x, scale_y)
            scaled = scaled.translate(center_x, center_y)
            
            # Add to update list if changed
            if scaled != shape:
                updated_shapes.append((shape_id, scaled))
                
        # Update shapes
        for shape_id, scaled in updated_shapes:
            self.update_shape(shape_id, scaled, update_group=False)
            
        # Clear caches
        self._group_cache = None
    
    def _update_shapes_from_properties(self) -> None:
        """
        Update shapes based on custom properties.
        
        This is a template method that can be overridden by subclasses.
        The default implementation does nothing.
        """
        pass
    
    def __eq__(self, other: object) -> bool:
        """Check if objects are equal."""
        if not isinstance(other, VisualObject):
            return False
            
        # Compare basic properties
        if (self._id != other._id or
            self._position != other._position or
            self._dimension != other._dimension or
            self._category != other._category):
            return False
            
        # Compare shapes
        if len(self._shapes) != len(other._shapes):
            return False
            
        # Compare shape order
        if self._shape_order != other._shape_order:
            return False
            
        # Compare each shape
        for shape_id, shape in self._shapes.items():
            if shape_id not in other._shapes:
                return False
            if shape != other._shapes[shape_id]:
                return False
                
        # Compare children
        if len(self._children) != len(other._children):
            return False
            
        for i, child in enumerate(self._children):
            if child != other._children[i]:
                return False
                
        return True
    
    def __hash__(self) -> int:
        """Hash for dictionary keys."""
        if self._hash_cache is None:
            # Build hash from essential properties
            obj_hash = hashlib.md5()
            
            obj_hash.update(f"{self._id}:{self._position}:{self._dimension}".encode())
            obj_hash.update(self._category.name.encode())
            
            # Add shape hashes
            for shape_id in self._shape_order:
                if shape_id in self._shapes:
                    shape = self._shapes[shape_id]
                    obj_hash.update(f"{shape_id}:{hash(shape)}".encode())
                    
            # Convert to integer
            self._hash_cache = int(obj_hash.hexdigest(), 16) % (2**32)
            
        return self._hash_cache
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(id={self._id})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return (f"{self.__class__.__name__}(id={self._id}, "
                f"pos={self._position}, dim={self._dimension})")


class Rectangle(VisualObject):
    """Rectangle visual object with optional rounded corners."""
    
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 100.0,
        height: float = 100.0,
        corner_radius: float = 0.0,
        **kwargs
    ):
        """
        Initialize a rectangle object.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            width: Width
            height: Height
            corner_radius: Corner radius for rounded corners
            **kwargs: Additional parameters for VisualObject
        """
        # Set corner radius as property
        properties = kwargs.pop('properties', {})
        properties['corner_radius'] = corner_radius
        
        # Set category
        kwargs['category'] = ObjectCategory.GEOMETRIC
        
        # Initialize base
        super().__init__(
            position=(x, y),
            dimension=(width, height),
            properties=properties,
            **kwargs
        )
    
    def _create_shapes(self) -> None:
        """Create rectangle shape."""
        x, y = self._position
        width, height = self._dimension
        corner_radius = self._properties.get('corner_radius', 0.0)
        
        # Apply style properties
        fill = self._style.get('fill')
        stroke = self._style.get('stroke')
        stroke_width = self._style.get('stroke_width', 1.0)
        opacity = self._style.get('opacity', 1.0)
        
        # Create rectangle
        rect = Rect(
            x=x,
            y=y,
            width=width,
            height=height,
            rx=corner_radius if corner_radius > 0 else None,
            ry=corner_radius if corner_radius > 0 else None,
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            opacity=opacity,
            shape_id=f"rect_{self._id}"
        )
        
        # Add to shapes
        self.add_shape(rect, update_group=False)
    
    def _update_shapes_from_properties(self) -> None:
        """Update shapes based on corner_radius property."""
        # Get corner radius
        corner_radius = self._properties.get('corner_radius', 0.0)
        
        # Update rectangle
        rect_id = f"rect_{self._id}"
        if rect_id in self._shapes:
            rect = self._shapes[rect_id]
            
            # Create updated rect with new corner radius
            updated = rect.with_corner_radius(
                corner_radius if corner_radius > 0 else None
            )
            
            # Update if changed
            if updated != rect:
                self.update_shape(rect_id, updated)


class Circle(VisualObject):
    """Circle visual object."""
    
    def __init__(
        self,
        cx: float = 0.0,
        cy: float = 0.0,
        radius: float = 50.0,
        **kwargs
    ):
        """
        Initialize a circle object.
        
        Args:
            cx: Center x-coordinate
            cy: Center y-coordinate
            radius: Radius
            **kwargs: Additional parameters for VisualObject
        """
        # Calculate position (top-left corner)
        x = cx - radius
        y = cy - radius
        
        # Calculate dimension
        diameter = radius * 2
        
        # Set center and radius as properties
        properties = kwargs.pop('properties', {})
        properties.update({
            'center_x': cx,
            'center_y': cy,
            'radius': radius
        })
        
        # Set category
        kwargs['category'] = ObjectCategory.GEOMETRIC
        
        # Initialize base
        super().__init__(
            position=(x, y),
            dimension=(diameter, diameter),
            properties=properties,
            **kwargs
        )
    
    def _create_shapes(self) -> None:
        """Create circle shape."""
        # Get center and radius
        cx = self._properties.get('center_x', 0.0)
        cy = self._properties.get('center_y', 0.0)
        radius = self._properties.get('radius', 50.0)
        
        # Apply style properties
        fill = self._style.get('fill')
        stroke = self._style.get('stroke')
        stroke_width = self._style.get('stroke_width', 1.0)
        opacity = self._style.get('opacity', 1.0)
        
        # Create circle
        circle = Circle(
            cx=cx,
            cy=cy,
            radius=radius,
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            opacity=opacity,
            shape_id=f"circle_{self._id}"
        )
        
        # Add to shapes
        self.add_shape(circle, update_group=False)
    
    def _update_shapes_from_properties(self) -> None:
        """Update shapes based on center and radius properties."""
        # Get center and radius
        cx = self._properties.get('center_x', 0.0)
        cy = self._properties.get('center_y', 0.0)
        radius = self._properties.get('radius', 50.0)
        
        # Update position and dimension
        x = cx - radius
        y = cy - radius
        diameter = radius * 2
        
        # Check if position or dimension changed
        if (self._position != (x, y) or 
            self._dimension != (diameter, diameter)):
            # Update base properties
            self._position = (x, y)
            self._dimension = (diameter, diameter)
            
            # Update circle
            circle_id = f"circle_{self._id}"
            if circle_id in self._shapes:
                circle = self._shapes[circle_id]
                
                # Create updated circle
                updated = circle
                
                if circle.cx != cx or circle.cy != cy:
                    updated = updated.with_center(cx, cy)
                    
                if circle.radius != radius:
                    updated = updated.with_radius(radius)
                    
                # Update if changed
                if updated != circle:
                    self.update_shape(circle_id, updated)


class Ellipse(VisualObject):
    """Ellipse visual object."""
    
    def __init__(
        self,
        cx: float = 0.0,
        cy: float = 0.0,
        rx: float = 50.0,
        ry: float = 30.0,
        **kwargs
    ):
        """
        Initialize an ellipse object.
        
        Args:
            cx: Center x-coordinate
            cy: Center y-coordinate
            rx: X-axis radius
            ry: Y-axis radius
            **kwargs: Additional parameters for VisualObject
        """
        # Calculate position (top-left corner)
        x = cx - rx
        y = cy - ry
        
        # Calculate dimension
        width = rx * 2
        height = ry * 2
        
        # Set center and radii as properties
        properties = kwargs.pop('properties', {})
        properties.update({
            'center_x': cx,
            'center_y': cy,
            'radius_x': rx,
            'radius_y': ry
        })
        
        # Set category
        kwargs['category'] = ObjectCategory.GEOMETRIC
        
        # Initialize base
        super().__init__(
            position=(x, y),
            dimension=(width, height),
            properties=properties,
            **kwargs
        )
    
    def _create_shapes(self) -> None:
        """Create ellipse shape."""
        # Get center and radii
        cx = self._properties.get('center_x', 0.0)
        cy = self._properties.get('center_y', 0.0)
        rx = self._properties.get('radius_x', 50.0)
        ry = self._properties.get('radius_y', 30.0)
        
        # Apply style properties
        fill = self._style.get('fill')
        stroke = self._style.get('stroke')
        stroke_width = self._style.get('stroke_width', 1.0)
        opacity = self._style.get('opacity', 1.0)
        
        # Create ellipse
        ellipse = Ellipse(
            cx=cx,
            cy=cy,
            rx=rx,
            ry=ry,
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            opacity=opacity,
            shape_id=f"ellipse_{self._id}"
        )
        
        # Add to shapes
        self.add_shape(ellipse, update_group=False)
    
    def _update_shapes_from_properties(self) -> None:
        """Update shapes based on center and radii properties."""
        # Get center and radii
        cx = self._properties.get('center_x', 0.0)
        cy = self._properties.get('center_y', 0.0)
        rx = self._properties.get('radius_x', 50.0)
        ry = self._properties.get('radius_y', 30.0)
        
        # Update position and dimension
        x = cx - rx
        y = cy - ry
        width = rx * 2
        height = ry * 2
        
        # Check if position or dimension changed
        if (self._position != (x, y) or 
            self._dimension != (width, height)):
            # Update base properties
            self._position = (x, y)
            self._dimension = (width, height)
            
            # Update ellipse
            ellipse_id = f"ellipse_{self._id}"
            if ellipse_id in self._shapes:
                ellipse = self._shapes[ellipse_id]
                
                # Create updated ellipse
                updated = ellipse
                
                if ellipse.cx != cx or ellipse.cy != cy:
                    updated = updated.with_center(cx, cy)
                    
                if ellipse.rx != rx or ellipse.ry != ry:
                    updated = updated.with_radii(rx, ry)
                    
                # Update if changed
                if updated != ellipse:
                    self.update_shape(ellipse_id, updated)


class Line(VisualObject):
    """Line visual object."""
    
    def __init__(
        self,
        x1: float = 0.0,
        y1: float = 0.0,
        x2: float = 100.0,
        y2: float = 100.0,
        **kwargs
    ):
        """
        Initialize a line object.
        
        Args:
            x1: Start x-coordinate
            y1: Start y-coordinate
            x2: End x-coordinate
            y2: End y-coordinate
            **kwargs: Additional parameters for VisualObject
        """
        # Calculate position (top-left corner of bounding box)
        x = min(x1, x2)
        y = min(y1, y2)
        
        # Calculate dimension (size of bounding box)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Set points as properties
        properties = kwargs.pop('properties', {})
        properties.update({
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        })
        
        # Set category
        kwargs['category'] = ObjectCategory.GEOMETRIC
        
        # Initialize base
        super().__init__(
            position=(x, y),
            dimension=(width, height),
            properties=properties,
            **kwargs
        )
    
    def _create_shapes(self) -> None:
        """Create line shape."""
        # Get points
        x1 = self._properties.get('x1', 0.0)
        y1 = self._properties.get('y1', 0.0)
        x2 = self._properties.get('x2', 100.0)
        y2 = self._properties.get('y2', 100.0)
        
        # Apply style properties
        stroke = self._style.get('stroke', '#000000')  # Default to black
        stroke_width = self._style.get('stroke_width', 1.0)
        opacity = self._style.get('opacity', 1.0)
        
        # Create line (always needs a stroke)
        line = Line(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            stroke=stroke,
            stroke_width=stroke_width,
            opacity=opacity,
            shape_id=f"line_{self._id}"
        )
        
        # Add to shapes
        self.add_shape(line, update_group=False)
    
    def _update_shapes_from_properties(self) -> None:
        """Update shapes based on point properties."""
        # Get points
        x1 = self._properties.get('x1', 0.0)
        y1 = self._properties.get('y1', 0.0)
        x2 = self._properties.get('x2', 100.0)
        y2 = self._properties.get('y2', 100.0)
        
        # Update position and dimension
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Update base properties
        self._position = (x, y)
        self._dimension = (width, height)
        
        # Update line
        line_id = f"line_{self._id}"
        if line_id in self._shapes:
            line = self._shapes[line_id]
            
            # Create updated line
            updated = line.with_points(x1, y1, x2, y2)
            
            # Update if changed
            if updated != line:
                self.update_shape(line_id, updated)


# Factory function to create visual objects by type
def create_visual_object(
    obj_type: str,
    **kwargs
) -> VisualObject:
    """
    Create a visual object by type.
    
    Args:
        obj_type: Type of object to create
        **kwargs: Parameters for the object
        
    Returns:
        Created visual object
        
    Raises:
        VisualObjectError: If object type is unknown
    """
    type_lower = obj_type.lower()
    
    if type_lower == 'rectangle':
        return Rectangle(**kwargs)
    elif type_lower == 'circle':
        return Circle(**kwargs)
    elif type_lower == 'ellipse':
        return Ellipse(**kwargs)
    elif type_lower == 'line':
        return Line(**kwargs)
    else:
        # Default to generic visual object
        return VisualObject(**kwargs)