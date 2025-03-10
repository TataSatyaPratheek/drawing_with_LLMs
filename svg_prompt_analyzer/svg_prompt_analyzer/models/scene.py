"""
Production-grade scene model for SVG visualization and analysis.
Provides efficient representation of complex visual scenes with
optimized memory usage and rendering capabilities.
"""

import os
import math
import json
import time
import threading
import hashlib
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable, Iterator
from enum import Enum, auto
import xml.etree.ElementTree as ET

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
SceneID = str
ShapeID = str
Viewport = Tuple[float, float, float, float]  # x, y, width, height

# Constants
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600
DEFAULT_BACKGROUND = None  # Transparent background
MAX_UNDO_HISTORY = 50
SCENE_ID_PREFIX = "scene_"


class SceneEvent(Enum):
    """Events that can occur within a scene."""
    SHAPE_ADDED = auto()
    SHAPE_REMOVED = auto()
    SHAPE_UPDATED = auto()
    SCENE_CLEARED = auto()
    VIEWPORT_CHANGED = auto()
    BACKGROUND_CHANGED = auto()
    SCENE_LOADED = auto()
    SCENE_SAVED = auto()


class SceneError(Exception):
    """Custom exception for scene-related errors."""
    pass


class Scene:
    """
    Memory-efficient scene representation for SVG visualization.
    
    Represents a visual scene with shapes, background, viewport,
    and operations for manipulation and rendering.
    """
    
    def __init__(
        self,
        scene_id: Optional[SceneID] = None,
        width: float = DEFAULT_WIDTH,
        height: float = DEFAULT_HEIGHT,
        background: Optional[Union[Color, str]] = DEFAULT_BACKGROUND,
        shapes: Optional[List[Shape]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        viewport: Optional[Viewport] = None
    ):
        """
        Initialize a scene.
        
        Args:
            scene_id: Unique ID for the scene (generated if None)
            width: Scene width
            height: Scene height
            background: Background color
            shapes: Initial shapes in the scene
            metadata: Additional scene metadata
            viewport: Initial viewport (x, y, width, height)
        """
        # Generate ID if not provided
        self._id = scene_id or self._generate_id()
        self._width = max(1, width)
        self._height = max(1, height)
        
        # Process background color
        if background is None:
            self._background = None
        elif isinstance(background, Color):
            self._background = background
        else:
            try:
                self._background = Color(background)
            except Exception as e:
                logger.warning(f"Invalid background color: {background} - {str(e)}")
                self._background = None
                
        # Set up shape storage with indexing
        self._shapes: Dict[ShapeID, Shape] = {}
        self._shape_order: List[ShapeID] = []
        
        # Add initial shapes if provided
        if shapes:
            for shape in shapes:
                self.add_shape(shape)
                
        # Set up metadata
        self._metadata = metadata or {}
        
        # Set up viewport
        if viewport:
            self._viewport = viewport
        else:
            # Default to full scene size
            self._viewport = (0, 0, self._width, self._height)
            
        # Initialize undo/redo history
        self._history: List[Dict[str, Any]] = []
        self._history_position = -1
        self._recording_history = True
        
        # Initialize event handlers
        self._event_handlers: Dict[SceneEvent, List[Callable]] = {
            event: [] for event in SceneEvent
        }
        
        # Thread lock for thread safety
        self._lock = threading.RLock()
        
        # Cache for common operations
        self._bounding_box_cache = None
        self._svg_string_cache = None
        self._hash_cache = None
    
    @staticmethod
    def _generate_id() -> SceneID:
        """Generate a unique ID for a scene."""
        import uuid
        return f"{SCENE_ID_PREFIX}{uuid.uuid4().hex[:8]}"
    
    @property
    def id(self) -> SceneID:
        """Get scene ID."""
        return self._id
    
    @property
    def width(self) -> float:
        """Get scene width."""
        return self._width
    
    @property
    def height(self) -> float:
        """Get scene height."""
        return self._height
    
    @property
    def background(self) -> Optional[Color]:
        """Get background color."""
        return self._background
    
    @property
    def viewport(self) -> Viewport:
        """Get current viewport."""
        return self._viewport
    
    @property
    def shape_count(self) -> int:
        """Get number of shapes in the scene."""
        return len(self._shapes)
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get scene metadata."""
        return self._metadata.copy()
    
    @property
    def bounding_box(self) -> BoundingBox:
        """
        Get the bounding box of all shapes in the scene.
        
        Returns:
            Bounding box (x, y, width, height)
        """
        with self._lock:
            # Use cached value if available
            if self._bounding_box_cache is not None:
                return self._bounding_box_cache
                
            if not self._shapes:
                bbox = (0, 0, self._width, self._height)
            else:
                # Collect all shape bounding boxes
                shape_boxes = [shape.get_bounding_box() 
                              for shape in self._shapes.values()]
                
                if not shape_boxes:
                    bbox = (0, 0, self._width, self._height)
                else:
                    # Find extremes
                    x_min = min(box[0] for box in shape_boxes)
                    y_min = min(box[1] for box in shape_boxes)
                    x_max = max(box[0] + box[2] for box in shape_boxes)
                    y_max = max(box[1] + box[3] for box in shape_boxes)
                    
                    # Ensure bounds are within scene dimensions
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(self._width, x_max)
                    y_max = min(self._height, y_max)
                    
                    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    
            self._bounding_box_cache = bbox
            return bbox
    
    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._history_position > 0
    
    @property
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._history_position < len(self._history) - 1
    
    def set_size(self, width: float, height: float) -> 'Scene':
        """
        Set scene size.
        
        Args:
            width: New width
            height: New height
            
        Returns:
            Self for method chaining
            
        Raises:
            SceneError: If dimensions are invalid
        """
        if width <= 0 or height <= 0:
            raise SceneError(f"Invalid dimensions: {width}x{height}")
            
        with self._lock:
            # Record history
            self._record_history('set_size', {
                'old_width': self._width,
                'old_height': self._height,
                'new_width': width,
                'new_height': height
            })
            
            # Update size
            self._width = width
            self._height = height
            
            # Adjust viewport if needed
            vx, vy, vw, vh = self._viewport
            if vx + vw > width or vy + vh > height:
                self._viewport = (
                    vx,
                    vy,
                    min(vw, width - vx),
                    min(vh, height - vy)
                )
                self._trigger_event(SceneEvent.VIEWPORT_CHANGED)
                
            # Invalidate caches
            self._invalidate_caches()
            
        return self
    
    def set_background(self, color: Optional[Union[Color, str]]) -> 'Scene':
        """
        Set background color.
        
        Args:
            color: New background color or None for transparent
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            old_background = self._background
            
            # Process color
            if color is None:
                new_background = None
            elif isinstance(color, Color):
                new_background = color
            else:
                try:
                    new_background = Color(color)
                except Exception as e:
                    logger.warning(f"Invalid background color: {color} - {str(e)}")
                    new_background = None
                    
            # Record history if changed
            if new_background != old_background:
                self._record_history('set_background', {
                    'old_background': old_background,
                    'new_background': new_background
                })
                
                self._background = new_background
                self._trigger_event(SceneEvent.BACKGROUND_CHANGED)
                self._invalidate_caches()
                
        return self
    
    def set_viewport(
        self,
        x: float,
        y: float,
        width: float,
        height: float
    ) -> 'Scene':
        """
        Set viewport for the scene.
        
        Args:
            x: Viewport x-coordinate
            y: Viewport y-coordinate
            width: Viewport width
            height: Viewport height
            
        Returns:
            Self for method chaining
            
        Raises:
            SceneError: If viewport is invalid
        """
        if width <= 0 or height <= 0:
            raise SceneError(f"Invalid viewport dimensions: {width}x{height}")
            
        with self._lock:
            # Ensure viewport is within scene bounds
            x = max(0, min(x, self._width))
            y = max(0, min(y, self._height))
            width = min(width, self._width - x)
            height = min(height, self._height - y)
            
            new_viewport = (x, y, width, height)
            
            # Record history if changed
            if new_viewport != self._viewport:
                self._record_history('set_viewport', {
                    'old_viewport': self._viewport,
                    'new_viewport': new_viewport
                })
                
                self._viewport = new_viewport
                self._trigger_event(SceneEvent.VIEWPORT_CHANGED)
                self._invalidate_caches()
                
        return self
    
    def reset_viewport(self) -> 'Scene':
        """
        Reset viewport to full scene.
        
        Returns:
            Self for method chaining
        """
        return self.set_viewport(0, 0, self._width, self._height)
    
    def fit_viewport_to_content(self, padding: float = 10) -> 'Scene':
        """
        Adjust viewport to fit all content.
        
        Args:
            padding: Padding around content
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            if not self._shapes:
                return self.reset_viewport()
                
            # Get content bounding box
            x, y, width, height = self.bounding_box
            
            # Add padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            width = min(self._width - x, width + 2 * padding)
            height = min(self._height - y, height + 2 * padding)
            
            # Set viewport
            return self.set_viewport(x, y, width, height)
    
    def zoom_viewport(
        self,
        scale_factor: float,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None
    ) -> 'Scene':
        """
        Zoom viewport by a scale factor.
        
        Args:
            scale_factor: Scale factor (> 1 to zoom in, < 1 to zoom out)
            center_x: X-coordinate of zoom center (default: viewport center)
            center_y: Y-coordinate of zoom center (default: viewport center)
            
        Returns:
            Self for method chaining
        """
        if scale_factor <= 0:
            raise SceneError(f"Invalid scale factor: {scale_factor}")
            
        with self._lock:
            # Get current viewport
            vx, vy, vw, vh = self._viewport
            
            # Default zoom center to viewport center
            if center_x is None:
                center_x = vx + vw / 2
            if center_y is None:
                center_y = vy + vh / 2
                
            # Calculate new dimensions
            new_width = vw / scale_factor
            new_height = vh / scale_factor
            
            # Calculate new position, keeping zoom center fixed
            cx_ratio = (center_x - vx) / vw
            cy_ratio = (center_y - vy) / vh
            
            new_x = center_x - cx_ratio * new_width
            new_y = center_y - cy_ratio * new_height
            
            # Set new viewport
            return self.set_viewport(new_x, new_y, new_width, new_height)
    
    def pan_viewport(self, delta_x: float, delta_y: float) -> 'Scene':
        """
        Pan viewport by a specified amount.
        
        Args:
            delta_x: X-axis movement
            delta_y: Y-axis movement
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            # Get current viewport
            vx, vy, vw, vh = self._viewport
            
            # Calculate new position
            new_x = vx + delta_x
            new_y = vy + delta_y
            
            # Set new viewport
            return self.set_viewport(new_x, new_y, vw, vh)
    
    def add_shape(self, shape: Shape) -> 'Scene':
        """
        Add a shape to the scene.
        
        Args:
            shape: Shape to add
            
        Returns:
            Self for method chaining
            
        Raises:
            SceneError: If shape is invalid or ID conflicts
        """
        if shape is None:
            raise SceneError("Cannot add None as a shape")
            
        with self._lock:
            shape_id = shape.id
            
            # Check for duplicate IDs
            if shape_id in self._shapes:
                raise SceneError(f"Shape ID already exists: {shape_id}")
                
            # Record history
            self._record_history('add_shape', {
                'shape_id': shape_id,
                'shape_data': shape.to_dict()
            })
            
            # Add shape
            self._shapes[shape_id] = shape
            self._shape_order.append(shape_id)
            
            # Invalidate caches
            self._invalidate_caches()
            
            # Trigger event
            self._trigger_event(SceneEvent.SHAPE_ADDED, shape)
            
        return self
    
    def add_shapes(self, shapes: List[Shape]) -> 'Scene':
        """
        Add multiple shapes to the scene.
        
        Args:
            shapes: List of shapes to add
            
        Returns:
            Self for method chaining
        """
        if not shapes:
            return self
            
        with self._lock:
            # Check for duplicate IDs
            shape_ids = set(self._shapes.keys())
            for shape in shapes:
                if shape.id in shape_ids:
                    raise SceneError(f"Shape ID already exists: {shape.id}")
                shape_ids.add(shape.id)
                
            # Record history (as batch operation)
            shape_data = [(shape.id, shape.to_dict()) for shape in shapes]
            self._record_history('add_shapes', {
                'shape_data': shape_data
            })
            
            # Add shapes
            for shape in shapes:
                self._shapes[shape.id] = shape
                self._shape_order.append(shape.id)
                
            # Invalidate caches
            self._invalidate_caches()
            
            # Trigger events
            for shape in shapes:
                self._trigger_event(SceneEvent.SHAPE_ADDED, shape)
                
        return self
    
    def remove_shape(self, shape_id: ShapeID) -> 'Scene':
        """
        Remove a shape from the scene.
        
        Args:
            shape_id: ID of shape to remove
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            if shape_id in self._shapes:
                # Get shape for event trigger
                shape = self._shapes[shape_id]
                
                # Record history
                self._record_history('remove_shape', {
                    'shape_id': shape_id,
                    'shape_data': shape.to_dict(),
                    'shape_index': self._shape_order.index(shape_id)
                })
                
                # Remove shape
                del self._shapes[shape_id]
                self._shape_order.remove(shape_id)
                
                # Invalidate caches
                self._invalidate_caches()
                
                # Trigger event
                self._trigger_event(SceneEvent.SHAPE_REMOVED, shape)
                
        return self
    
    def remove_shapes(self, shape_ids: List[ShapeID]) -> 'Scene':
        """
        Remove multiple shapes from the scene.
        
        Args:
            shape_ids: List of shape IDs to remove
            
        Returns:
            Self for method chaining
        """
        if not shape_ids:
            return self
            
        with self._lock:
            # Filter valid shape IDs
            valid_ids = [sid for sid in shape_ids if sid in self._shapes]
            
            if valid_ids:
                # Record history (as batch operation)
                shape_data = []
                for shape_id in valid_ids:
                    shape = self._shapes[shape_id]
                    shape_index = self._shape_order.index(shape_id)
                    shape_data.append((shape_id, shape.to_dict(), shape_index))
                    
                self._record_history('remove_shapes', {
                    'shape_data': shape_data
                })
                
                # Get shapes for event triggers
                shapes = [self._shapes[sid] for sid in valid_ids]
                
                # Remove shapes
                for shape_id in valid_ids:
                    del self._shapes[shape_id]
                    self._shape_order.remove(shape_id)
                    
                # Invalidate caches
                self._invalidate_caches()
                
                # Trigger events
                for shape in shapes:
                    self._trigger_event(SceneEvent.SHAPE_REMOVED, shape)
                    
        return self
    
    def update_shape(self, shape_id: ShapeID, shape: Shape) -> 'Scene':
        """
        Update a shape in the scene.
        
        Args:
            shape_id: ID of shape to update
            shape: New shape to replace it with
            
        Returns:
            Self for method chaining
            
        Raises:
            SceneError: If shape ID doesn't exist
        """
        with self._lock:
            if shape_id not in self._shapes:
                raise SceneError(f"Shape ID not found: {shape_id}")
                
            # Get old shape for history and event
            old_shape = self._shapes[shape_id]
            
            # Record history
            self._record_history('update_shape', {
                'shape_id': shape_id,
                'old_shape_data': old_shape.to_dict(),
                'new_shape_data': shape.to_dict()
            })
            
            # Update shape
            self._shapes[shape_id] = shape
            
            # Update shape ID in order list if changed
            if shape_id != shape.id:
                idx = self._shape_order.index(shape_id)
                self._shape_order[idx] = shape.id
                self._shapes[shape.id] = shape
                del self._shapes[shape_id]
                
            # Invalidate caches
            self._invalidate_caches()
            
            # Trigger event
            self._trigger_event(SceneEvent.SHAPE_UPDATED, old_shape, shape)
            
        return self
    
    def update_shapes(
        self,
        updates: List[Tuple[ShapeID, Shape]]
    ) -> 'Scene':
        """
        Update multiple shapes in the scene.
        
        Args:
            updates: List of (shape_id, new_shape) tuples
            
        Returns:
            Self for method chaining
        """
        if not updates:
            return self
            
        with self._lock:
            # Filter valid updates
            valid_updates = [(sid, shape) for sid, shape in updates 
                            if sid in self._shapes]
            
            if valid_updates:
                # Record history (as batch operation)
                update_data = []
                for shape_id, new_shape in valid_updates:
                    old_shape = self._shapes[shape_id]
                    update_data.append((
                        shape_id,
                        old_shape.to_dict(),
                        new_shape.to_dict()
                    ))
                    
                self._record_history('update_shapes', {
                    'update_data': update_data
                })
                
                # Get old shapes for event triggers
                old_shapes = [self._shapes[sid] for sid, _ in valid_updates]
                
                # Update shapes
                for i, (shape_id, shape) in enumerate(valid_updates):
                    self._shapes[shape_id] = shape
                    
                    # Update shape ID in order list if changed
                    if shape_id != shape.id:
                        idx = self._shape_order.index(shape_id)
                        self._shape_order[idx] = shape.id
                        self._shapes[shape.id] = shape
                        del self._shapes[shape_id]
                        
                # Invalidate caches
                self._invalidate_caches()
                
                # Trigger events
                for i, (_, new_shape) in enumerate(valid_updates):
                    self._trigger_event(
                        SceneEvent.SHAPE_UPDATED,
                        old_shapes[i],
                        new_shape
                    )
                    
        return self
    
    def get_shape(self, shape_id: ShapeID) -> Optional[Shape]:
        """
        Get a shape by ID.
        
        Args:
            shape_id: Shape ID to get
            
        Returns:
            Shape with the ID or None if not found
        """
        with self._lock:
            return self._shapes.get(shape_id)
    
    def find_shapes_at_point(self, x: float, y: float) -> List[Shape]:
        """
        Find shapes that contain a point.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            
        Returns:
            List of shapes at the point, in z-order (top shape first)
        """
        with self._lock:
            # Check each shape in reverse z-order
            point = (x, y)
            matching_shapes = []
            
            # Iterate in reverse to get top-most shapes first
            for shape_id in reversed(self._shape_order):
                shape = self._shapes[shape_id]
                if shape.contains_point(point):
                    matching_shapes.append(shape)
                    
            return matching_shapes
    
    def find_shapes_in_region(
        self,
        x: float,
        y: float,
        width: float,
        height: float
    ) -> List[Shape]:
        """
        Find shapes that intersect with a region.
        
        Args:
            x: Region x-coordinate
            y: Region y-coordinate
            width: Region width
            height: Region height
            
        Returns:
            List of shapes in the region, in z-order
        """
        with self._lock:
            # Create region bounding box
            region = (x, y, width, height)
            
            # Check each shape
            matching_shapes = []
            
            for shape_id in self._shape_order:
                shape = self._shapes[shape_id]
                shape_bbox = shape.get_bounding_box()
                
                # Check for intersection
                if self._bounding_boxes_intersect(region, shape_bbox):
                    matching_shapes.append(shape)
                    
            return matching_shapes
    
    def clear(self) -> 'Scene':
        """
        Remove all shapes from the scene.
        
        Returns:
            Self for method chaining
        """
        with self._lock:
            if not self._shapes:
                return self
                
            # Record history
            self._record_history('clear', {
                'shape_data': [(sid, shape.to_dict()) 
                              for sid, shape in self._shapes.items()],
                'shape_order': self._shape_order.copy()
            })
            
            # Clear shapes
            self._shapes = {}
            self._shape_order = []
            
            # Invalidate caches
            self._invalidate_caches()
            
            # Trigger event
            self._trigger_event(SceneEvent.SCENE_CLEARED)
            
        return self
    
    def bring_to_front(self, shape_id: ShapeID) -> 'Scene':
        """
        Move a shape to the front (top of z-order).
        
        Args:
            shape_id: ID of shape to move
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            if shape_id in self._shapes:
                # Skip if already at front
                if self._shape_order[-1] == shape_id:
                    return self
                    
                # Record history
                self._record_history('reorder_shape', {
                    'shape_id': shape_id,
                    'old_index': self._shape_order.index(shape_id),
                    'new_index': len(self._shape_order) - 1
                })
                
                # Move shape to front
                self._shape_order.remove(shape_id)
                self._shape_order.append(shape_id)
                
                # Invalidate caches
                self._invalidate_caches()
                
        return self
    
    def send_to_back(self, shape_id: ShapeID) -> 'Scene':
        """
        Move a shape to the back (bottom of z-order).
        
        Args:
            shape_id: ID of shape to move
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            if shape_id in self._shapes:
                # Skip if already at back
                if self._shape_order[0] == shape_id:
                    return self
                    
                # Record history
                self._record_history('reorder_shape', {
                    'shape_id': shape_id,
                    'old_index': self._shape_order.index(shape_id),
                    'new_index': 0
                })
                
                # Move shape to back
                self._shape_order.remove(shape_id)
                self._shape_order.insert(0, shape_id)
                
                # Invalidate caches
                self._invalidate_caches()
                
        return self
    
    def move_forward(self, shape_id: ShapeID) -> 'Scene':
        """
        Move a shape one level forward in z-order.
        
        Args:
            shape_id: ID of shape to move
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            if shape_id in self._shapes:
                # Get current index
                current_index = self._shape_order.index(shape_id)
                
                # Skip if already at front
                if current_index == len(self._shape_order) - 1:
                    return self
                    
                # Record history
                self._record_history('reorder_shape', {
                    'shape_id': shape_id,
                    'old_index': current_index,
                    'new_index': current_index + 1
                })
                
                # Move shape forward
                self._shape_order.remove(shape_id)
                self._shape_order.insert(current_index + 1, shape_id)
                
                # Invalidate caches
                self._invalidate_caches()
                
        return self
    
    def move_backward(self, shape_id: ShapeID) -> 'Scene':
        """
        Move a shape one level backward in z-order.
        
        Args:
            shape_id: ID of shape to move
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            if shape_id in self._shapes:
                # Get current index
                current_index = self._shape_order.index(shape_id)
                
                # Skip if already at back
                if current_index == 0:
                    return self
                    
                # Record history
                self._record_history('reorder_shape', {
                    'shape_id': shape_id,
                    'old_index': current_index,
                    'new_index': current_index - 1
                })
                
                # Move shape backward
                self._shape_order.remove(shape_id)
                self._shape_order.insert(current_index - 1, shape_id)
                
                # Invalidate caches
                self._invalidate_caches()
                
        return self
    
    def set_shape_order(self, shape_ids: List[ShapeID]) -> 'Scene':
        """
        Set the z-order of shapes.
        
        Args:
            shape_ids: List of shape IDs in desired order
            
        Returns:
            Self for method chaining
            
        Raises:
            SceneError: If shape_ids doesn't match current shapes
        """
        with self._lock:
            # Validate shape IDs
            if set(shape_ids) != set(self._shapes.keys()):
                raise SceneError("Shape ID list doesn't match current shapes")
                
            # Skip if order is unchanged
            if shape_ids == self._shape_order:
                return self
                
            # Record history
            self._record_history('set_shape_order', {
                'old_order': self._shape_order.copy(),
                'new_order': shape_ids.copy()
            })
            
            # Update order
            self._shape_order = shape_ids.copy()
            
            # Invalidate caches
            self._invalidate_caches()
            
        return self
    
    def get_shapes(self) -> List[Shape]:
        """
        Get all shapes in z-order.
        
        Returns:
            List of shapes
        """
        with self._lock:
            return [self._shapes[sid] for sid in self._shape_order]
    
    def get_shape_ids(self) -> List[ShapeID]:
        """
        Get all shape IDs in z-order.
        
        Returns:
            List of shape IDs
        """
        with self._lock:
            return self._shape_order.copy()
    
    def get_shape_count_by_type(self) -> Dict[ShapeType, int]:
        """
        Get count of shapes by type.
        
        Returns:
            Dictionary mapping shape types to counts
        """
        with self._lock:
            counts = {shape_type: 0 for shape_type in ShapeType}
            
            for shape in self._shapes.values():
                counts[shape.type] += 1
                
            return counts
    
    def set_metadata(
        self, 
        key: str,
        value: Any,
        merge_dict: bool = False
    ) -> 'Scene':
        """
        Set metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
            merge_dict: If True and both existing and new values are dicts,
                        merge them instead of replacing
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            old_value = self._metadata.get(key)
            
            # Handle dictionary merging
            if (merge_dict and 
                isinstance(old_value, dict) and 
                isinstance(value, dict)):
                new_value = old_value.copy()
                new_value.update(value)
            else:
                new_value = value
                
            # Skip if unchanged
            if old_value == new_value:
                return self
                
            # Record history
            self._record_history('set_metadata', {
                'key': key,
                'old_value': old_value,
                'new_value': new_value
            })
            
            # Update metadata
            self._metadata[key] = new_value
            
        return self
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        with self._lock:
            return self._metadata.get(key, default)
    
    def remove_metadata(self, key: str) -> 'Scene':
        """
        Remove metadata value.
        
        Args:
            key: Metadata key to remove
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            if key in self._metadata:
                # Record history
                self._record_history('remove_metadata', {
                    'key': key,
                    'value': self._metadata[key]
                })
                
                # Remove metadata
                del self._metadata[key]
                
        return self
    
    def to_svg(self) -> ET.Element:
        """
        Convert scene to SVG element.
        
        Returns:
            XML element representing the scene
        """
        with self._lock:
            with Profiler("scene_to_svg"):
                # Create SVG root element
                svg = ET.Element('svg')
                svg.set('width', str(self._width))
                svg.set('height', str(self._height))
                svg.set('viewBox', f"0 0 {self._width} {self._height}")
                svg.set('xmlns', "http://www.w3.org/2000/svg")
                
                # Add background if specified
                if self._background is not None:
                    bg = ET.SubElement(svg, 'rect')
                    bg.set('width', str(self._width))
                    bg.set('height', str(self._height))
                    bg.set('fill', self._background.to_svg_string())
                    
                # Add shapes in z-order
                for shape_id in self._shape_order:
                    shape = self._shapes[shape_id]
                    svg.append(shape.to_svg_element())
                    
                return svg
    
    def to_svg_string(self) -> str:
        """
        Convert scene to SVG string.
        
        Returns:
            SVG string representation
        """
        with self._lock:
            # Use cached value if available
            if self._svg_string_cache is not None:
                return self._svg_string_cache
                
            with Profiler("scene_to_svg_string"):
                svg_element = self.to_svg()
                
                # Convert to string
                from xml.etree.ElementTree import tostring
                svg_string = tostring(svg_element, encoding='unicode')
                
                # Add XML declaration
                svg_string = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + svg_string
                
                # Cache result
                self._svg_string_cache = svg_string
                
                return svg_string
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert scene to dictionary representation.
        
        Returns:
            Dictionary with scene data
        """
        with self._lock:
            return {
                'id': self._id,
                'width': self._width,
                'height': self._height,
                'background': str(self._background) if self._background else None,
                'shapes': [self._shapes[sid].to_dict() for sid in self._shape_order],
                'viewport': self._viewport,
                'metadata': self._metadata
            }
    
    def to_json(self) -> str:
        """
        Convert scene to JSON string.
        
        Returns:
            JSON string representation
        """
        with self._lock:
            scene_dict = self.to_dict()
            return json.dumps(scene_dict, indent=2)
    
    def save(self, filepath: str) -> 'Scene':
        """
        Save scene to a file.
        
        Args:
            filepath: Path to save file
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            try:
                # Determine file format from extension
                ext = os.path.splitext(filepath)[1].lower()
                
                if ext == '.svg':
                    # Save as SVG
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(self.to_svg_string())
                else:
                    # Default to JSON format
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(self.to_json())
                        
                # Trigger event
                self._trigger_event(SceneEvent.SCENE_SAVED, filepath)
                
            except Exception as e:
                logger.error(f"Error saving scene to {filepath}: {str(e)}")
                raise SceneError(f"Error saving scene: {str(e)}")
                
        return self
    
    @staticmethod
    def load(filepath: str) -> 'Scene':
        """
        Load scene from a file.
        
        Args:
            filepath: Path to load file from
            
        Returns:
            Loaded scene
            
        Raises:
            SceneError: If file cannot be loaded
        """
        try:
            # Determine file format from extension
            ext = os.path.splitext(filepath)[1].lower()
            
            if ext == '.svg':
                # Load from SVG
                from models.shape import parse_svg_document
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    svg_content = f.read()
                    
                # Parse SVG
                root_group = parse_svg_document(svg_content)
                
                # Extract dimensions from root group data
                data = root_group.data
                
                # Parse viewBox if available
                width = DEFAULT_WIDTH
                height = DEFAULT_HEIGHT
                
                if 'viewBox' in data:
                    try:
                        viewbox_parts = data['viewBox'].split()
                        if len(viewbox_parts) == 4:
                            width = float(viewbox_parts[2])
                            height = float(viewbox_parts[3])
                    except (ValueError, IndexError):
                        pass
                        
                # Use width/height attributes if available
                if 'width' in data and data['width'] is not None:
                    try:
                        width = float(data['width'])
                    except ValueError:
                        pass
                        
                if 'height' in data and data['height'] is not None:
                    try:
                        height = float(data['height'])
                    except ValueError:
                        pass
                        
                # Create scene
                scene = Scene(width=width, height=height)
                
                # Add shapes from root group
                scene.add_shapes(root_group.shapes)
                
            else:
                # Load from JSON
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Create scene with basic properties
                scene = Scene(
                    scene_id=data.get('id'),
                    width=data.get('width', DEFAULT_WIDTH),
                    height=data.get('height', DEFAULT_HEIGHT),
                    background=data.get('background'),
                    metadata=data.get('metadata', {}),
                    viewport=data.get('viewport')
                )
                
                # Load shapes
                from models.shape import create_shape_from_dict
                
                shapes = []
                for shape_data in data.get('shapes', []):
                    try:
                        shape = create_shape_from_dict(shape_data)
                        shapes.append(shape)
                    except Exception as e:
                        logger.warning(f"Error loading shape: {str(e)}")
                        
                if shapes:
                    scene.add_shapes(shapes)
                    
            # Trigger event
            scene._trigger_event(SceneEvent.SCENE_LOADED, filepath)
            
            return scene
            
        except Exception as e:
            logger.error(f"Error loading scene from {filepath}: {str(e)}")
            raise SceneError(f"Error loading scene: {str(e)}")
    
    def undo(self) -> bool:
        """
        Undo the last action.
        
        Returns:
            True if undo was successful
        """
        with self._lock:
            if not self.can_undo:
                return False
                
            # Move to previous history entry
            self._history_position -= 1
            
            # Apply the inverse operation
            self._apply_history_entry(
                self._history[self._history_position],
                is_undo=True
            )
            
            # Invalidate caches
            self._invalidate_caches()
            
            return True
    
    def redo(self) -> bool:
        """
        Redo the last undone action.
        
        Returns:
            True if redo was successful
        """
        with self._lock:
            if not self.can_redo:
                return False
                
            # Apply the current history entry
            self._apply_history_entry(
                self._history[self._history_position],
                is_undo=False
            )
            
            # Move to next history entry
            self._history_position += 1
            
            # Invalidate caches
            self._invalidate_caches()
            
            return True
    
    def add_event_handler(
        self,
        event: SceneEvent,
        handler: Callable
    ) -> 'Scene':
        """
        Add an event handler.
        
        Args:
            event: Event to handle
            handler: Handler function
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            self._event_handlers[event].append(handler)
            
        return self
    
    def remove_event_handler(
        self,
        event: SceneEvent,
        handler: Callable
    ) -> 'Scene':
        """
        Remove an event handler.
        
        Args:
            event: Event to remove handler for
            handler: Handler function to remove
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            if event in self._event_handlers:
                try:
                    self._event_handlers[event].remove(handler)
                except ValueError:
                    pass
                    
        return self
    
    def _trigger_event(self, event: SceneEvent, *args) -> None:
        """
        Trigger an event.
        
        Args:
            event: Event to trigger
            *args: Arguments to pass to handlers
        """
        # Get handlers (copying to avoid issues if handlers modify the list)
        handlers = self._event_handlers[event].copy()
        
        # Call handlers
        for handler in handlers:
            try:
                handler(self, event, *args)
            except Exception as e:
                logger.error(f"Error in event handler for {event}: {str(e)}")
    
    def _record_history(self, operation: str, data: Dict[str, Any]) -> None:
        """
        Record an operation in the history.
        
        Args:
            operation: Operation name
            data: Operation data
        """
        if not self._recording_history:
            return
            
        with self._lock:
            # If we're not at the latest history entry, truncate history
            if self._history_position < len(self._history) - 1:
                self._history = self._history[:self._history_position + 1]
                
            # Add new history entry
            self._history.append({
                'operation': operation,
                'data': data,
                'timestamp': time.time()
            })
            
            # Update position
            self._history_position = len(self._history) - 1
            
            # Limit history size
            if len(self._history) > MAX_UNDO_HISTORY:
                self._history = self._history[-MAX_UNDO_HISTORY:]
                self._history_position = len(self._history) - 1
    
    def _apply_history_entry(self, entry: Dict[str, Any], is_undo: bool) -> None:
        """
        Apply a history entry (undo or redo).
        
        Args:
            entry: History entry to apply
            is_undo: True for undo, False for redo
        """
        # Temporarily disable history recording
        old_recording = self._recording_history
        self._recording_history = False
        
        try:
            operation = entry['operation']
            data = entry['data']
            
            if operation == 'add_shape':
                if is_undo:
                    # Undo: Remove the shape
                    shape_id = data['shape_id']
                    if shape_id in self._shapes:
                        shape = self._shapes[shape_id]
                        del self._shapes[shape_id]
                        self._shape_order.remove(shape_id)
                        self._trigger_event(SceneEvent.SHAPE_REMOVED, shape)
                else:
                    # Redo: Add the shape
                    from models.shape import create_shape_from_dict
                    shape = create_shape_from_dict(data['shape_data'])
                    self._shapes[shape.id] = shape
                    self._shape_order.append(shape.id)
                    self._trigger_event(SceneEvent.SHAPE_ADDED, shape)
                    
            elif operation == 'add_shapes':
                if is_undo:
                    # Undo: Remove the shapes
                    for shape_id, shape_data in data['shape_data']:
                        if shape_id in self._shapes:
                            shape = self._shapes[shape_id]
                            del self._shapes[shape_id]
                            self._shape_order.remove(shape_id)
                            self._trigger_event(SceneEvent.SHAPE_REMOVED, shape)
                else:
                    # Redo: Add the shapes
                    from models.shape import create_shape_from_dict
                    for shape_id, shape_data in data['shape_data']:
                        shape = create_shape_from_dict(shape_data)
                        self._shapes[shape.id] = shape
                        self._shape_order.append(shape.id)
                        self._trigger_event(SceneEvent.SHAPE_ADDED, shape)
                        
            elif operation == 'remove_shape':
                if is_undo:
                    # Undo: Restore the shape
                    from models.shape import create_shape_from_dict
                    shape = create_shape_from_dict(data['shape_data'])
                    index = data['shape_index']
                    self._shapes[shape.id] = shape
                    if index < len(self._shape_order):
                        self._shape_order.insert(index, shape.id)
                    else:
                        self._shape_order.append(shape.id)
                    self._trigger_event(SceneEvent.SHAPE_ADDED, shape)
                else:
                    # Redo: Remove the shape
                    shape_id = data['shape_id']
                    if shape_id in self._shapes:
                        shape = self._shapes[shape_id]
                        del self._shapes[shape_id]
                        self._shape_order.remove(shape_id)
                        self._trigger_event(SceneEvent.SHAPE_REMOVED, shape)
                        
            elif operation == 'remove_shapes':
                if is_undo:
                    # Undo: Restore the shapes
                    from models.shape import create_shape_from_dict
                    for shape_id, shape_data, index in data['shape_data']:
                        shape = create_shape_from_dict(shape_data)
                        self._shapes[shape.id] = shape
                        if index < len(self._shape_order):
                            self._shape_order.insert(index, shape.id)
                        else:
                            self._shape_order.append(shape.id)
                        self._trigger_event(SceneEvent.SHAPE_ADDED, shape)
                else:
                    # Redo: Remove the shapes
                    for shape_id, shape_data, _ in data['shape_data']:
                        if shape_id in self._shapes:
                            shape = self._shapes[shape_id]
                            del self._shapes[shape_id]
                            self._shape_order.remove(shape_id)
                            self._trigger_event(SceneEvent.SHAPE_REMOVED, shape)
                            
            elif operation == 'update_shape':
                if is_undo:
                    # Undo: Restore the old shape
                    from models.shape import create_shape_from_dict
                    old_shape = create_shape_from_dict(data['old_shape_data'])
                    shape_id = data['shape_id']
                    if shape_id in self._shapes:
                        new_shape = self._shapes[shape_id]
                        self._shapes[old_shape.id] = old_shape
                        
                        # Update shape order if ID changed
                        if shape_id != old_shape.id:
                            idx = self._shape_order.index(shape_id)
                            self._shape_order[idx] = old_shape.id
                            
                        self._trigger_event(
                            SceneEvent.SHAPE_UPDATED,
                            new_shape,
                            old_shape
                        )
                else:
                    # Redo: Apply the new shape
                    from models.shape import create_shape_from_dict
                    new_shape = create_shape_from_dict(data['new_shape_data'])
                    shape_id = data['shape_id']
                    if shape_id in self._shapes:
                        old_shape = self._shapes[shape_id]
                        self._shapes[new_shape.id] = new_shape
                        
                        # Update shape order if ID changed
                        if shape_id != new_shape.id:
                            idx = self._shape_order.index(shape_id)
                            self._shape_order[idx] = new_shape.id
                            del self._shapes[shape_id]
                            
                        self._trigger_event(
                            SceneEvent.SHAPE_UPDATED,
                            old_shape,
                            new_shape
                        )
                        
            elif operation == 'update_shapes':
                if is_undo:
                    # Undo: Restore the old shapes
                    from models.shape import create_shape_from_dict
                    for shape_id, old_data, _ in data['update_data']:
                        old_shape = create_shape_from_dict(old_data)
                        if shape_id in self._shapes:
                            new_shape = self._shapes[shape_id]
                            self._shapes[old_shape.id] = old_shape
                            
                            # Update shape order if ID changed
                            if shape_id != old_shape.id:
                                idx = self._shape_order.index(shape_id)
                                self._shape_order[idx] = old_shape.id
                                
                            self._trigger_event(
                                SceneEvent.SHAPE_UPDATED,
                                new_shape,
                                old_shape
                            )
                else:
                    # Redo: Apply the new shapes
                    from models.shape import create_shape_from_dict
                    for shape_id, _, new_data in data['update_data']:
                        new_shape = create_shape_from_dict(new_data)
                        if shape_id in self._shapes:
                            old_shape = self._shapes[shape_id]
                            self._shapes[new_shape.id] = new_shape
                            
                            # Update shape order if ID changed
                            if shape_id != new_shape.id:
                                idx = self._shape_order.index(shape_id)
                                self._shape_order[idx] = new_shape.id
                                del self._shapes[shape_id]
                                
                            self._trigger_event(
                                SceneEvent.SHAPE_UPDATED,
                                old_shape,
                                new_shape
                            )
                            
            elif operation == 'clear':
                if is_undo:
                    # Undo: Restore all shapes
                    from models.shape import create_shape_from_dict
                    
                    self._shapes = {}
                    for shape_id, shape_data in data['shape_data']:
                        shape = create_shape_from_dict(shape_data)
                        self._shapes[shape.id] = shape
                        
                    self._shape_order = data['shape_order'].copy()
                    
                    self._trigger_event(SceneEvent.SCENE_LOADED)
                else:
                    # Redo: Clear all shapes
                    self._shapes = {}
                    self._shape_order = []
                    
                    self._trigger_event(SceneEvent.SCENE_CLEARED)
                    
            elif operation == 'reorder_shape':
                shape_id = data['shape_id']
                if shape_id in self._shapes:
                    # Remove shape from current position
                    self._shape_order.remove(shape_id)
                    
                    # Insert at target position
                    target_index = data['old_index'] if is_undo else data['new_index']
                    if target_index < len(self._shape_order):
                        self._shape_order.insert(target_index, shape_id)
                    else:
                        self._shape_order.append(shape_id)
                        
            elif operation == 'set_shape_order':
                # Apply the appropriate order
                new_order = data['old_order'] if is_undo else data['new_order']
                self._shape_order = new_order.copy()
                
            elif operation == 'set_size':
                if is_undo:
                    self._width = data['old_width']
                    self._height = data['old_height']
                else:
                    self._width = data['new_width']
                    self._height = data['new_height']
                    
                # Adjust viewport if needed
                vx, vy, vw, vh = self._viewport
                if vx + vw > self._width or vy + vh > self._height:
                    self._viewport = (
                        vx,
                        vy,
                        min(vw, self._width - vx),
                        min(vh, self._height - vy)
                    )
                    self._trigger_event(SceneEvent.VIEWPORT_CHANGED)
                    
            elif operation == 'set_background':
                if is_undo:
                    old_bg = data['old_background']
                    if isinstance(old_bg, dict):
                        # Convert from dict representation
                        self._background = Color(old_bg['value']) if old_bg else None
                    else:
                        self._background = old_bg
                else:
                    new_bg = data['new_background']
                    if isinstance(new_bg, dict):
                        # Convert from dict representation
                        self._background = Color(new_bg['value']) if new_bg else None
                    else:
                        self._background = new_bg
                        
                self._trigger_event(SceneEvent.BACKGROUND_CHANGED)
                
            elif operation == 'set_viewport':
                if is_undo:
                    self._viewport = data['old_viewport']
                else:
                    self._viewport = data['new_viewport']
                    
                self._trigger_event(SceneEvent.VIEWPORT_CHANGED)
                
            elif operation == 'set_metadata':
                key = data['key']
                if is_undo:
                    if 'old_value' in data:
                        self._metadata[key] = data['old_value']
                    else:
                        if key in self._metadata:
                            del self._metadata[key]
                else:
                    self._metadata[key] = data['new_value']
                    
            elif operation == 'remove_metadata':
                key = data['key']
                if is_undo:
                    self._metadata[key] = data['value']
                else:
                    if key in self._metadata:
                        del self._metadata[key]
                        
        finally:
            # Restore history recording
            self._recording_history = old_recording
    
    def _invalidate_caches(self) -> None:
        """Invalidate cached values."""
        self._bounding_box_cache = None
        self._svg_string_cache = None
        self._hash_cache = None
    
    @staticmethod
    def _bounding_boxes_intersect(
        bbox1: BoundingBox,
        bbox2: BoundingBox
    ) -> bool:
        """
        Check if two bounding boxes intersect.
        
        Args:
            bbox1: First bounding box (x, y, width, height)
            bbox2: Second bounding box (x, y, width, height)
            
        Returns:
            True if bounding boxes intersect
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        return (
            x1 < x2 + w2 and
            x1 + w1 > x2 and
            y1 < y2 + h2 and
            y1 + h1 > y2
        )
    
    def __eq__(self, other: object) -> bool:
        """Check if scenes are equal."""
        if not isinstance(other, Scene):
            return False
            
        with self._lock, other._lock:
            # Compare basic properties
            if (self._width != other._width or
                self._height != other._height or
                self._background != other._background or
                self._viewport != other._viewport or
                len(self._shapes) != len(other._shapes) or
                self._shape_order != other._shape_order):
                return False
                
            # Compare shapes
            for shape_id, shape in self._shapes.items():
                if shape_id not in other._shapes:
                    return False
                if shape != other._shapes[shape_id]:
                    return False
                    
            # Compare metadata (only check keys that are in both)
            common_keys = set(self._metadata.keys()) & set(other._metadata.keys())
            for key in common_keys:
                if self._metadata[key] != other._metadata[key]:
                    return False
                    
            return True
    
    def __hash__(self) -> int:
        """Hash for dictionary keys."""
        with self._lock:
            if self._hash_cache is None:
                # Build a hash from essential properties
                scene_hash = hashlib.md5()
                
                scene_hash.update(f"{self._id}:{self._width}:{self._height}".encode())
                
                if self._background:
                    scene_hash.update(str(self._background).encode())
                    
                # Add shape hashes in order
                for shape_id in self._shape_order:
                    shape = self._shapes[shape_id]
                    scene_hash.update(f"{shape_id}:{hash(shape)}".encode())
                    
                # Convert to integer
                self._hash_cache = int(scene_hash.hexdigest(), 16) % (2**32)
                
            return self._hash_cache
    
    def __str__(self) -> str:
        """String representation."""
        return f"Scene(id={self._id}, shapes={len(self._shapes)})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return (f"Scene(id={self._id}, width={self._width}, "
                f"height={self._height}, shapes={len(self._shapes)})")


class SceneManager:
    """
    Manager for multiple scenes with efficient memory usage.
    
    Provides utilities for loading, saving, and switching between scenes.
    """
    
    def __init__(self):
        """Initialize scene manager."""
        self._scenes: Dict[SceneID, Scene] = {}
        self._active_scene_id: Optional[SceneID] = None
        self._scene_lookup: Dict[str, SceneID] = {}  # Name to ID mapping
        self._lock = threading.RLock()
    
    @property
    def active_scene_id(self) -> Optional[SceneID]:
        """Get the active scene ID."""
        with self._lock:
            return self._active_scene_id
    
    @property
    def active_scene(self) -> Optional[Scene]:
        """Get the active scene."""
        with self._lock:
            scene_id = self._active_scene_id
            if scene_id:
                return self._scenes.get(scene_id)
            return None
    
    @property
    def scene_count(self) -> int:
        """Get the number of managed scenes."""
        with self._lock:
            return len(self._scenes)
    
    @property
    def scene_ids(self) -> List[SceneID]:
        """Get list of all scene IDs."""
        with self._lock:
            return list(self._scenes.keys())
    
    @property
    def scene_names(self) -> Dict[SceneID, str]:
        """Get mapping of scene IDs to names."""
        with self._lock:
            return {
                scene_id: scene.get_metadata('name', scene_id)
                for scene_id, scene in self._scenes.items()
            }
    
    def add_scene(
        self,
        scene: Scene,
        name: Optional[str] = None,
        make_active: bool = False
    ) -> SceneID:
        """
        Add a scene to the manager.
        
        Args:
            scene: Scene to add
            name: Optional name for the scene
            make_active: Whether to make this the active scene
            
        Returns:
            Scene ID
            
        Raises:
            SceneError: If a scene with the same ID already exists
        """
        with self._lock:
            scene_id = scene.id
            
            if scene_id in self._scenes:
                raise SceneError(f"Scene ID already exists: {scene_id}")
                
            # Add scene
            self._scenes[scene_id] = scene
            
            # Add name lookup if provided
            if name:
                scene.set_metadata('name', name)
                self._scene_lookup[name] = scene_id
                
            ## Make active if requested or if first scene
            if make_active or not self._active_scene_id:
                self._active_scene_id = scene_id
                
            return scene_id
    
    def remove_scene(self, scene_id: SceneID) -> bool:
        """
        Remove a scene from the manager.
        
        Args:
            scene_id: ID of scene to remove
            
        Returns:
            True if scene was removed
        """
        with self._lock:
            if scene_id not in self._scenes:
                return False
                
            # Get scene for name lookup
            scene = self._scenes[scene_id]
            name = scene.get_metadata('name')
            
            # Remove scene
            del self._scenes[scene_id]
            
            # Remove from name lookup
            if name and name in self._scene_lookup:
                del self._scene_lookup[name]
                
            # If active scene was removed, set a new active scene
            if self._active_scene_id == scene_id:
                self._active_scene_id = next(iter(self._scenes.keys())) if self._scenes else None
                
            return True
    
    def get_scene(self, scene_id: SceneID) -> Optional[Scene]:
        """
        Get a scene by ID.
        
        Args:
            scene_id: Scene ID to get
            
        Returns:
            Scene or None if not found
        """
        with self._lock:
            return self._scenes.get(scene_id)
    
    def get_scene_by_name(self, name: str) -> Optional[Scene]:
        """
        Get a scene by name.
        
        Args:
            name: Scene name to get
            
        Returns:
            Scene or None if not found
        """
        with self._lock:
            scene_id = self._scene_lookup.get(name)
            if scene_id:
                return self._scenes.get(scene_id)
            return None
    
    def set_active_scene(self, scene_id: SceneID) -> bool:
        """
        Set the active scene.
        
        Args:
            scene_id: ID of scene to make active
            
        Returns:
            True if scene was made active
        """
        with self._lock:
            if scene_id not in self._scenes:
                return False
                
            self._active_scene_id = scene_id
            return True
    
    def create_scene(
        self,
        width: float = DEFAULT_WIDTH,
        height: float = DEFAULT_HEIGHT,
        name: Optional[str] = None,
        make_active: bool = False
    ) -> Scene:
        """
        Create a new scene and add it to the manager.
        
        Args:
            width: Scene width
            height: Scene height
            name: Optional name for the scene
            make_active: Whether to make this the active scene
            
        Returns:
            Created scene
        """
        # Create new scene
        scene = Scene(width=width, height=height)
        
        # Set name if provided
        if name:
            scene.set_metadata('name', name)
            
        # Add to manager
        self.add_scene(scene, name=name, make_active=make_active)
        
        return scene
    
    def duplicate_scene(
        self,
        scene_id: SceneID,
        new_name: Optional[str] = None,
        make_active: bool = False
    ) -> Optional[Scene]:
        """
        Duplicate a scene.
        
        Args:
            scene_id: ID of scene to duplicate
            new_name: Optional name for the new scene
            make_active: Whether to make the new scene active
            
        Returns:
            Duplicated scene or None if original not found
        """
        with self._lock:
            # Get original scene
            original = self.get_scene(scene_id)
            if not original:
                return None
                
            # Generate new name if not provided
            if not new_name:
                original_name = original.get_metadata('name', scene_id)
                new_name = f"{original_name} (Copy)"
                
            # Create scene JSON and load new instance
            scene_json = original.to_json()
            
            # Parse JSON to dictionary
            import json
            scene_data = json.loads(scene_json)
            
            # Create new scene with new ID
            scene = Scene(
                width=scene_data.get('width', DEFAULT_WIDTH),
                height=scene_data.get('height', DEFAULT_HEIGHT),
                background=scene_data.get('background'),
                viewport=scene_data.get('viewport'),
                metadata=scene_data.get('metadata', {})
            )
            
            # Set new name
            scene.set_metadata('name', new_name)
            
            # Add shapes from original
            from models.shape import create_shape_from_dict
            
            shapes = []
            for shape_data in scene_data.get('shapes', []):
                try:
                    shape = create_shape_from_dict(shape_data)
                    shapes.append(shape)
                except Exception as e:
                    logger.warning(f"Error duplicating shape: {str(e)}")
                    
            if shapes:
                scene.add_shapes(shapes)
                
            # Add to manager
            self.add_scene(scene, name=new_name, make_active=make_active)
            
            return scene
    
    def load_scene(
        self,
        filepath: str,
        name: Optional[str] = None,
        make_active: bool = False
    ) -> Scene:
        """
        Load a scene from a file and add it to the manager.
        
        Args:
            filepath: Path to load scene from
            name: Optional name for the scene
            make_active: Whether to make this the active scene
            
        Returns:
            Loaded scene
            
        Raises:
            SceneError: If file cannot be loaded
        """
        # Load scene
        scene = Scene.load(filepath)
        
        # Set name if provided, otherwise use filename
        if name:
            scene.set_metadata('name', name)
        else:
            import os
            filename = os.path.basename(filepath)
            name, _ = os.path.splitext(filename)
            scene.set_metadata('name', name)
            
        # Set file path in metadata
        scene.set_metadata('filepath', filepath)
        
        # Add to manager
        self.add_scene(scene, name=name, make_active=make_active)
        
        return scene
    
    def save_scene(
        self,
        scene_id: SceneID,
        filepath: Optional[str] = None
    ) -> str:
        """
        Save a scene to a file.
        
        Args:
            scene_id: ID of scene to save
            filepath: Path to save to, or None to use previous path
            
        Returns:
            Path the scene was saved to
            
        Raises:
            SceneError: If scene not found or cannot be saved
        """
        with self._lock:
            # Get scene
            scene = self.get_scene(scene_id)
            if not scene:
                raise SceneError(f"Scene not found: {scene_id}")
                
            # Use provided path or get from metadata
            save_path = filepath
            if not save_path:
                save_path = scene.get_metadata('filepath')
                if not save_path:
                    raise SceneError("No filepath provided and no previous path found")
                    
            # Save scene
            scene.save(save_path)
            
            # Update metadata
            scene.set_metadata('filepath', save_path)
            
            return save_path
    
    def save_all_scenes(self, directory: str) -> Dict[SceneID, str]:
        """
        Save all scenes to a directory.
        
        Args:
            directory: Directory to save scenes to
            
        Returns:
            Dictionary mapping scene IDs to filepaths
            
        Raises:
            SceneError: If directory cannot be created
        """
        import os
        
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as e:
                raise SceneError(f"Error creating directory: {str(e)}")
                
        with self._lock:
            result = {}
            
            # Save each scene
            for scene_id, scene in self._scenes.items():
                # Get scene name or use ID
                name = scene.get_metadata('name', scene_id)
                
                # Generate filepath
                filepath = os.path.join(directory, f"{name}.json")
                
                # Save scene
                scene.save(filepath)
                scene.set_metadata('filepath', filepath)
                
                result[scene_id] = filepath
                
            return result
    
    def clear(self) -> None:
        """Remove all scenes from the manager."""
        with self._lock:
            self._scenes = {}
            self._active_scene_id = None
            self._scene_lookup = {}
    
    def __contains__(self, scene_id: SceneID) -> bool:
        """Check if scene is in manager."""
        with self._lock:
            return scene_id in self._scenes
    
    def __len__(self) -> int:
        """Get number of scenes."""
        with self._lock:
            return len(self._scenes)
    
    def __iter__(self) -> Iterator[SceneID]:
        """Iterate over scene IDs."""
        with self._lock:
            return iter(self._scenes.keys())