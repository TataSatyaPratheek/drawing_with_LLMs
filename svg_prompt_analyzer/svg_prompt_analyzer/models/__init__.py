"""
SVG Prompt Analyzer - Data Models
================================
This package contains data models for representing visual elements,
scenes, and their properties.
"""

from svg_prompt_analyzer.models.color import (
    ColorFormat, ColorError, Color, ColorPalette,
    parse_color, blend_colors, generate_gradient_colors,
    get_contrast_color, find_nearest_color
)
from svg_prompt_analyzer.models.shape import (
    ShapeType, ShapeError, Transform, Shape,
    Rect, Circle, Ellipse, Line, Group, 
    create_shape_from_dict, create_shape_from_svg,
    parse_svg_document
    )
from svg_prompt_analyzer.models.visual_object import (
    ObjectCategory, VisualObjectError, VisualObject,
    Rectangle, Circle, Ellipse, Line, create_visual_object
    )
from svg_prompt_analyzer.models.scene import (
    SceneEvent, SceneError, Scene, SceneManager
    )

__all__ = [
    'ColorFormat', 'ColorError', 'Color', 'ColorPalette',
    'parse_color', 'blend_colors', 'generate_gradient_colors',
    'get_contrast_color', 'find_nearest_color',
    'ShapeType', 'ShapeError', 'Transform', 'Shape',
    'Rect', 'Circle', 'Ellipse', 'Line', 'Group',
    'create_shape_from_dict', 'create_shape_from_svg',
    'parse_svg_document',
    'ObjectCategory', 'VisualObjectError', 'VisualObject',
    'Rectangle', 'Circle', 'Ellipse', 'Line', 'create_visual_object',
    'SceneEvent', 'SceneError', 'Scene', 'SceneManager'
]