"""
SVG Prompt Analyzer - Data Models
================================
This package contains data models for representing visual elements,
scenes, and their properties.
"""

from svg_prompt_analyzer.models.color import Color
from svg_prompt_analyzer.models.material import Material
from svg_prompt_analyzer.models.shape import Shape, Attribute
from svg_prompt_analyzer.models.visual_object import VisualObject, ObjectType
from svg_prompt_analyzer.models.scene import Scene
from svg_prompt_analyzer.models.spatial import SpatialRelation

__all__ = [
    "Color",
    "Material",
    "Shape",
    "Attribute",
    "VisualObject",
    "ObjectType",
    "Scene",
    "SpatialRelation"
]