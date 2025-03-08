"""
SVG Generation Evaluator - A toolkit for evaluating SVG generation models.

This package provides utilities for evaluating models that generate
Scalable Vector Graphics (SVG) from text descriptions, with a particular
focus on Kaggle competition standards.
"""

__version__ = "0.1.0"

from svg_gen_evaluator.core.evaluator import SVGEvaluator
from svg_gen_evaluator.core.clip import CLIPSimilarityCalculator
from svg_gen_evaluator.core.renderer import SVGRenderer
from svg_gen_evaluator.core.validator import SVGValidator
from svg_gen_evaluator.models.base import BaseSVGModel
from svg_gen_evaluator.models.simple import SimpleModel

__all__ = [
    "SVGEvaluator",
    "CLIPSimilarityCalculator",
    "SVGRenderer",
    "SVGValidator",
    "BaseSVGModel",
    "SimpleModel",
]