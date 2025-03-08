"""
Core functionality for SVG generation evaluation.
"""

from svg_gen_evaluator.core.evaluator import SVGEvaluator
from svg_gen_evaluator.core.clip import CLIPSimilarityCalculator
from svg_gen_evaluator.core.renderer import SVGRenderer
from svg_gen_evaluator.core.validator import SVGValidator

__all__ = [
    "SVGEvaluator",
    "CLIPSimilarityCalculator",
    "SVGRenderer",
    "SVGValidator",
]