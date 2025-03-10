"""
SVG Prompt Analyzer - Validation Package
====================================
This package contains modules for validating scene representations
and ensuring visual coherence.
"""

from svg_prompt_analyzer.validation.visual_validator import (
    SVGValidationError, ValidationResult, SVGValidator, 
    batch_validate, fix_common_issues, validate_svg
)


__all__ = [
    'SVGValidationError', 'ValidationResult', 'SVGValidator',
    'batch_validate', 'fix_common_issues', 'validate_svg'
]