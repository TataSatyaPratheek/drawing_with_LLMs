"""
SVG Prompt Analyzer - Optimization Package
=====================================
This package contains modules for optimizing model performance,
memory usage, and inference speed.
"""

from svg_prompt_analyzer.optimization.model_prunning import (
    ModelPruner, batch_process_tensors, transfer_weights
)


__all__ = [
    "ModelPruner", "batch_process_tensors", "transfer_weights"
]