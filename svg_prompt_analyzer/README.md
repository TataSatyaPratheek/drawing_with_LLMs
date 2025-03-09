# SVG Prompt Analyzer and Generator

A Python package for analyzing text prompts and generating SVG images based on natural language descriptions.

## Overview

This tool breaks down text prompts into parts of speech, semantic segments, and spatial relationships to create SVG images. The system analyzes the prompt to understand:

- Objects and their types (geometric, nature, clothing, etc.)
- Visual attributes (colors, materials, shapes)
- Spatial relationships between objects
- Scene composition and layout

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/svg-prompt-analyzer.git
cd svg-prompt-analyzer

# Install the package
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Process prompts from a CSV file
svg-analyzer your_prompts.csv --output-dir generated_svgs

# With custom logging level
svg-analyzer your_prompts.csv --output-dir generated_svgs --log-level DEBUG
```

### Python API

```python
from svg_prompt_analyzer import SVGPromptAnalyzerApp

# Initialize the application
app = SVGPromptAnalyzerApp("your_prompts.csv", "output_directory")

# Process all prompts
results = app.run()

# Or process a single prompt
scene = app.analyzer.analyze_prompt("prompt_id", "a starlit night over snow-covered peaks")
svg_path = app.generator.save_svg(scene)
```

## Features

- **Advanced NLP Analysis**: Uses spaCy and NLTK for deep linguistic analysis
- **Comprehensive Visual Object Model**: Categorizes and represents visual elements
- **Spatial Understanding**: Detects relationships like "above", "inside", "around"
- **Scene-Aware Layout**: Optimizes object placement based on scene type
- **Enhanced Material & Texture Rendering**: Rich visualization of materials and textures

## Optimized for CLIP Similarity

The system is specifically designed to generate SVG images that score highly on CLIP similarity metrics, making it ideal for evaluation with the SigLIP SoViT-400m model.
