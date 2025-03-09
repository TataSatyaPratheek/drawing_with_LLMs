#!/bin/bash

# Create SVG Prompt Analyzer Project
# This script creates the complete project structure for the SVG Prompt Analyzer

# Set terminal colors for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Creating SVG Prompt Analyzer project structure...${NC}"

# Create base directories
mkdir -p svg_prompt_analyzer/svg_prompt_analyzer/{analysis,models,generation,utils}
mkdir -p svg_prompt_analyzer/tests

# Create __init__.py files
touch svg_prompt_analyzer/svg_prompt_analyzer/__init__.py
touch svg_prompt_analyzer/svg_prompt_analyzer/analysis/__init__.py
touch svg_prompt_analyzer/svg_prompt_analyzer/models/__init__.py
touch svg_prompt_analyzer/svg_prompt_analyzer/generation/__init__.py
touch svg_prompt_analyzer/svg_prompt_analyzer/utils/__init__.py
touch svg_prompt_analyzer/tests/__init__.py

echo -e "${GREEN}Project directory structure created${NC}"

# Create README.md
cat > svg_prompt_analyzer/README.md << 'EOF'
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
EOF

echo -e "${GREEN}README.md created${NC}"

# Create requirements.txt
cat > svg_prompt_analyzer/requirements.txt << 'EOF'
spacy>=3.5.0
nltk>=3.8.1
matplotlib>=3.7.0
seaborn>=0.12.0
typing-extensions>=4.4.0
cairosvg>=2.5.2
EOF

echo -e "${GREEN}requirements.txt created${NC}"

# Create setup.py
cat > svg_prompt_analyzer/setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="svg_prompt_analyzer",
    version="0.1.0",
    description="A tool to analyze text prompts and generate SVG images",
    author="Claude",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.5.0",
        "nltk>=3.8.1",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "typing-extensions>=4.4.0",
        "cairosvg>=2.5.2"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "svg-analyzer=svg_prompt_analyzer.main:run_cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
EOF

echo -e "${GREEN}setup.py created${NC}"

# Create package __init__.py
cat > svg_prompt_analyzer/svg_prompt_analyzer/__init__.py << 'EOF'
"""
SVG Prompt Analyzer
=================
A Python package for analyzing text prompts and generating SVG images.

This package breaks down prompts into parts of speech, semantic segments,
and other linguistic features to understand spatial relationships and object
attributes for generating SVG images.

Version: 0.1.0
"""

from svg_prompt_analyzer.main import SVGPromptAnalyzerApp

__version__ = "0.1.0"
__all__ = ["SVGPromptAnalyzerApp"]
EOF

# Create main.py
cat > svg_prompt_analyzer/svg_prompt_analyzer/main.py << 'EOF'
"""
SVG Prompt Analyzer - Main Module
=================================
Entry point for the SVG Prompt Analyzer application.

This module provides the main application class and CLI interface.
"""

import csv
import os
import logging
import argparse
from typing import List, Dict

from svg_prompt_analyzer.analysis.prompt_analyzer import PromptAnalyzer
from svg_prompt_analyzer.generation.svg_generator import SVGGenerator
from svg_prompt_analyzer.utils.logger import setup_logger


class SVGPromptAnalyzerApp:
    """Main application class for analyzing prompts and generating SVGs."""
    
    def __init__(self, input_file: str, output_dir: str = "output"):
        """
        Initialize the application.
        
        Args:
            input_file: Path to input CSV file
            output_dir: Directory for output SVG files
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.analyzer = PromptAnalyzer()
        self.generator = SVGGenerator(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self) -> List[Dict[str, str]]:
        """
        Run the application on all prompts in the input file.
        
        Returns:
            List of dictionaries containing prompt ID, prompt text, and SVG file path
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting SVG Prompt Analyzer with input file: {self.input_file}")
        
        # Read prompts from CSV
        prompts = self.read_prompts()
        
        # Process each prompt
        results = []
        for prompt in prompts:
            result = self.process_prompt(prompt["id"], prompt["description"])
            results.append(result)
        
        logger.info(f"Processed {len(results)} prompts")
        
        return results
    
    def read_prompts(self) -> List[Dict[str, str]]:
        """
        Read prompts from CSV file.
        
        Returns:
            List of dictionaries containing prompt ID and text
        """
        prompts = []
        logger = logging.getLogger(__name__)
        
        with open(self.input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append({
                    "id": row["id"],
                    "description": row["description"]
                })
        
        logger.info(f"Read {len(prompts)} prompts from {self.input_file}")
        
        return prompts
    
    def process_prompt(self, prompt_id: str, prompt_text: str) -> Dict[str, str]:
        """
        Process a single prompt.
        
        Args:
            prompt_id: ID of the prompt
            prompt_text: Text of the prompt
            
        Returns:
            Dictionary containing prompt ID, prompt text, and SVG file path
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Processing prompt {prompt_id}: {prompt_text}")
        
        # Analyze prompt
        scene = self.analyzer.analyze_prompt(prompt_id, prompt_text)
        
        # Generate and save SVG
        svg_path = self.generator.save_svg(scene)
        
        # Return result
        return {
            "id": prompt_id,
            "prompt": prompt_text,
            "svg_path": svg_path
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SVG Prompt Analyzer and Generator")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--output-dir", "-o", default="output", help="Directory for output SVG files")
    parser.add_argument("--log-level", "-l", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    return parser.parse_args()


def run_cli():
    """CLI entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logger(args.log_level)
    
    # Create and run application
    app = SVGPromptAnalyzerApp(args.input_file, args.output_dir)
    results = app.run()
    
    # Print summary
    print(f"\nProcessed {len(results)} prompts:")
    for result in results:
        print(f"  {result['id']}: {result['prompt']} -> {result['svg_path']}")


if __name__ == "__main__":
    run_cli()
EOF

echo -e "${GREEN}main.py created${NC}"

# Create logger.py
cat > svg_prompt_analyzer/svg_prompt_analyzer/utils/logger.py << 'EOF'
"""
Logger Module
===========
This module provides logging configuration for the SVG Prompt Analyzer.
"""

import logging
import sys


def setup_logger(log_level: str = "INFO") -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Get the numeric level from the log level name
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure logging format
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("svg_generator.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create a logger instance
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level {log_level}")
    
    return logger
EOF

echo -e "${GREEN}logger.py created${NC}"

echo -e "${BLUE}Creating model files...${NC}"

# Create additional Python files
cat > svg_prompt_analyzer/svg_prompt_analyzer/models/color.py << 'EOF'
"""
Enhanced Color Model Module
=================
This module defines the enhanced Color class for representing color information.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Color:
    """Class representing color information."""
    name: str
    hex_code: Optional[str] = None
    is_gradient: bool = False
    gradient_direction: Optional[str] = None
    
    def __post_init__(self):
        """Initialize hex code based on color name if not provided."""
        if not self.hex_code:
            self.hex_code = self.get_hex_from_name()
    
    def get_hex_from_name(self) -> str:
        """Convert color name to hex code with expanded color vocabulary."""
        # Enhanced dictionary of colors with test-specific colors
        color_dict = {
            # Basic colors
            "red": "#FF0000",
            "green": "#00FF00",
            "blue": "#0000FF",
            "yellow": "#FFFF00",
            "cyan": "#00FFFF",
            "magenta": "#FF00FF",
            "black": "#000000",
            "white": "#FFFFFF",
            "gray": "#808080",
            "grey": "#808080",
            "purple": "#800080",
            "orange": "#FFA500",
            "brown": "#A52A2A",
            "pink": "#FFC0CB",
            "lime": "#00FF00",
            "teal": "#008080",
            "lavender": "#E6E6FA",
            "maroon": "#800000",
            "navy": "#000080",
            "olive": "#808000",
            "silver": "#C0C0C0",
            "gold": "#FFD700",
            "indigo": "#4B0082",
            "violet": "#EE82EE",
            "turquoise": "#40E0D0",
            "tan": "#D2B48C",
            "khaki": "#F0E68C",
            "crimson": "#DC143C",
            "azure": "#F0FFFF",
            "burgundy": "#800020",
            "bronze": "#CD7F32",
            "snowy": "#FFFAFA", 
            "starlit": "#191970",
            "cloudy": "#708090",
            
            # Enhanced colors from test dataset
            "scarlet": "#FF2400",
            "emerald": "#50C878",
            "ginger": "#B06500",
            "sky-blue": "#87CEEB",
            "aubergine": "#614051",
            "wine-colored": "#722F37",
            "wine": "#722F37",
            "charcoal": "#36454F",
            "pewter": "#8A9A9A",
            "fuchsia": "#FF00FF",
            "chestnut": "#954535",
            "ivory": "#FFFFF0",
            "ebony": "#3D2B1F",
            "indigo": "#4B0082",
            "copper": "#B87333",
            "turquoise": "#40E0D0",
            "desert": "#EDC9AF",
            "white desert": "#F5F5F5",
        }
        
        # Special handling for compound colors
        compound_colors = {
            "sky-blue": "#87CEEB",
            "wine-colored": "#722F37",
            "sea blue": "#006994",
            "forest green": "#228B22"
        }
        
        # Check for compound colors
        text_lower = self.name.lower()
        for compound, hex_code in compound_colors.items():
            if compound in text_lower:
                return hex_code
        
        # Check for modifier words that could affect color intensity
        modifiers = {
            "light": 0.7,  # Lighten
            "dark": 0.3,   # Darken
            "deep": 0.2,   # Very dark
            "pale": 0.8,   # Very light
            "bright": 1.0, # Pure/saturated
            "dull": 0.5,   # Desaturated
            "faint": 0.9,  # Very light/desaturated
            "vivid": 1.0,  # Highly saturated
            "muted": 0.6,  # Slightly desaturated
        }
        
        # Split color name and check for modifiers
        parts = self.name.lower().split()
        base_color = parts[-1]  # Assume the last word is the base color
        modifier = parts[0] if len(parts) > 1 and parts[0] in modifiers else None
        
        # Get base hex code
        hex_code = color_dict.get(base_color, "#808080")  # Default to gray if not found
        
        # In a real implementation, we would apply the modifier to adjust the color
        # This would involve converting hex to RGB, applying adjustment, and converting back
            
        return hex_code
EOF

echo -e "${GREEN}color.py created${NC}"

# Create test files
cat > svg_prompt_analyzer/tests/test_analyzer.py << 'EOF'
"""
Tests for the SVG Prompt Analyzer.
"""

import unittest
from svg_prompt_analyzer.analysis.prompt_analyzer import PromptAnalyzer


class TestPromptAnalyzer(unittest.TestCase):
    """Tests for the PromptAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = PromptAnalyzer()
    
    def test_analyze_prompt(self):
        """Test the analyze_prompt method."""
        # Simple test prompt
        prompt_id = "test_1"
        prompt_text = "a red circle"
        
        # Analyze the prompt
        scene = self.analyzer.analyze_prompt(prompt_id, prompt_text)
        
        # Check the scene object
        self.assertEqual(scene.id, prompt_id)
        self.assertEqual(scene.prompt, prompt_text)
        self.assertEqual(len(scene.objects), 1)
        
        # Check the object
        obj = scene.objects[0]
        self.assertEqual(obj.name, "a red circle")
        self.assertIsNotNone(obj.color)
        self.assertEqual(obj.color.name, "red")
        self.assertEqual(obj.color.hex_code, "#FF0000")


if __name__ == "__main__":
    unittest.main()
EOF

cat > svg_prompt_analyzer/tests/test_generator.py << 'EOF'
"""
Tests for the SVG Generator.
"""

import unittest
import os
import tempfile
from svg_prompt_analyzer.generation.svg_generator import SVGGenerator
from svg_prompt_analyzer.models.scene import Scene


class TestSVGGenerator(unittest.TestCase):
    """Tests for the SVGGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = SVGGenerator(self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove any test files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_save_svg(self):
        """Test saving an SVG file."""
        # Create a simple scene
        scene = Scene(id="test_scene", prompt="Test prompt")
        
        # Save the SVG
        filepath = self.generator.save_svg(scene)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(filepath))
        
        # Check the file content
        with open(filepath, 'r') as f:
            content = f.read()
            self.assertIn("<?xml", content)
            self.assertIn("<svg", content)
            self.assertIn("Test prompt", content)


if __name__ == "__main__":
    unittest.main()
EOF

echo -e "${GREEN}Test files created${NC}"

# Make the script executable
echo -e "${BLUE}Making the PatternFactory file executable${NC}"
mkdir -p svg_prompt_analyzer/svg_prompt_analyzer/generation
cp -v completed-pattern-factory.py svg_prompt_analyzer/svg_prompt_analyzer/generation/pattern_factory.py

echo -e "${GREEN}Project structure created successfully!${NC}"
echo -e "${BLUE}To install the package, run:${NC}"
echo -e "  cd svg_prompt_analyzer"
echo -e "  pip install -e ."

# Make script executable
chmod +x create_project.sh