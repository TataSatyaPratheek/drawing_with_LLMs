"""
SVG Generator Module
=================
This module provides functionality for generating SVG code from scene data.
"""

import os
import logging
from svg_prompt_analyzer.models.scene import Scene

logger = logging.getLogger(__name__)


class SVGGenerator:
    """Class for generating SVG from scene data."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the generator.
        
        Args:
            output_dir: Directory where SVG files will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_svg(self, scene: Scene) -> str:
        """
        Generate SVG code from a scene object.
        
        Args:
            scene: Scene object containing all visual elements
            
        Returns:
            String containing the SVG code
        """
        logger.debug(f"Generating SVG for scene: {scene.id}")
        return scene.get_svg_code()
    
    def save_svg(self, scene: Scene) -> str:
        """
        Generate SVG and save it to a file.
        
        Args:
            scene: Scene object containing all visual elements
            
        Returns:
            Path to the saved SVG file
        """
        svg_code = self.generate_svg(scene)
        
        # Create filename
        filename = f"{scene.id}.svg"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(svg_code)
        
        logger.info(f"SVG saved to {filepath}")
        
        return filepath