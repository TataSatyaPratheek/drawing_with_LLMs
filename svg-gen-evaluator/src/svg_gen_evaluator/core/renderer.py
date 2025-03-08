"""
SVG rendering utilities to convert SVG code to images.
"""
import io
import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)


class SVGRenderer:
    """
    Renders SVG code to PIL Images for visual inspection and CLIP evaluation.
    """
    
    def __init__(self, default_size: tuple = (512, 512)):
        """
        Initialize the SVG renderer.
        
        Args:
            default_size: Default size (width, height) for rendered images
        """
        self.default_size = default_size
        
        # Verify cairosvg is installed
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if necessary dependencies are installed."""
        try:
            import cairosvg
            self.cairosvg = cairosvg
        except ImportError:
            logger.error("cairosvg package not installed. Installing...")
            import subprocess
            import sys
            
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cairosvg"])
            import cairosvg
            self.cairosvg = cairosvg
    
    def render_svg(
        self, 
        svg_code: str, 
        output_path: Optional[Union[str, Path]] = None,
        size: Optional[tuple] = None
    ) -> Image.Image:
        """
        Convert SVG code to a PIL Image.
        
        Args:
            svg_code: SVG code as a string
            output_path: Optional path to save the rendered image
            size: Optional (width, height) tuple for rendered image
            
        Returns:
            PIL Image of the rendered SVG
        """
        try:
            # Convert SVG to PNG using cairosvg
            width, height = size or self.default_size
            png_data = self.cairosvg.svg2png(
                bytestring=svg_code.encode('utf-8'),
                output_width=width,
                output_height=height
            )
            
            # Create PIL Image from PNG data
            image = Image.open(io.BytesIO(png_data))
            
            # Save image if output path is provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(exist_ok=True, parents=True)
                image.save(output_path)
            
            return image
        
        except Exception as e:
            logger.error(f"Error rendering SVG: {e}")
            logger.debug(f"Problematic SVG code: {svg_code[:100]}...")
            
            # Return a blank image as fallback
            blank_image = Image.new('RGB', size or self.default_size, color='white')
            
            if output_path:
                blank_image.save(output_path)
                
            return blank_image
    
    def render_svg_file(
        self, 
        svg_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        size: Optional[tuple] = None
    ) -> Image.Image:
        """
        Render an SVG file to a PIL Image.
        
        Args:
            svg_path: Path to the SVG file
            output_path: Optional path to save the rendered image
            size: Optional (width, height) tuple for rendered image
            
        Returns:
            PIL Image of the rendered SVG
        """
        svg_path = Path(svg_path)
        
        if not svg_path.exists():
            raise FileNotFoundError(f"SVG file not found: {svg_path}")
        
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_code = f.read()
        
        return self.render_svg(svg_code, output_path, size)