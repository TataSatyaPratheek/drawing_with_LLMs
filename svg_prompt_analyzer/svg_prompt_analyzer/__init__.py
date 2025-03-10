"""
SVG Prompt Analyzer Package
===========================
This package provides tools for generating SVG images from text prompts
using LLMs and reinforcement learning optimization.
"""

from svg_prompt_analyzer.core import CONFIG, memoize, Profiler
from svg_prompt_analyzer.core.memory_manager import MemoryManager
from svg_prompt_analyzer.core.hardware_manager import HardwareManager

# Create simple interface for backwards compatibility
class SVGPromptAnalyzerApp:
    """
    Simple interface for SVG Prompt Analyzer.
    
    This class provides a backwards-compatible interface that leverages
    the enhanced LLM and reinforcement learning capabilities internally.
    """
    
    def __init__(
        self, 
        input_file: str, 
        output_dir: str = "output",
        batch_size: int = 5,
        use_optimization: bool = True,
        optimization_level: int = 1,
        parallel: bool = True,
        config_file: str = None
    ):
        """
        Initialize SVG Prompt Analyzer.
        
        Args:
            input_file: Path to input CSV file with prompts
            output_dir: Directory for output SVG files
            batch_size: Number of prompts to process in each batch
            use_optimization: Whether to use RL optimization
            optimization_level: Optimization level (1-3)
            parallel: Whether to use parallel processing
            config_file: Path to configuration file
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.use_optimization = use_optimization
        self.optimization_level = optimization_level
        self.parallel = parallel
        self.config_file = config_file
        
        # Lazy-loaded enhanced app
        self._app = None
        
    def _get_app(self):
        """Lazy-load the enhanced app."""
        if self._app is None:
            from svg_prompt_analyzer.main import LLMEnhancedSVGApp
            
            # Create the enhanced app
            self._app = LLMEnhancedSVGApp(
                input_file=self.input_file,
                output_dir=self.output_dir,
                config_file=self.config_file,
                batch_size=self.batch_size,
                use_llm=True,  # Always use LLM for enhanced generation
                use_optimization=self.use_optimization,
                optimization_level=self.optimization_level,
                parallel_processing=self.parallel,
                memory_efficient=True
            )
            
        return self._app
        
    def run(self):
        """
        Run SVG generation.
        
        Returns:
            List of result dictionaries
        """
        # Get the enhanced app and run it
        app = self._get_app()
        results = app.run()
        return results


# Make key components available at package level
__all__ = [
    'SVGPromptAnalyzerApp',
    'CONFIG',
    'memoize',
    'Profiler',
    'MemoryManager',
    'HardwareManager'
]