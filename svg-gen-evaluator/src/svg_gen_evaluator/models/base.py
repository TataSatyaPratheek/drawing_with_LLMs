"""
Base model interface for SVG generation models.
"""
from abc import ABC, abstractmethod


class BaseSVGModel(ABC):
    """
    Abstract base class for SVG generation models.
    
    All SVG generation models should implement this interface to ensure
    compatibility with the evaluation system.
    """
    
    @abstractmethod
    def predict(self, description: str) -> str:
        """
        Generate an SVG based on a textual description.
        
        Args:
            description: Textual description of the image to generate
            
        Returns:
            SVG code as a string
        """
        pass