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
