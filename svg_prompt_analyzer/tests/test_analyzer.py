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
