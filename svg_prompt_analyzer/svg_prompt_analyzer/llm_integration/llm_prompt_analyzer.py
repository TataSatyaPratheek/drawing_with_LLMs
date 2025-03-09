"""
Production-grade prompt analyzer for SVG generation.
Provides optimized implementation for analyzing and enhancing text prompts
to improve SVG generation results with memory and performance optimizations.
"""

import re
import json
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from enum import Enum, auto
from dataclasses import dataclass, field

# Import core optimizations
from core import CONFIG, memoize, Profiler
from utils.logger import get_logger, log_function_call

# Import LLM manager for prompt enhancement
from llm_integration.llm_manager import (
    LLMManager, ModelType, default_llm_manager, 
    extract_json_from_text
)

# Configure logger
logger = get_logger(__name__)

# Type aliases
PromptStr = str
JsonDict = Dict[str, Any]


class PromptCategory(Enum):
    """Categories of prompts for SVG generation."""
    SCENE = auto()         # Scene description
    OBJECT = auto()        # Object description
    STYLE = auto()         # Style description
    CONCEPT = auto()       # Abstract concept
    TECHNICAL = auto()     # Technical specification
    AMBIGUOUS = auto()     # Unclear or ambiguous
    COMPLEX = auto()       # Complex or compound
    MINIMAL = auto()       # Minimal or simple


@dataclass
class PromptAnalysis:
    """Analysis result for an SVG generation prompt."""
    
    # Original prompt
    original_prompt: str
    
    # Enhanced prompt
    enhanced_prompt: Optional[str] = None
    
    # Parsed components
    subject: Optional[str] = None
    objects: List[str] = field(default_factory=list)
    attributes: Dict[str, List[str]] = field(default_factory=dict)
    colors: List[str] = field(default_factory=list)
    style_cues: List[str] = field(default_factory=list)
    
    # Categorization
    category: Optional[PromptCategory] = None
    complexity: float = 0.0  # 0.0-1.0
    
    # Potential issues
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Keywords for searching
    keywords: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_time: float = 0.0
    
    def to_dict(self) -> JsonDict:
        """Convert analysis to dictionary."""
        return {
            "original_prompt": self.original_prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "subject": self.subject,
            "objects": self.objects,
            "attributes": self.attributes,
            "colors": self.colors,
            "style_cues": self.style_cues,
            "category": self.category.name if self.category else None,
            "complexity": self.complexity,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "keywords": self.keywords,
            "processing_time": self.processing_time
        }
    
    @classmethod
    def from_dict(cls, data: JsonDict) -> 'PromptAnalysis':
        """Create analysis from dictionary."""
        # Convert category string to enum if present
        category = None
        if 'category' in data and data['category']:
            try:
                category = PromptCategory[data['category']]
            except KeyError:
                pass
                
        # Create instance with basic fields
        analysis = cls(
            original_prompt=data.get('original_prompt', ''),
            enhanced_prompt=data.get('enhanced_prompt'),
            subject=data.get('subject'),
            objects=data.get('objects', []),
            colors=data.get('colors', []),
            style_cues=data.get('style_cues', []),
            category=category,
            complexity=data.get('complexity', 0.0),
            issues=data.get('issues', []),
            suggestions=data.get('suggestions', []),
            keywords=data.get('keywords', []),
            processing_time=data.get('processing_time', 0.0)
        )
        
        # Add attributes if present
        if 'attributes' in data:
            analysis.attributes = data['attributes']
            
        return analysis


class PromptParser:
    """
    Parser for extracting structured information from text prompts.
    
    Uses regex patterns and basic NLP techniques to extract key components
    from text prompts for SVG generation.
    """
    
    # Common color terms for detection
    COLOR_TERMS = [
        # Basic colors
        "red", "green", "blue", "yellow", "orange", "purple", "pink",
        "brown", "black", "white", "gray", "grey", "cyan", "magenta",
        "violet", "indigo", "teal", "maroon", "navy", "olive", "salmon",
        "turquoise", "gold", "silver", "bronze", "copper",
        
        # Color modifiers
        "dark", "light", "pale", "bright", "deep", "vivid",
        "saturated", "desaturated", "muted", "vibrant",
        
        # Color formats
        "rgb", "rgba", "hex", "hsl", "hsla"
    ]
    
    # Common style terms for detection
    STYLE_TERMS = [
        # Art styles
        "minimalist", "abstract", "realistic", "cartoon", "sketch",
        "3d", "flat", "geometric", "organic", "hand-drawn", "pixel art",
        "vector", "isometric", "wireframe", "outline", "silhouette",
        "gradient", "monochrome", "duotone", "watercolor", "oil painting",
        
        # Design styles
        "modern", "retro", "vintage", "futuristic", "classic",
        "industrial", "natural", "clean", "messy", "simple", "complex",
        "detailed", "stylized", "decorative", "functional", "technical",
        
        # Visual effects
        "shadow", "glow", "reflection", "transparency", "glossy", "matte",
        "shiny", "textured", "pattern", "noise", "blur", "sharp"
    ]
    
    def __init__(self):
        """Initialize prompt parser."""
        # Compile regex patterns for performance
        self._color_pattern = self._compile_color_pattern()
        self._style_pattern = self._compile_style_pattern()
        
        # Initialize keyword extractor
        self._keyword_extractor = KeywordExtractor()
    
    @memoize
    def _compile_color_pattern(self) -> re.Pattern:
        """
        Compile regex pattern for color detection.
        
        Returns:
            Compiled regex pattern
        """
        # Color formats: names, hex, rgb/rgba, hsl/hsla
        color_formats = [
            r'#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?',  # Hex colors
            r'rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)',  # RGB
            r'rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*[\d.]+\s*\)',  # RGBA
            r'hsl\(\s*\d+\s*,\s*\d+%?\s*,\s*\d+%?\s*\)',  # HSL
            r'hsla\(\s*\d+\s*,\s*\d+%?\s*,\s*\d+%?\s*,\s*[\d.]+\s*\)'  # HSLA
        ]
        
        # Add color names with word boundaries
        for color in self.COLOR_TERMS:
            color_formats.append(rf'\b{re.escape(color)}\b')
            
        # Build final pattern
        pattern = '|'.join(color_formats)
        return re.compile(pattern, re.IGNORECASE)
    
    @memoize
    def _compile_style_pattern(self) -> re.Pattern:
        """
        Compile regex pattern for style detection.
        
        Returns:
            Compiled regex pattern
        """
        # Build pattern from style terms with word boundaries
        style_terms = [rf'\b{re.escape(style)}\b' for style in self.STYLE_TERMS]
        pattern = '|'.join(style_terms)
        return re.compile(pattern, re.IGNORECASE)
    
    def extract_colors(self, text: str) -> List[str]:
        """
        Extract color references from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of color references
        """
        return [match.group(0).lower() for match in self._color_pattern.finditer(text)]
    
    def extract_style_cues(self, text: str) -> List[str]:
        """
        Extract style cues from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of style cues
        """
        return [match.group(0).lower() for match in self._style_pattern.finditer(text)]
    
    def extract_objects(self, text: str) -> List[str]:
        """
        Extract object references from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of object references
        """
        # Tokenize and extract noun phrases (simplified)
        words = text.lower().split()
        object_candidates = []
        
        # Common object-indicating patterns
        patterns = [
            r'(?:a|an|the)\s+([a-z]+(?:\s+[a-z]+){0,2})',
            r'(?:\d+)\s+([a-z]+(?:\s+[a-z]+){0,2})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            object_candidates.extend(matches)
            
        # Add single words that might be objects (nouns)
        for word in words:
            # Skip common non-object words
            if word in ["a", "an", "the", "with", "in", "on", "of", "and", "or"]:
                continue
                
            # Skip color and style terms
            if word in self.COLOR_TERMS or word in self.STYLE_TERMS:
                continue
                
            # Add as potential object
            if len(word) > 3:  # Skip very short words
                object_candidates.append(word)
                
        # Remove duplicates while preserving order
        seen = set()
        unique_objects = []
        for obj in object_candidates:
            if obj not in seen:
                seen.add(obj)
                unique_objects.append(obj)
                
        return unique_objects[:5]  # Limit to top 5
    
    def extract_attributes(self, text: str, objects: List[str]) -> Dict[str, List[str]]:
        """
        Extract attributes for objects.
        
        Args:
            text: Text to analyze
            objects: List of objects to find attributes for
            
        Returns:
            Dictionary mapping objects to their attributes
        """
        attributes = {}
        
        for obj in objects:
            # Find adjectives near object (simplified)
            obj_pattern = rf'\b((?:\w+\s){{0,2}})({re.escape(obj)})(?:\s+(?:with|that|having|has)\s+([^,.]+))?'
            matches = re.findall(obj_pattern, text.lower())
            
            obj_attributes = []
            
            for match in matches:
                # Check for preceding adjectives
                if match[0]:
                    adj_candidates = match[0].strip().split()
                    obj_attributes.extend(adj_candidates)
                    
                # Check for following attributes
                if len(match) > 2 and match[2]:
                    attr_text = match[2]
                    # Extract key attributes
                    attr_pattern = r'([a-z]+)'
                    attr_candidates = re.findall(attr_pattern, attr_text)
                    obj_attributes.extend(attr_candidates)
                    
            # Remove duplicates
            obj_attributes = list(set(obj_attributes))
            
            # Skip colors and style terms
            obj_attributes = [
                attr for attr in obj_attributes 
                if attr not in self.COLOR_TERMS and attr not in self.STYLE_TERMS
            ]
            
            if obj_attributes:
                attributes[obj] = obj_attributes
                
        return attributes
    
    def extract_subject(self, text: str, objects: List[str]) -> Optional[str]:
        """
        Extract main subject from text.
        
        Args:
            text: Text to analyze
            objects: List of objects to consider
            
        Returns:
            Main subject or None if not found
        """
        if not objects:
            return None
            
        # Simple heuristic: first object or most mentioned
        counts = {}
        for obj in objects:
            counts[obj] = text.lower().count(obj.lower())
            
        # Sort by count (descending) and then by position in original list
        sorted_objects = sorted(
            objects,
            key=lambda x: (counts[x], -objects.index(x)),
            reverse=True
        )
        
        return sorted_objects[0] if sorted_objects else None
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords for search and matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of keywords
        """
        return self._keyword_extractor.extract_keywords(text)


class KeywordExtractor:
    """Extracts keywords from text for search and matching."""
    
    def __init__(self):
        """Initialize keyword extractor."""
        # Common stop words to exclude
        self.stop_words = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "from", "by", "on", "off", "for", "in", "out", "over", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "any", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "can", "will", "just", "should", "now"
        }
    
    @memoize
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Tokenize and clean text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stop words
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        # Sort by frequency (descending)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, _ in sorted_words[:max_keywords]]


class PromptAnalyzer:
    """
    Production-grade prompt analyzer for SVG generation.
    
    Analyzes and enhances text prompts for SVG generation using
    a combination of rule-based techniques and LLM-based enhancement.
    """
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        use_llm_enhancement: bool = True,
        cache_results: bool = True,
        cache_size: int = 100
    ):
        """
        Initialize prompt analyzer.
        
        Args:
            llm_manager: LLM manager for advanced analysis
            use_llm_enhancement: Whether to use LLM for enhancement
            cache_results: Whether to cache analysis results
            cache_size: Maximum cache size
        """
        # Set up LLM manager
        self.llm_manager = llm_manager or default_llm_manager
        self.use_llm_enhancement = use_llm_enhancement
        
        # Set up parser
        self.parser = PromptParser()
        
        # Set up cache
        self.cache_results = cache_results
        self._cache = {}
        self._cache_size = cache_size
    
    @log_function_call()
    def analyze(self, prompt: str) -> PromptAnalysis:
        """
        Analyze prompt for SVG generation.
        
        Args:
            prompt: Text prompt to analyze
            
        Returns:
            Prompt analysis results
        """
        # Check cache
        if self.cache_results:
            cache_key = self._create_cache_key(prompt)
            if cache_key in self._cache:
                return self._cache[cache_key]
                
        start_time = time.time()
        
        with Profiler("prompt_analysis"):
            # Create basic analysis
            analysis = PromptAnalysis(original_prompt=prompt)
            
            # Extract basic components
            analysis.colors = self.parser.extract_colors(prompt)
            analysis.style_cues = self.parser.extract_style_cues(prompt)
            analysis.objects = self.parser.extract_objects(prompt)
            analysis.attributes = self.parser.extract_attributes(prompt, analysis.objects)
            analysis.subject = self.parser.extract_subject(prompt, analysis.objects)
            analysis.keywords = self.parser.extract_keywords(prompt)
            
            # Categorize prompt
            analysis.category = self._categorize_prompt(prompt, analysis)
            
            # Calculate complexity
            analysis.complexity = self._calculate_complexity(prompt, analysis)
            
            # Identify issues and suggestions
            analysis.issues, analysis.suggestions = self._identify_issues(prompt, analysis)
            
            # Enhanced prompt (if enabled)
            if self.use_llm_enhancement:
                analysis.enhanced_prompt = self._enhance_prompt(prompt, analysis)
                
            # Record processing time
            analysis.processing_time = time.time() - start_time
            
            # Cache result
            if self.cache_results:
                self._cache_result(cache_key, analysis)
                
            return analysis
    
    @log_function_call()
    def batch_analyze(self, prompts: List[str]) -> List[PromptAnalysis]:
        """
        Analyze multiple prompts in batch.
        
        Args:
            prompts: List of prompts to analyze
            
        Returns:
            List of analysis results
        """
        results = []
        
        for prompt in prompts:
            results.append(self.analyze(prompt))
            
        return results
    
    @log_function_call()
    def enhance_prompt(
        self,
        prompt: str,
        target_tokens: int = 150,
        add_details: bool = True,
        add_style: bool = True
    ) -> str:
        """
        Enhance prompt for better SVG generation.
        
        Args:
            prompt: Original prompt
            target_tokens: Target token length for enhancement
            add_details: Whether to add details
            add_style: Whether to add style suggestions
            
        Returns:
            Enhanced prompt
        """
        with Profiler("prompt_enhancement"):
            # Analyze prompt
            analysis = self.analyze(prompt)
            
            # Use enhanced prompt if already generated
            if analysis.enhanced_prompt:
                return analysis.enhanced_prompt
                
            # Generate enhanced prompt with LLM
            system_prompt = """You are an expert SVG designer. Your task is to enhance user prompts for SVG image generation. 
            Clarify vague terms, add relevant details, and suggest specific visual styles. Be concise but specific. 
            Focus on visual elements that can be represented in SVG. Return only the enhanced prompt text without explanations."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Enhance this prompt for SVG generation. Keep the core meaning but add visual details and clarity:
                
                Original prompt: "{prompt}"
                
                - Target length: ~{target_tokens} tokens
                - Add descriptive details: {'Yes' if add_details else 'No'}
                - Add style suggestions: {'Yes' if add_style else 'No'}
                - Focus on what can be represented in SVG
                
                Enhanced prompt:"""}
            ]
            
            try:
                response = self.llm_manager.complete_sync(
                    messages,
                    model_type=ModelType.GPT_3_5_TURBO,
                    temperature=0.7,
                    max_tokens=300
                )
                
                enhanced = response.get("content", "").strip()
                
                # Clean up response format if needed
                if enhanced.startswith('"') and enhanced.endswith('"'):
                    enhanced = enhanced[1:-1]
                    
                # Fall back to original if enhancement failed
                if not enhanced:
                    enhanced = prompt
                    
                return enhanced
                
            except Exception as e:
                logger.error(f"Error enhancing prompt: {str(e)}")
                return prompt
    
    def get_structured_prompt(self, text: str) -> JsonDict:
        """
        Convert text prompt to structured format.
        
        Args:
            text: Text prompt
            
        Returns:
            Structured prompt as JSON dictionary
        """
        with Profiler("structured_prompt"):
            # Analyze prompt
            analysis = self.analyze(text)
            
            # Build structured prompt
            structured = {
                "subject": analysis.subject or "",
                "elements": [],
                "style": {
                    "colors": analysis.colors,
                    "style_cues": analysis.style_cues
                }
            }
            
            # Add elements based on objects
            for obj in analysis.objects:
                element = {
                    "type": obj,
                    "attributes": analysis.attributes.get(obj, [])
                }
                structured["elements"].append(element)
                
            return structured
    
    def _create_cache_key(self, prompt: str) -> str:
        """
        Create cache key for a prompt.
        
        Args:
            prompt: Prompt to create key for
            
        Returns:
            Cache key
        """
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()
    
    def _cache_result(self, key: str, analysis: PromptAnalysis) -> None:
        """
        Cache analysis result.
        
        Args:
            key: Cache key
            analysis: Analysis result to cache
        """
        # Add to cache
        self._cache[key] = analysis
        
        # Prune cache if needed
        if len(self._cache) > self._cache_size:
            # Remove random 25% of entries (simple approach)
            keys_to_remove = list(self._cache.keys())[:len(self._cache) // 4]
            for k in keys_to_remove:
                del self._cache[k]
    
    def _categorize_prompt(self, prompt: str, analysis: PromptAnalysis) -> PromptCategory:
        """
        Categorize prompt.
        
        Args:
            prompt: Original prompt
            analysis: Current analysis results
            
        Returns:
            Prompt category
        """
        # Check for scene indicators
        scene_indicators = ["scene", "landscape", "background", "environment"]
        if any(indicator in prompt.lower() for indicator in scene_indicators):
            return PromptCategory.SCENE
            
        # Check for style focus
        if len(analysis.style_cues) > 3:
            return PromptCategory.STYLE
            
        # Check for technical specifications
        technical_indicators = ["pixels", "resolution", "aspect ratio", "dimensions", "viewbox", "canvas"]
        if any(indicator in prompt.lower() for indicator in technical_indicators):
            return PromptCategory.TECHNICAL
            
        # Check complexity
        word_count = len(prompt.split())
        if word_count < 5:
            return PromptCategory.MINIMAL
        elif word_count > 20:
            return PromptCategory.COMPLEX
            
        # Check for ambiguity
        ambiguous_terms = ["something", "anything", "whatever", "like", "maybe", "sort of", "kind of"]
        if any(term in prompt.lower() for term in ambiguous_terms):
            return PromptCategory.AMBIGUOUS
            
        # Default: if we have objects, it's an object prompt
        if analysis.objects:
            return PromptCategory.OBJECT
            
        # Fallback for abstract concepts
        return PromptCategory.CONCEPT
    
    def _calculate_complexity(self, prompt: str, analysis: PromptAnalysis) -> float:
        """
        Calculate prompt complexity score.
        
        Args:
            prompt: Original prompt
            analysis: Current analysis results
            
        Returns:
            Complexity score (0.0-1.0)
        """
        # Factors affecting complexity
        factors = {
            "length": min(1.0, len(prompt) / 200),  # Length factor
            "objects": min(1.0, len(analysis.objects) / 5),  # Number of objects
            "colors": min(1.0, len(analysis.colors) / 4),  # Number of colors
            "styles": min(1.0, len(analysis.style_cues) / 3),  # Number of style cues
            "attributes": min(1.0, sum(len(attrs) for attrs in analysis.attributes.values()) / 10)  # Attributes
        }
        
        # Additional complexity indicators
        if "gradient" in prompt.lower():
            factors["gradient"] = 0.2
            
        if "pattern" in prompt.lower():
            factors["pattern"] = 0.2
            
        if any(term in prompt.lower() for term in ["3d", "perspective", "isometric"]):
            factors["dimension"] = 0.3
            
        # Calculate weighted score
        weights = {
            "length": 0.1,
            "objects": 0.3,
            "colors": 0.15,
            "styles": 0.15,
            "attributes": 0.2,
            "gradient": 1.0,
            "pattern": 1.0,
            "dimension": 1.0
        }
        
        score = sum(factors.get(factor, 0) * weights.get(factor, 0) for factor in factors)
        
        # Normalize to 0.0-1.0
        normalized_score = min(1.0, score / sum(weights.get(factor, 0) for factor in factors))
        
        return normalized_score
    
    def _identify_issues(
        self,
        prompt: str,
        analysis: PromptAnalysis
    ) -> Tuple[List[str], List[str]]:
        """
        Identify issues and suggestions for prompt.
        
        Args:
            prompt: Original prompt
            analysis: Current analysis results
            
        Returns:
            Tuple of (issues, suggestions)
        """
        issues = []
        suggestions = []
        
        # Check for minimal prompts
        if len(prompt.split()) < 3:
            issues.append("Prompt is too brief")
            suggestions.append("Add more descriptive details")
            
        # Check for missing subject
        if not analysis.subject and not analysis.objects:
            issues.append("No clear subject identified")
            suggestions.append("Specify what should be the main element of the SVG")
            
        # Check for missing colors
        if not analysis.colors:
            issues.append("No color information")
            suggestions.append("Specify colors for better results")
            
        # Check for missing style
        if not analysis.style_cues:
            issues.append("No style information")
            suggestions.append("Add style guidance (e.g., 'minimalist', 'geometric')")
            
        # Check for ambiguity
        ambiguous_terms = ["something", "anything", "whatever", "like", "maybe", "sort of", "kind of"]
        found_ambiguous = [term for term in ambiguous_terms if term in prompt.lower()]
        if found_ambiguous:
            issues.append(f"Ambiguous terms: {', '.join(found_ambiguous)}")
            suggestions.append("Replace ambiguous terms with specific descriptions")
            
        # Check for complex prompts
        if analysis.complexity > 0.8:
            issues.append("Prompt may be too complex for SVG generation")
            suggestions.append("Consider simplifying or breaking into separate components")
            
        # Check for 3D references
        if any(term in prompt.lower() for term in ["3d", "realistic", "photorealistic"]):
            issues.append("SVG format has limitations with 3D and photorealistic effects")
            suggestions.append("Consider focusing on 2D vector-friendly elements")
            
        # Check for animation references
        if any(term in prompt.lower() for term in ["animate", "animation", "moving"]):
            issues.append("Basic SVG generation typically does not include animation")
            suggestions.append("Focus on static elements and apply animation separately")
            
        return issues, suggestions
    
    def _enhance_prompt(self, prompt: str, analysis: PromptAnalysis) -> str:
        """
        Enhance prompt using LLM.
        
        Args:
            prompt: Original prompt
            analysis: Current analysis results
            
        Returns:
            Enhanced prompt or original if enhancement fails
        """
        try:
            return self.enhance_prompt(
                prompt,
                target_tokens=150,
                add_details=True,
                add_style=True
            )
        except Exception as e:
            logger.error(f"Error in LLM prompt enhancement: {str(e)}")
            return prompt
    
    def analyze_with_llm(self, prompt: str) -> JsonDict:
        """
        Perform deep analysis using LLM.
        
        Args:
            prompt: Text prompt to analyze
            
        Returns:
            Detailed analysis as JSON dictionary
        """
        if not self.use_llm_enhancement:
            # Return rule-based analysis if LLM enhancement disabled
            return self.analyze(prompt).to_dict()
            
        with Profiler("llm_prompt_analysis"):
            # System prompt for analysis
            system_prompt = """You are an expert SVG designer analyzing prompts for SVG generation. 
            Extract structured information from the prompt to help generate better SVG images.
            Return your analysis as a JSON object with the following fields:
            - subject: The main subject/focus of the prompt
            - objects: Array of objects/elements mentioned
            - attributes: Object containing attributes for each object
            - colors: Array of color references
            - style_cues: Array of style indicators
            - category: One of "SCENE", "OBJECT", "STYLE", "CONCEPT", "TECHNICAL", "AMBIGUOUS", "COMPLEX", "MINIMAL"
            - complexity: Number from 0.0-1.0 indicating prompt complexity
            - issues: Array of potential issues with the prompt
            - suggestions: Array of suggestions to improve the prompt
            - keywords: Array of relevant keywords for the prompt
            
            Be concise and focus only on visual elements that can be represented in SVG."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Analyze this prompt for SVG generation: "{prompt}"'}
            ]
            
            try:
                response = self.llm_manager.complete_sync(
                    messages,
                    model_type=ModelType.GPT_3_5_TURBO,
                    temperature=0.3,
                    max_tokens=1000
                )
                
                content = response.get("content", "")
                
                # Extract JSON from response
                result = extract_json_from_text(content)
                
                if not result:
                    # Fall back to rule-based analysis
                    logger.warning("LLM did not return valid JSON, falling back to rule-based analysis")
                    return self.analyze(prompt).to_dict()
                    
                # Fill in missing fields with rule-based analysis
                rule_based = self.analyze(prompt).to_dict()
                for key in rule_based:
                    if key not in result or not result[key]:
                        result[key] = rule_based[key]
                        
                return result
                
            except Exception as e:
                logger.error(f"Error in LLM prompt analysis: {str(e)}")
                # Fall back to rule-based analysis
                return self.analyze(prompt).to_dict()


# Create singleton instance for easy import
default_prompt_analyzer = PromptAnalyzer()


# Utility functions for direct use
def analyze_prompt(prompt: str) -> PromptAnalysis:
    """
    Analyze a prompt for SVG generation.
    
    Args:
        prompt: Text prompt to analyze
        
    Returns:
        Prompt analysis
    """
    return default_prompt_analyzer.analyze(prompt)


def enhance_prompt(prompt: str) -> str:
    """
    Enhance a prompt for better SVG generation.
    
    Args:
        prompt: Original prompt
        
    Returns:
        Enhanced prompt
    """
    return default_prompt_analyzer.enhance_prompt(prompt)


def get_structured_prompt(prompt: str) -> JsonDict:
    """
    Convert text prompt to structured format.
    
    Args:
        prompt: Text prompt
        
    Returns:
        Structured prompt as JSON dictionary
    """
    return default_prompt_analyzer.get_structured_prompt(prompt)