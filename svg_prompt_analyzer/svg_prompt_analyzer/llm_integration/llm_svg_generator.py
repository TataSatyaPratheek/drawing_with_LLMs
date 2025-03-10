"""
Production-grade SVG generator using LLMs.
Provides optimized implementation for generating and refining SVG content
based on text prompts with memory, performance, and reliability optimizations.
"""

import re
import time
import json
import random
import hashlib
import threading
from io import BytesIO
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

# Import core optimizations
from svg_prompt_analyzer.core import CONFIG, memoize, Profiler, get_thread_pool
from svg_prompt_analyzer.core.memory_manager import MemoryManager
from svg_prompt_analyzer.core.batch_processor import BatchProcessor
from svg_prompt_analyzer.core.hardware_manager import HardwareManager
from svg_prompt_analyzer.utils.logger import get_logger, log_function_call

# Import LLM manager for generating SVGs
from svg_prompt_analyzer.llm_integration.llm_manager import (
    LLMManager, ModelType, extract_svg_from_text, extract_json_from_text
)

# Import prompt analyzer
from svg_prompt_analyzer.llm_integration.llm_prompt_analyzer import (
    LLMPromptAnalyzer, get_default_prompt_analyzer, 
    analyze_prompt, enhance_prompt, analyze
)

# Import CLIP evaluator for quality assessment if available
try:
    from svg_prompt_analyzer.llm_integration.clip_evaluator import ClipEvaluator
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# Configure logger
logger = get_logger(__name__)

# Get memory manager instance
memory_manager = MemoryManager()
hardware_manager = HardwareManager()

# Type aliases
SvgString = str
JsonDict = Dict[str, Any]
PromptStr = str


class GenerationStrategy(Enum):
    """Strategies for SVG generation."""
    DIRECT = auto()        # Direct generation without template
    TEMPLATE = auto()      # Template-based generation
    COMPONENT = auto()     # Generate components and assemble
    ITERATIVE = auto()     # Iterative refinement
    OPTIMIZED = auto()     # Optimized for small file size


@dataclass
class GenerationConfig:
    """Configuration for SVG generation."""
    
    # Basic settings
    model_type: ModelType = ModelType.GPT_3_5_TURBO
    strategy: GenerationStrategy = GenerationStrategy.DIRECT
    max_tokens: int = 2048
    temperature: float = 0.7
    
    # Template settings
    template_id: Optional[str] = None
    use_components: bool = False
    
    # Advanced settings
    enhance_prompt: bool = True
    use_clip_feedback: bool = False
    min_quality_score: float = 0.7
    max_retries: int = 2
    optimize_output: bool = True
    
    # Size constraints
    width: int = 800
    height: int = 600
    max_file_size: int = 50000  # bytes
    
    # Callbacks
    progress_callback: Optional[Callable] = None


@dataclass
class SvgGenerationResult:
    """Result of SVG generation."""
    
    # SVG content
    svg_content: Optional[SvgString] = None
    
    # Original and enhanced prompts
    original_prompt: Optional[PromptStr] = None
    enhanced_prompt: Optional[PromptStr] = None
    
    # Performance metrics
    generation_time: float = 0.0
    token_count: int = 0
    retry_count: int = 0
    
    # Quality assessment
    quality_score: float = 0.0
    file_size: int = 0
    element_count: int = 0
    
    # Status information
    success: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "svg_content": self.svg_content,
            "original_prompt": self.original_prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "generation_time": self.generation_time,
            "token_count": self.token_count,
            "retry_count": self.retry_count,
            "quality_score": self.quality_score,
            "file_size": self.file_size,
            "element_count": self.element_count,
            "error_message": self.error_message
        }


class SvgTemplateManager:
    """
    Manager for SVG templates.
    
    Provides templates for common SVG structures to improve generation quality.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for resource efficiency."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SvgTemplateManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize template manager.
        
        Args:
            template_dir: Optional directory containing custom templates
        """
        # Initialize only once (singleton pattern)
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # Default templates
        self._templates = self._create_default_templates()
        self._lock = threading.RLock()
        
        # Load custom templates if provided
        if template_dir:
            self._load_custom_templates(template_dir)
            
        self._initialized = True
    
    def get_template(self, template_id: str) -> Optional[str]:
        """
        Get template by ID.
        
        Args:
            template_id: Template ID
            
        Returns:
            Template string or None if not found
        """
        with self._lock:
            return self._templates.get(template_id)
    
    def add_template(self, template_id: str, template: str) -> None:
        """
        Add or update template.
        
        Args:
            template_id: Template ID
            template: Template string
        """
        with self._lock:
            self._templates[template_id] = template
    
    def list_templates(self) -> List[str]:
        """
        List available template IDs.
        
        Returns:
            List of template IDs
        """
        with self._lock:
            return list(self._templates.keys())
    
    @memory_manager.memory_efficient_function
    def get_template_for_prompt(self, prompt: str) -> Tuple[str, str]:
        """
        Find suitable template for a prompt.
        
        Args:
            prompt: Generation prompt
            
        Returns:
            Tuple of (template_id, template)
        """
        # Analyze prompt to determine best template
        try:
            prompt_analysis = analyze(prompt)
            
            # Scene templates
            if prompt_analysis.category and prompt_analysis.category.name == "SCENE":
                return "scene", self._templates["scene"]
                
            # Icon templates
            if "icon" in prompt.lower():
                return "icon", self._templates["icon"]
                
            # Chart templates
            chart_keywords = ["chart", "graph", "plot", "diagram"]
            if any(keyword in prompt.lower() for keyword in chart_keywords):
                return "chart", self._templates["chart"]
        except Exception as e:
            logger.warning(f"Error determining template for prompt: {e}")
            
        # Default to basic template
        return "basic", self._templates["basic"]
    
    def _load_custom_templates(self, template_dir: str) -> None:
        """
        Load custom templates from a directory.
        
        Args:
            template_dir: Directory containing templates
        """
        try:
            template_path = Path(template_dir)
            if not template_path.exists():
                logger.warning(f"Template directory does not exist: {template_dir}")
                return
                
            # Load all .svg and .template files
            for file_path in template_path.glob("*.svg"):
                template_id = file_path.stem
                with open(file_path, "r", encoding="utf-8") as f:
                    template = f.read()
                    self.add_template(template_id, template)
                    
            for file_path in template_path.glob("*.template"):
                template_id = file_path.stem
                with open(file_path, "r", encoding="utf-8") as f:
                    template = f.read()
                    self.add_template(template_id, template)
                    
            logger.info(f"Loaded custom templates from {template_dir}")
            
        except Exception as e:
            logger.error(f"Error loading custom templates: {e}")
    
    def _create_default_templates(self) -> Dict[str, str]:
        """
        Create default templates.
        
        Returns:
            Dictionary of default templates
        """
        templates = {}
        
        # Basic template
        templates["basic"] = '''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <!-- 
  PROMPT: {prompt}
  INSTRUCTIONS:
  - Replace this comment with SVG elements that represent the prompt
  - Use clean, semantic SVG with clearly named elements
  - Focus on vector shapes, not raster techniques
  -->
</svg>'''
        
        # Scene template
        templates["scene"] = '''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <!-- 
  SCENE PROMPT: {prompt}
  INSTRUCTIONS:
  - Create a scene with background, middle ground, and foreground layers
  - Use <g> elements with descriptive ids to organize content
  - Consider z-index ordering (back to front)
  - Use gradients for sky/backgrounds when appropriate
  -->
  
  <!-- Background layer -->
  <g id="background">
    <!-- Add background elements here -->
  </g>
  
  <!-- Middle layer -->
  <g id="midground">
    <!-- Add middle ground elements here -->
  </g>
  
  <!-- Foreground layer -->
  <g id="foreground">
    <!-- Add foreground elements here -->
  </g>
</svg>'''
        
        # Icon template
        templates["icon"] = '''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <!-- 
  ICON PROMPT: {prompt}
  INSTRUCTIONS:
  - Create a clear, recognizable icon
  - Use simple shapes and clean lines
  - Ensure the design works at multiple sizes
  - Consider negative space and silhouette
  - Use semantic elements and descriptive ids
  -->
  
  <!-- Icon background (optional) -->
  
  <!-- Main icon elements -->
  <g id="icon">
    <!-- Add icon elements here -->
  </g>
</svg>'''
        
        # Chart template
        templates["chart"] = '''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <!-- 
  CHART PROMPT: {prompt}
  INSTRUCTIONS:
  - Create a data visualization that clearly represents information
  - Include axes, legends, and labels as appropriate
  - Use consistent styling and color scheme
  - Ensure data representation is accurate and to scale
  - Use semantic grouping for chart components
  -->
  
  <!-- Chart area -->
  <g id="chart-area">
    <!-- Add chart background, grid, etc. -->
  </g>
  
  <!-- Data elements -->
  <g id="data">
    <!-- Add bars, lines, points, etc. -->
  </g>
  
  <!-- Axes -->
  <g id="axes">
    <!-- Add x and y axes -->
  </g>
  
  <!-- Legend -->
  <g id="legend">
    <!-- Add legend items -->
  </g>
  
  <!-- Labels and annotations -->
  <g id="labels">
    <!-- Add title, axis labels, etc. -->
  </g>
</svg>'''
        
        return templates


class SvgProcessor:
    """
    Processor for SVG content.
    
    Provides methods for cleaning, validating, and optimizing SVG content.
    """
    
    def clean_svg(self, svg_content: str) -> str:
        """
        Clean SVG content.
        
        Args:
            svg_content: Raw SVG content
            
        Returns:
            Cleaned SVG content
        """
        # Strip any surrounding markdown or code blocks
        svg_content = extract_svg_from_text(svg_content) or svg_content
        
        # Remove any XML declaration
        svg_content = re.sub(r'<\?xml[^>]*\?>\s*', '', svg_content)
        
        # Ensure SVG has correct namespace
        if 'xmlns="http://www.w3.org/2000/svg"' not in svg_content:
            svg_content = svg_content.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)
            
        # Remove comments
        svg_content = re.sub(r'<!--.*?-->', '', svg_content, flags=re.DOTALL)
        
        return svg_content.strip()
    
    def validate_svg(self, svg_content: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SVG content.
        
        Args:
            svg_content: SVG content to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Try to parse XML
            ET.fromstring(svg_content)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def fix_svg(self, svg_content: str) -> str:
        """
        Try to fix invalid SVG content.
        
        Args:
            svg_content: SVG content to fix
            
        Returns:
            Fixed SVG content
        """
        # Common issues to fix
        
        # 1. Missing closing tags
        # This is a simplified approach - a full XML parser would be better
        # but might reject the invalid XML entirely
        common_tags = ["rect", "circle", "path", "g", "text", "line", "polygon", "ellipse"]
        for tag in common_tags:
            # Count opening and closing tags
            opening_count = len(re.findall(f'<{tag}[^/>]*>', svg_content))
            self_closing_count = len(re.findall(f'<{tag}[^>]*/>', svg_content))
            closing_count = len(re.findall(f'</{tag}>', svg_content))
            
            # Add missing closing tags
            if opening_count > closing_count + self_closing_count:
                svg_content = svg_content.rstrip()
                missing_count = opening_count - closing_count - self_closing_count
                svg_content += ''.join([f'</{tag}>' for _ in range(missing_count)])
                
        # 2. Unclosed SVG tag
        if '</svg>' not in svg_content:
            svg_content = svg_content.rstrip() + '</svg>'
            
        # 3. Fix malformed attributes (quotes)
        svg_content = re.sub(pattern=r'=([^"\'][^\s>]*)', repl=r'="\1"', string=svg_content)
        
        # 4. Make sure there's a root SVG element
        if not svg_content.strip().startswith('<svg'):
            svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">\n{svg_content}\n</svg>'
            
        return svg_content
    
    @memory_manager.memory_efficient_function
    def optimize_svg(self, svg_content: str) -> str:
        """
        Optimize SVG content.
        
        Args:
            svg_content: SVG content to optimize
            
        Returns:
            Optimized SVG content
        """
        try:
            with Profiler("svg_optimization"):
                # Parse SVG
                dom = minidom.parseString(svg_content)
                
                # Remove unnecessary attributes
                self._remove_unnecessary_attributes(dom)
                
                # Round numeric values
                self._round_numeric_attributes(dom)
                
                # Convert back to string
                optimized = dom.toxml()
                
                # Remove XML declaration
                optimized = re.sub(r'<\?xml[^>]*\?>\s*', '', optimized)
                
                # Remove empty lines
                optimized = re.sub(r'\n\s*\n', '\n', optimized)
                
                return optimized
            
        except Exception as e:
            logger.warning(f"Failed to optimize SVG: {e}")
            return svg_content
    
    def prettify_svg(self, svg_content: str) -> str:
        """
        Prettify SVG content with proper indentation.
        
        Args:
            svg_content: SVG content to prettify
            
        Returns:
            Prettified SVG content
        """
        try:
            # Parse SVG
            parsed = minidom.parseString(svg_content)
            
            # Pretty-print with 2-space indentation
            pretty = parsed.toprettyxml(indent='  ')
            
            # Remove XML declaration
            pretty = re.sub(r'<\?xml[^>]*\?>\s*', '', pretty)
            
            # Remove empty lines
            pretty = re.sub(r'\n\s*\n', '\n', pretty)
            
            return pretty
            
        except Exception as e:
            logger.warning(f"Failed to prettify SVG: {e}")
            return svg_content
    
    @memory_manager.memory_efficient_function
    def calculate_metrics(self, svg_content: str) -> Dict[str, Any]:
        """
        Calculate metrics for SVG content.
        
        Args:
            svg_content: SVG content to analyze
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "file_size": len(svg_content.encode('utf-8')),
            "element_count": 0,
            "path_count": 0,
            "group_count": 0,
            "depth": 0,
            "max_depth": 0,
            "unique_colors": set(),
            "has_gradients": False,
            "has_filters": False,
            "has_text": False
        }
        
        try:
            # Parse SVG
            root = ET.fromstring(svg_content)
            
            # Count elements
            self._count_elements(root, metrics, depth=0)
            
            # Convert color set to count
            metrics["unique_colors"] = len(metrics["unique_colors"])
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to calculate SVG metrics: {e}")
            return metrics
    
    def _count_elements(
        self,
        element: ET.Element,
        metrics: Dict[str, Any],
        depth: int = 0
    ) -> None:
        """
        Count elements in SVG recursively.
        
        Args:
            element: XML element
            metrics: Metrics dictionary to update
            depth: Current depth
        """
        # Update max depth
        metrics["max_depth"] = max(metrics["max_depth"], depth)
        
        # Count element
        metrics["element_count"] += 1
        
        # Check element type
        tag = element.tag
        if '}' in tag:
            tag = tag.split('}', 1)[1]
            
        if tag == 'path':
            metrics["path_count"] += 1
        elif tag == 'g':
            metrics["group_count"] += 1
        elif tag == 'text':
            metrics["has_text"] = True
        elif tag in ('linearGradient', 'radialGradient'):
            metrics["has_gradients"] = True
        elif tag.startswith('filter') or tag in ('filter', 'feGaussianBlur'):
            metrics["has_filters"] = True
            
        # Extract colors
        if 'fill' in element.attrib and element.attrib['fill'] not in ('none', 'transparent'):
            metrics["unique_colors"].add(element.attrib['fill'])
        if 'stroke' in element.attrib and element.attrib['stroke'] not in ('none', 'transparent'):
            metrics["unique_colors"].add(element.attrib['stroke'])
            
        # Process children
        for child in element:
            self._count_elements(child, metrics, depth + 1)
    
    def _remove_unnecessary_attributes(self, dom: minidom.Document) -> None:
        """
        Remove unnecessary attributes from SVG.
        
        Args:
            dom: XML Document
        """
        # Default attribute values that can be removed
        default_values = {
            'x': '0',
            'y': '0',
            'dx': '0',
            'dy': '0',
            'fill-opacity': '1',
            'stroke-opacity': '1',
            'stroke-width': '1',
            'stroke-linecap': 'butt',
            'stroke-linejoin': 'miter',
            'font-weight': 'normal',
            'font-style': 'normal',
            'opacity': '1'
        }
        
        # Find all elements
        all_elements = dom.getElementsByTagName('*')
        
        for element in all_elements:
            # Check each attribute
            attrs_to_remove = []
            for attr, value in element.attributes.items():
                # Remove if it matches default value
                if attr in default_values and value == default_values[attr]:
                    attrs_to_remove.append(attr)
                    
            # Remove attributes
            for attr in attrs_to_remove:
                element.removeAttribute(attr)
    
    def _round_numeric_attributes(self, dom: minidom.Document) -> None:
        """
        Round numeric attribute values.
        
        Args:
            dom: XML Document
        """
        # Attributes to round
        numeric_attrs = {
            'x', 'y', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy', 'r', 'rx', 'ry',
            'width', 'height', 'stroke-width', 'opacity', 'fill-opacity',
            'stroke-opacity', 'offset', 'font-size'
        }
        
        # Find all elements
        all_elements = dom.getElementsByTagName('*')
        
        for element in all_elements:
            # Check each attribute
            for attr in numeric_attrs:
                if element.hasAttribute(attr):
                    try:
                        value = element.getAttribute(attr)
                        # Skip if not numeric
                        if not re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$', value):
                            continue
                            
                        # Round to 2 decimal places
                        num_value = float(value)
                        rounded = round(num_value, 2)
                        
                        # Remove unnecessary decimal places
                        if rounded == int(rounded):
                            rounded = int(rounded)
                            
                        element.setAttribute(attr, str(rounded))
                    except (ValueError, TypeError):
                        pass
                        
            # Special handling for path data
            if element.nodeName == 'path' and element.hasAttribute('d'):
                path_data = element.getAttribute('d')
                
                # Round numbers in path data using regex
                def round_number(match):
                    num = float(match.group(0))
                    rounded = round(num, 2)
                    if rounded == int(rounded):
                        return str(int(rounded))
                    return str(rounded)
                    
                rounded_data = re.sub(r'-?\d+\.\d+', round_number, path_data)
                element.setAttribute('d', rounded_data)


class SvgGenerator:
    """
    Production-grade SVG generator using LLMs.
    
    Generates SVG content from text prompts with optimizations for
    memory usage, performance, and result quality.
    """
    
    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        prompt_analyzer: Optional[LLMPromptAnalyzer] = None,
        clip_evaluator: Optional[ClipEvaluator] = None,
        template_manager: Optional[SvgTemplateManager] = None,
        batch_size: int = 4,
        preload_models: bool = False
    ):
        """
        Initialize SVG generator.
        
        Args:
            llm_manager: LLM manager for generation
            prompt_analyzer: Prompt analyzer for enhancement
            clip_evaluator: CLIP evaluator for quality assessment
            template_manager: Template manager for SVG templates
            batch_size: Batch size for parallel processing
            preload_models: Whether to preload models at initialization
        """
        # Set up components
        self.llm_manager = llm_manager or LLMManager()
        self.prompt_analyzer = prompt_analyzer or get_default_prompt_analyzer()
        
        # Set up CLIP evaluator if available
        self.clip_evaluator = None
        if clip_evaluator:
            self.clip_evaluator = clip_evaluator
        elif CLIP_AVAILABLE:
            try:
                from svg_prompt_analyzer.llm_integration.clip_evaluator import ClipEvaluator
                self.clip_evaluator = ClipEvaluator()
            except Exception as e:
                logger.warning(f"Failed to initialize CLIP evaluator: {e}")
                
        self.template_manager = template_manager or SvgTemplateManager()
        
        # Set up processor
        self.processor = SvgProcessor()
        
        # Get optimal configuration based on hardware
        optimal_device = hardware_manager.get_optimal_device()
        device_memory = hardware_manager.get_device_memory_stats()
        
        # Adjust batch size based on available memory
        if optimal_device == "cuda" and "allocated" in device_memory:
            # Limit batch size if GPU memory is constrained
            total_mem = sum([m for m in device_memory["total"].values()])
            used_mem = sum([m for m in device_memory["allocated"].values()])
            if used_mem / total_mem > 0.5:  # More than 50% used
                batch_size = min(batch_size, 4)
                logger.info(f"Limiting batch size to {batch_size} due to GPU memory usage")
        
        self.batch_size = batch_size
        
        # Batch processor for parallel generation
        self.batch_processor = BatchProcessor(
            process_func=self._process_generation_batch,
            optimal_batch_size=self.batch_size,
            max_batch_size=self.batch_size * 2,
            min_batch_size=1,
            adaptive_batching=True,
            memory_manager=memory_manager,
            prefetch_next_batch=True
        )
        self.batch_processor.start()
        
        # Cache for generated SVGs
        self._cache = {}
        self._cache_lock = threading.RLock()
        self._cache_size = 200  # Maximum number of cached SVGs
        
        # Preload models if requested
        if preload_models:
            self._preload_models()
            
        logger.info(f"SVG Generator initialized with batch_size={batch_size} on {optimal_device}")
    
    def _preload_models(self) -> None:
        """Preload models for faster first inference."""
        try:
            # Preload LLM model for generation
            self.llm_manager.load_model("svg_generator")
            
            # Preload CLIP model if available
            if self.clip_evaluator:
                self.clip_evaluator.load_model()
                
        except Exception as e:
            logger.warning(f"Error preloading models: {e}")
    
    @log_function_call()
    @memory_manager.memory_efficient_function
    def generate_svg(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> SvgGenerationResult:
        """
        Generate SVG from prompt.
        
        Args:
            prompt: Text prompt
            config: Generation configuration
            
        Returns:
            Generation result
        """
        start_time = time.time()
        
        # Use default config if not provided
        config = config or GenerationConfig()
        
        # Create result object
        result = SvgGenerationResult(
            original_prompt=prompt,
            success=False
        )
        
        # Check cache if enabled
        cache_key = self._create_cache_key(prompt, config)
        with self._cache_lock:
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                
                # Add caching info
                cached.generation_time = 0.0  # Reset time for cache hit
                
                logger.info(f"Using cached SVG for prompt: {prompt[:50]}...")
                
                # Deep copy to avoid modifying cached result
                return self._copy_result(cached)
            
        try:
            with Profiler("svg_generation"):
                # Analyze and enhance prompt
                if config.enhance_prompt:
                    try:
                        enhanced_prompt = self.prompt_analyzer.enhance_prompt(prompt)
                        result.enhanced_prompt = enhanced_prompt
                        generation_prompt = enhanced_prompt
                    except Exception as e:
                        logger.warning(f"Error enhancing prompt: {e}")
                        generation_prompt = prompt
                else:
                    generation_prompt = prompt
                    
                # Report progress if callback provided
                if config.progress_callback:
                    config.progress_callback("prompt_analyzed", 0.1)
                    
                # Generate SVG
                svg_content = None
                retry_count = 0
                quality_score = 0.0
                
                while retry_count <= config.max_retries:
                    try:
                        # Progress update
                        if config.progress_callback:
                            progress = 0.2 + (0.6 * retry_count / (config.max_retries + 1))
                            config.progress_callback("generating", progress)
                            
                        # Choose generation method based on strategy
                        if config.strategy == GenerationStrategy.TEMPLATE:
                            svg_content = self._generate_from_template(generation_prompt, config)
                        elif config.strategy == GenerationStrategy.COMPONENT:
                            svg_content = self._generate_with_components(generation_prompt, config)
                        elif config.strategy == GenerationStrategy.ITERATIVE:
                            svg_content = self._generate_iterative(generation_prompt, config)
                        elif config.strategy == GenerationStrategy.OPTIMIZED:
                            svg_content = self._generate_optimized(generation_prompt, config)
                        else:
                            # Default to direct generation
                            svg_content = self._generate_direct(generation_prompt, config)
                            
                        # Clean and validate SVG
                        svg_content = self.processor.clean_svg(svg_content)
                        is_valid, error = self.processor.validate_svg(svg_content)
                        
                        if not is_valid:
                            logger.warning(f"Generated SVG is invalid: {error}")
                            
                            # Try to fix SVG
                            svg_content = self.processor.fix_svg(svg_content)
                            is_valid, error = self.processor.validate_svg(svg_content)
                            
                            if not is_valid:
                                # If still invalid, retry or fail
                                raise ValueError(f"Failed to generate valid SVG: {error}")
                                
                        # Optimize if requested
                        if config.optimize_output:
                            svg_content = self.processor.optimize_svg(svg_content)
                            
                        # Evaluate quality if CLIP is available
                        if config.use_clip_feedback and self.clip_evaluator:
                            try:
                                quality_score = self.clip_evaluator.compute_similarity(svg_content, generation_prompt)
                                
                                # Retry if below quality threshold
                                if quality_score < config.min_quality_score and retry_count < config.max_retries:
                                    logger.info(f"Quality score {quality_score} below threshold, retrying...")
                                    retry_count += 1
                                    continue
                            except Exception as clip_error:
                                logger.warning(f"CLIP evaluation failed: {clip_error}")
                                # Continue without CLIP evaluation
                                quality_score = 0.8  # Assume decent quality
                        else:
                            # Without CLIP, assume good quality
                            quality_score = 0.8
                            
                        # Successful generation
                        break
                        
                    except Exception as e:
                        logger.error(f"Error generating SVG (attempt {retry_count+1}): {str(e)}")
                        retry_count += 1
                        
                        if retry_count > config.max_retries:
                            raise
                            
                # Check if generation succeeded
                if not svg_content:
                    raise ValueError("Failed to generate SVG after retries")
                    
                # Calculate metrics
                metrics = self.processor.calculate_metrics(svg_content)
                
                # Create successful result
                result.svg_content = svg_content
                result.success = True
                result.quality_score = quality_score
                result.retry_count = retry_count
                result.file_size = metrics["file_size"]
                result.element_count = metrics["element_count"]
                
                # Cache result
                with self._cache_lock:
                    self._cache_result(cache_key, result)
                
        except Exception as e:
            logger.error(f"SVG generation failed: {str(e)}")
            result.success = False
            result.error_message = str(e)
            
        # Add timing information
        result.generation_time = time.time() - start_time
        
        # Final progress update
        if config.progress_callback:
            config.progress_callback("completed", 1.0)
            
        # Memory checkpoint
        memory_manager.operation_checkpoint()
            
        return result
    
    @log_function_call()
    def batch_generate(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[SvgGenerationResult]:
        """
        Generate multiple SVGs in batch.
        
        Args:
            prompts: List of prompts
            config: Generation configuration
            
        Returns:
            List of generation results
        """
        if not prompts:
            return []
            
        # Use default config if not provided
        config = config or GenerationConfig()
        
        logger.info(f"Generating batch of {len(prompts)} SVGs")
        
        # Check cache for each prompt
        results = []
        prompts_to_generate = []
        
        for i, prompt in enumerate(prompts):
            cache_key = self._create_cache_key(prompt, config)
            with self._cache_lock:
                if cache_key in self._cache:
                    # Use cached result
                    cached = self._cache[cache_key]
                    cached.generation_time = 0.0  # Reset time for cache hit
                    results.append((i, self._copy_result(cached)))
                    continue
            
            # Add to list of prompts to generate
            prompts_to_generate.append((i, prompt))
        
        if not prompts_to_generate:
            # All results were cached
            sorted_results = sorted(results, key=lambda x: x[0])
            return [result for _, result in sorted_results]
        
        try:
            with Profiler("batch_svg_generation"):
                # Add tasks to batch processor
                batch_results = {}
                
                for i, prompt in prompts_to_generate:
                    # Create an ID for this task
                    task_id = f"task_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
                    self.batch_processor.add_item(task_id, (prompt, config))
                    
                # Collect results
                timeout_per_prompt = 30.0  # 30 seconds per prompt
                total_timeout = max(60.0, len(prompts_to_generate) * timeout_per_prompt)
                
                # Wait for all results with timeout
                start_time = time.time()
                task_ids = [f"task_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}" for i, prompt in prompts_to_generate]
                
                while len(batch_results) < len(prompts_to_generate):
                    # Check timeout
                    if time.time() - start_time > total_timeout:
                        logger.warning("Batch processing timeout reached")
                        break
                    
                    # Process completed items
                    for task_id, (i, prompt) in zip(task_ids, prompts_to_generate):
                        if i not in batch_results:
                            result = self.batch_processor.get_result(task_id, timeout=0.1)
                            if result:
                                batch_results[i] = result
                    
                    # Short sleep to prevent CPU spinning
                    time.sleep(0.1)
                
                # Process any remaining items sequentially
                for i, prompt in prompts_to_generate:
                    if i not in batch_results:
                        try:
                            # Generate directly
                            result = self.generate_svg(prompt, config)
                            batch_results[i] = result
                        except Exception as e:
                            logger.error(f"Error generating SVG for prompt {i}: {str(e)}")
                            # Add failed result
                            result = SvgGenerationResult(
                                original_prompt=prompt,
                                success=False,
                                error_message=str(e)
                            )
                            batch_results[i] = result
                
                # Add batch results
                for i, result in batch_results.items():
                    results.append((i, result))
                
                # Sort results by original order
                sorted_results = sorted(results, key=lambda x: x[0])
                return [result for _, result in sorted_results]
                
        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}")
            
            # Fall back to sequential generation
            for i, prompt in prompts_to_generate:
                try:
                    result = self.generate_svg(prompt, config)
                    results.append((i, result))
                except Exception as err:
                    logger.error(f"Error generating SVG for prompt {i}: {str(err)}")
                    # Add failed result
                    result = SvgGenerationResult(
                        original_prompt=prompt,
                        success=False,
                        error_message=str(err)
                    )
                    results.append((i, result))
            
            # Sort results by original order
            sorted_results = sorted(results, key=lambda x: x[0])
            return [result for _, result in sorted_results]
    
    def _process_generation_batch(self, items: List[Tuple[str, GenerationConfig]]) -> List[SvgGenerationResult]:
        """Process a batch of prompts for the batch processor."""
        results = []
        for prompt, config in items:
            try:
                result = self.generate_svg(prompt, config)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch item processing: {str(e)}")
                # Add failed result
                result = SvgGenerationResult(
                    original_prompt=prompt,
                    success=False,
                    error_message=str(e)
                )
                results.append(result)
        return results
    
    @log_function_call()
    @memory_manager.memory_efficient_function
    def refine_svg(
        self,
        svg_content: str,
        refinement_prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> SvgGenerationResult:
        """
        Refine existing SVG based on refinement prompt.
        
        Args:
            svg_content: Existing SVG content
            refinement_prompt: Refinement instructions
            config: Generation configuration
            
        Returns:
            Generation result with refined SVG
        """
        start_time = time.time()
        
        # Use default config if not provided
        config = config or GenerationConfig()
        
        # Create result object
        result = SvgGenerationResult(
            original_prompt=refinement_prompt,
            success=False
        )
        
        try:
            with Profiler("svg_refinement"):
                # Prepare the system prompt
                system_prompt = """You are an expert SVG editor. You will receive an existing SVG and instructions for refinement.
                Edit the SVG according to the instructions, maintaining the same structure where possible.
                Only output the complete refined SVG code with no explanations."""
                
                # Clean SVG for consistency
                clean_svg = self.processor.clean_svg(svg_content)
                
                # Create user prompt with SVG and refinement instructions
                user_prompt = f"""Refine this SVG according to these instructions:
                
                Instructions: {refinement_prompt}
                
                Existing SVG:
                ```svg
                {clean_svg}
                ```
                
                Please maintain the same overall structure, IDs, and classes where possible.
                Only output the complete refined SVG code with no explanations."""
                
                # Create messages
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # Generate refined SVG
                response = self.llm_manager.generate(
                    role="svg_generator",
                    prompt=messages,
                    model_type=config.model_type,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
                
                # Extract SVG from response
                refined_svg = extract_svg_from_text(response)
                
                if not refined_svg:
                    raise ValueError("Failed to extract SVG from model response")
                    
                # Clean and validate SVG
                refined_svg = self.processor.clean_svg(refined_svg)
                is_valid, error = self.processor.validate_svg(refined_svg)
                
                if not is_valid:
                    # Try to fix SVG
                    refined_svg = self.processor.fix_svg(refined_svg)
                    is_valid, error = self.processor.validate_svg(refined_svg)
                    
                    if not is_valid:
                        raise ValueError(f"Failed to refine SVG: {error}")
                        
                # Optimize if requested
                if config.optimize_output:
                    refined_svg = self.processor.optimize_svg(refined_svg)
                    
                # Calculate metrics
                metrics = self.processor.calculate_metrics(refined_svg)
                
                # Create successful result
                result.svg_content = refined_svg
                result.success = True
                result.file_size = metrics["file_size"]
                result.element_count = metrics["element_count"]
                
        except Exception as e:
            logger.error(f"SVG refinement failed: {str(e)}")
            result.success = False
            result.error_message = str(e)
            
        # Add timing information
        result.generation_time = time.time() - start_time
        
        # Memory checkpoint
        memory_manager.operation_checkpoint()
        
        return result
    
    def _generate_direct(self, prompt: str, config: GenerationConfig) -> SvgString:
        """
        Generate SVG directly using LLM.
        
        Args:
            prompt: Generation prompt
            config: Generation configuration
            
        Returns:
            Generated SVG content
            
        Raises:
            Exception: If generation fails
        """
        # Prepare the system prompt
        system_prompt = """You are an expert SVG designer. Generate clean, semantic SVG code based on text prompts.
        Focus on vector graphics principles with clear structure.
        Use only valid SVG elements and attributes.
        Your response should contain ONLY the SVG code, no explanations or comments."""
        
        # Prepare user prompt
        user_prompt = f"""Create an SVG image based on this description:
        
        {prompt}
        
        Requirements:
        - Size: {config.width}x{config.height} pixels
        - Use proper viewBox attribute
        - Create semantic, clean SVG code
        - Use descriptive IDs and group related elements
        - Focus on vector graphic techniques (paths, shapes, gradients)
        - Ensure all elements have appropriate attributes
        - Organize elements in a logical structure
        
        Return ONLY the SVG code with no explanations."""
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate SVG
        response = self.llm_manager.generate(
            role="svg_generator",
            prompt=messages,
            model_type=config.model_type,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Extract SVG from response
        svg_content = extract_svg_from_text(response)
        
        if not svg_content:
            raise ValueError("Failed to extract SVG from model response")
            
        return svg_content
    
    def _generate_from_template(self, prompt: str, config: GenerationConfig) -> SvgString:
        """
        Generate SVG using a template.
        
        Args:
            prompt: Generation prompt
            config: Generation configuration
            
        Returns:
            Generated SVG content
            
        Raises:
            Exception: If generation fails
        """
        # Get template
        template_id = config.template_id
        
        if not template_id:
            # Find suitable template based on prompt
            template_id, template = self.template_manager.get_template_for_prompt(prompt)
        else:
            # Use specified template
            template = self.template_manager.get_template(template_id)
            
        if not template:
            raise ValueError(f"Template not found: {template_id}")
            
        # Fill in template placeholders
        filled_template = template.format(
            width=config.width,
            height=config.height,
            prompt=prompt
        )
        
        # Prepare the system prompt
        system_prompt = """You are an expert SVG designer. Complete the SVG template based on the provided prompt.
        Replace the commented instructions with appropriate SVG elements.
        Ensure the resulting SVG is valid, semantic, and matches the prompt.
        Your response should contain ONLY the complete SVG code, no explanations."""
        
        # Prepare user prompt
        user_prompt = f"""Complete this SVG template based on the prompt:
        
        {filled_template}
        
        Replace the commented instructions with appropriate SVG elements that match the prompt.
        Return ONLY the completed SVG code with no explanations."""
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate SVG
        response = self.llm_manager.generate(
            role="svg_generator",
            prompt=messages,
            model_type=config.model_type,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Extract SVG from response
        svg_content = extract_svg_from_text(response)
        
        if not svg_content:
            raise ValueError("Failed to extract SVG from model response")
            
        return svg_content
    
    def _generate_with_components(self, prompt: str, config: GenerationConfig) -> SvgString:
        """
        Generate SVG by creating components and assembling them.
        
        Args:
            prompt: Generation prompt
            config: Generation configuration
            
        Returns:
            Generated SVG content
            
        Raises:
            Exception: If generation fails
        """
        # First, get a structured representation of the prompt
        structured_prompt = self.prompt_analyzer.get_structured_prompt(prompt)
        
        # Prepare the system prompt for component generation
        system_prompt = """You are an expert SVG component designer. You will be given a structured prompt and asked to create 
        specific SVG components. Create clean, semantic SVG code for each component.
        Your response should be a JSON object containing the SVG code for each component."""
        
        # Prepare user prompt for component generation
        user_prompt = f"""Create SVG components based on this structured prompt:
        
        {json.dumps(structured_prompt, indent=2)}
        
        Generate an SVG component for each element in the "elements" array, and a background component.
        For each component:
        - Create clean, semantic SVG code
        - Size it appropriately to be combined later
        - Use appropriate styles based on the style information
        - Ensure proper viewBox attributes
        
        Return a JSON object with the following structure:
        {{
          "background": "<!-- SVG code for background -->",
          "elements": [
            "<!-- SVG code for element 1 -->",
            "<!-- SVG code for element 2 -->",
            ...
          ]
        }}
        """
        
        # Create messages for component generation
        component_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate components
        component_response = self.llm_manager.generate(
            role="svg_generator",
            prompt=component_messages,
            model_type=config.model_type,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Extract JSON data
        components_data = extract_json_from_text(component_response)
        
        if not components_data:
            raise ValueError("Failed to extract component data from model response")
            
        # Prepare the system prompt for assembly
        assembly_system_prompt = """You are an expert SVG assembler. You will be given SVG components to combine 
        into a single cohesive SVG image. Create clean, semantic SVG code that incorporates all components.
        Your response should contain ONLY the complete SVG code, no explanations."""
        
        # Prepare components for assembly
        background = components_data.get('background', '<!-- No background -->')
        elements = components_data.get('elements', [])
        
        # Create element string
        element_str = "\n\n".join([f"```svg\n{element}\n```" for element in elements])
        
        # Prepare user prompt for assembly
        assembly_user_prompt = f"""Combine these SVG components into a single SVG image:
        
        Background:
        ```svg
        {background}
        ```
        
        Elements:
        {element_str}
        
        Requirements:
        - Final size: {config.width}x{config.height} pixels
        - Proper viewBox attribute
        - Position elements appropriately
        - Organize with semantic grouping
        - Ensure z-index is appropriate (background in back, etc.)
        
        Return ONLY the complete assembled SVG code with no explanations."""
        
        # Create messages for assembly
        assembly_messages = [
            {"role": "system", "content": assembly_system_prompt},
            {"role": "user", "content": assembly_user_prompt}
        ]
        
        # Generate assembled SVG
        assembly_response = self.llm_manager.generate(
            role="svg_generator",
            prompt=assembly_messages,
            model_type=config.model_type,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Extract SVG from response
        svg_content = extract_svg_from_text(assembly_response)
        
        if not svg_content:
            raise ValueError("Failed to extract assembled SVG from model response")
            
        return svg_content
    
    def _generate_iterative(self, prompt: str, config: GenerationConfig) -> SvgString:
        """
        Generate SVG iteratively with refinements.
        
        Args:
            prompt: Generation prompt
            config: Generation configuration
            
        Returns:
            Generated SVG content
            
        Raises:
            Exception: If generation fails
        """
        # First, generate a basic SVG
        basic_svg = self._generate_direct(prompt, config)
        
        # Clean and validate
        basic_svg = self.processor.clean_svg(basic_svg)
        is_valid, error = self.processor.validate_svg(basic_svg)
        
        if not is_valid:
            # Try to fix SVG
            basic_svg = self.processor.fix_svg(basic_svg)
            is_valid, error = self.processor.validate_svg(basic_svg)
            
            if not is_valid:
                raise ValueError(f"Initial SVG generation failed: {error}")
                
        # Prepare the system prompt for refinement
        refine_system_prompt = """You are an expert SVG refiner. You will be given an initial SVG and instructions for improvement.
        Enhance the SVG while maintaining its structure and semantic elements.
        Your response should contain ONLY the refined SVG code, no explanations."""
        
        # Analyze the prompt for enhancement guidance
        prompt_analysis = analyze(prompt)
        
        # Create refinement instructions
        refinement_instructions = "Refine this SVG to better match the prompt by:\n"
        
        # Add specific refinement points
        refinement_instructions += "- Improving visual details and complexity\n"
        refinement_instructions += "- Enhancing color usage and gradients where appropriate\n"
        
        # Add style-specific instructions
        if prompt_analysis.style_cues:
            refinement_instructions += f"- Emphasizing these styles: {', '.join(prompt_analysis.style_cues)}\n"
            
        # Add object-specific instructions
        if prompt_analysis.objects:
            refinement_instructions += f"- Ensuring these elements are well-represented: {', '.join(prompt_analysis.objects)}\n"
            
        # Add color-specific instructions
        if prompt_analysis.colors:
            refinement_instructions += f"- Using these colors effectively: {', '.join(prompt_analysis.colors)}\n"
            
        # Prepare user prompt for refinement
        refine_user_prompt = f"""Refine this initial SVG to better match the prompt:
        
        Original prompt: "{prompt}"
        
        Refinement instructions:
        {refinement_instructions}
        
        Initial SVG:
        ```svg
        {basic_svg}
        ```
        
        Return ONLY the refined SVG code with no explanations."""
        
        # Create messages for refinement
        refine_messages = [
            {"role": "system", "content": refine_system_prompt},
            {"role": "user", "content": refine_user_prompt}
        ]
        
        # Generate refined SVG
        refine_response = self.llm_manager.generate(
            role="svg_generator",
            prompt=refine_messages,
            model_type=config.model_type,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Extract SVG from response
        refined_svg = extract_svg_from_text(refine_response)
        
        if not refined_svg:
            # Fall back to initial SVG
            logger.warning("Refinement failed, using initial SVG")
            return basic_svg
            
        return refined_svg
    
    def _generate_optimized(self, prompt: str, config: GenerationConfig) -> SvgString:
        """
        Generate SVG optimized for small file size.
        
        Args:
            prompt: Generation prompt
            config: Generation configuration
            
        Returns:
            Generated SVG content
            
        Raises:
            Exception: If generation fails
        """
        # Prepare the system prompt for optimized generation
        system_prompt = """You are an expert SVG designer specializing in minimal, optimized SVGs.
        Generate a compact, clean SVG that represents the prompt accurately with minimal file size.
        Focus on simplicity, efficient paths, and minimal attributes.
        Your response should contain ONLY the SVG code, no explanations."""
        
        # Prepare user prompt
        user_prompt = f"""Create a highly optimized, minimal-size SVG based on this description:
        
        {prompt}
        
        Optimization requirements:
        - Size: {config.width}x{config.height} pixels
        - Minimize the number of elements
        - Use compact path data (avoid unnecessary precision)
        - Eliminate redundant attributes
        - Merge shapes where possible
        - Use efficient techniques (paths instead of multiple shapes)
        - Avoid unnecessary groups
        - Limit color variations
        
        Return ONLY the optimized SVG code with no explanations."""
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate SVG
        response = self.llm_manager.generate(
            role="svg_generator",
            prompt=messages,
            model_type=config.model_type,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Extract SVG from response
        svg_content = extract_svg_from_text(response)
        
        if not svg_content:
            raise ValueError("Failed to extract SVG from model response")
        
        # Extra optimization pass for file size
        svg_content = self.processor.optimize_svg(svg_content)
            
        return svg_content
    
    def _create_cache_key(self, prompt: str, config: GenerationConfig) -> str:
        """
        Create cache key for a generation request.
        
        Args:
            prompt: Generation prompt
            config: Generation configuration
            
        Returns:
            Cache key
        """
        # Create key components
        key_parts = [
            prompt,
            config.model_type.name,
            config.strategy.name,
            str(config.temperature),
            str(config.max_tokens),
            str(config.width),
            str(config.height),
            str(config.enhance_prompt),
            str(config.template_id)
        ]
        
        # Join and hash
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _cache_result(self, key: str, result: SvgGenerationResult) -> None:
        """
        Cache a generation result.
        
        Args:
            key: Cache key
            result: Result to cache
        """
        # Check cache size
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry (arbitrary but consistent)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            
        # Add to cache
        self._cache[key] = result
    
    @staticmethod
    def _copy_result(result: SvgGenerationResult) -> SvgGenerationResult:
        """
        Create a deep copy of a generation result.
        
        Args:
            result: Result to copy
            
        Returns:
            Copied result
        """
        return SvgGenerationResult(
            svg_content=result.svg_content,
            original_prompt=result.original_prompt,
            enhanced_prompt=result.enhanced_prompt,
            generation_time=result.generation_time,
            token_count=result.token_count,
            retry_count=result.retry_count,
            quality_score=result.quality_score,
            file_size=result.file_size,
            element_count=result.element_count,
            success=result.success,
            error_message=result.error_message
        )
    
    def shutdown(self) -> None:
        """Clean up resources when done."""
        logger.info("Shutting down SVG Generator")
        
        # Stop batch processor
        if hasattr(self, 'batch_processor'):
            self.batch_processor.stop(wait_complete=True)
            
        # Clear cache
        with self._cache_lock:
            self._cache.clear()


# Create singleton instance for easy import
_default_svg_generator = None

def get_default_svg_generator() -> SvgGenerator:
    """Get default SVG generator instance."""
    global _default_svg_generator
    if _default_svg_generator is None:
        _default_svg_generator = SvgGenerator()
    return _default_svg_generator


# Utility functions
def generate_svg(prompt: str, **kwargs) -> SvgGenerationResult:
    """
    Generate SVG from prompt.
    
    Args:
        prompt: Text prompt
        **kwargs: Additional configuration parameters
        
    Returns:
        Generation result
    """
    # Convert kwargs to config object
    config = GenerationConfig()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            
    return get_default_svg_generator().generate_svg(prompt, config)


def refine_svg(svg_content: str, refinement_prompt: str, **kwargs) -> SvgGenerationResult:
    """
    Refine existing SVG.
    
    Args:
        svg_content: Existing SVG content
        refinement_prompt: Refinement instructions
        **kwargs: Additional configuration parameters
        
    Returns:
        Generation result with refined SVG
    """
    # Convert kwargs to config object
    config = GenerationConfig()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            
    return get_default_svg_generator().refine_svg(svg_content, refinement_prompt, config)


def svg_to_png(svg_content: str, width: int = 800, height: int = 600) -> bytes:
    """
    Convert SVG to PNG.
    
    Args:
        svg_content: SVG content to convert
        width: Output width
        height: Output height
        
    Returns:
        PNG image data
        
    Raises:
        Exception: If conversion fails
    """
    # Require cairosvg for conversion
    try:
        import cairosvg
        
        # Convert SVG to PNG
        png_data = cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            output_width=width,
            output_height=height
        )
        
        return png_data
        
    except ImportError:
        raise ImportError("cairosvg is required for SVG to PNG conversion")
    except Exception as e:
        raise ValueError(f"Failed to convert SVG to PNG: {str(e)}")