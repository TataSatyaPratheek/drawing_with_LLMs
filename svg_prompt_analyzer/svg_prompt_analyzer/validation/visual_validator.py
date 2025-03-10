"""
Production-grade SVG validation tool with performance optimizations.
Validates SVG content against standards, checks for rendering issues,
and provides optimization suggestions.
"""

import re
import os
import io
import time
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import threading
import hashlib

# Import core optimizations
from svg_prompt_analyzer.core import CONFIG, memoize, get_thread_pool, Profiler
from svg_prompt_analyzer.utils.logger import get_logger, log_function_call

# Configure logger
logger = get_logger(__name__)

# Constants for validation
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"
NAMESPACES = {
    "svg": SVG_NAMESPACE,
    "xlink": XLINK_NAMESPACE
}

# Common SVG elements and attributes for validation
VALID_SVG_ELEMENTS = {
    "svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon",
    "text", "tspan", "textPath", "image", "use", "defs", "symbol", "linearGradient",
    "radialGradient", "stop", "clipPath", "mask", "pattern", "marker", "filter",
    "feBlend", "feColorMatrix", "feComponentTransfer", "feComposite", "feConvolveMatrix",
    "feDiffuseLighting", "feDisplacementMap", "feFlood", "feGaussianBlur", "feImage",
    "feMerge", "feMergeNode", "feMorphology", "feOffset", "feSpecularLighting",
    "feTile", "feTurbulence", "foreignObject", "style", "title", "desc", "metadata",
    "a", "switch", "view"
}

# LRU Cache for validators to improve performance
_validation_cache = {}
_cache_lock = threading.RLock()
_MAX_CACHE_SIZE = 1000  # Adjust based on memory constraints


class SVGValidationError(Exception):
    """Custom exception for SVG validation errors."""
    pass


class ValidationResult:
    """Results of SVG validation with detailed breakdown."""
    
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.info = []
        self.metrics = {
            "element_count": 0,
            "depth": 0,
            "size_bytes": 0,
            "unique_ids": 0,
            "complexity_score": 0,
            "validation_time_ms": 0
        }
        self.optimizations = []
        self._start_time = time.time()
        
    def add_error(self, message: str, location: Optional[str] = None):
        """Add an error to the validation results."""
        error_info = {"message": message}
        if location:
            error_info["location"] = location
        self.errors.append(error_info)
        self.is_valid = False
        
    def add_warning(self, message: str, location: Optional[str] = None):
        """Add a warning to the validation results."""
        warning_info = {"message": message}
        if location:
            warning_info["location"] = location
        self.warnings.append(warning_info)
        
    def add_info(self, message: str):
        """Add information to the validation results."""
        self.info.append({"message": message})
        
    def add_optimization(self, message: str, impact: str = "medium"):
        """Add an optimization suggestion."""
        self.optimizations.append({
            "message": message,
            "impact": impact
        })
        
    def update_metrics(self, key: str, value: Any):
        """Update a metric in the validation results."""
        self.metrics[key] = value
        
    def finalize(self):
        """Finalize validation results and compute metrics."""
        self.metrics["validation_time_ms"] = int((time.time() - self._start_time) * 1000)
        return self
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation results to a dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "metrics": self.metrics,
            "optimizations": self.optimizations
        }


class SVGValidator:
    """
    Production-grade SVG validation tool with performance optimizations.
    
    Validates SVG content against standards, checks for rendering issues,
    and provides optimization suggestions.
    """
    
    def __init__(
        self,
        strict_mode: bool = False,
        check_accessibility: bool = True,
        validate_rendering: bool = True,
        max_file_size_kb: int = 500,
        max_elements: int = 10000,
        parallelism: int = None
    ):
        """
        Initialize the SVG validator.
        
        Args:
            strict_mode: Whether to enforce strict validation
            check_accessibility: Whether to check accessibility guidelines
            validate_rendering: Whether to validate rendering capabilities
            max_file_size_kb: Maximum allowed file size in KB
            max_elements: Maximum allowed number of elements
            parallelism: Number of parallel validation threads
        """
        self.strict_mode = strict_mode
        self.check_accessibility = check_accessibility
        self.validate_rendering = validate_rendering
        self.max_file_size_kb = max_file_size_kb
        self.max_elements = max_elements
        
        # Set up parallelism based on available resources
        self.parallelism = parallelism or CONFIG.get(
            "thread_pool_size",
            min(8, (os.cpu_count() or 4))
        )
        
        # Initialize thread pool if needed
        self._thread_pool = None

    def _get_thread_pool(self) -> ThreadPoolExecutor:
        """Get thread pool for parallel validation."""
        if self._thread_pool is None:
            self._thread_pool = get_thread_pool()
        return self._thread_pool

    @memoize
    def _compute_file_hash(self, svg_content: str) -> str:
        """Compute hash of SVG content for caching."""
        return hashlib.md5(svg_content.encode('utf-8')).hexdigest()

    def _check_cache(self, svg_content: str) -> Optional[ValidationResult]:
        """Check if validation results are already cached."""
        file_hash = self._compute_file_hash(svg_content)
        
        with _cache_lock:
            if file_hash in _validation_cache:
                logger.debug(f"Cache hit for SVG validation: {file_hash}")
                return _validation_cache[file_hash]
        
        return None

    def _update_cache(self, svg_content: str, result: ValidationResult):
        """Update validation cache with results."""
        file_hash = self._compute_file_hash(svg_content)
        
        with _cache_lock:
            # Add to cache
            _validation_cache[file_hash] = result
            
            # Clean cache if needed
            if len(_validation_cache) > _MAX_CACHE_SIZE:
                # Remove oldest entries (approximately 10%)
                keys_to_remove = list(_validation_cache.keys())[:_MAX_CACHE_SIZE // 10]
                for key in keys_to_remove:
                    _validation_cache.pop(key, None)

    @log_function_call(level=logging.DEBUG)
    def validate(self, svg_content: str) -> ValidationResult:
        """
        Validate SVG content with comprehensive checks.
        
        Args:
            svg_content: SVG content to validate
            
        Returns:
            ValidationResult with validation details
        """
        with Profiler("svg_validation"):
            # Check cache first
            cached_result = self._check_cache(svg_content)
            if cached_result is not None:
                return cached_result
                
            result = ValidationResult()
            
            # Basic checks
            if not self._validate_basic(svg_content, result):
                # Skip further validation if basic checks fail
                result.finalize()
                self._update_cache(svg_content, result)
                return result
                
            try:
                # Parse SVG
                root = self._parse_svg(svg_content, result)
                if root is None:
                    result.finalize()
                    self._update_cache(svg_content, result)
                    return result
                    
                # Perform all validations in parallel for performance
                validations = [
                    (self._validate_structure, (root, result)),
                    (self._validate_ids, (root, result)),
                    (self._validate_css, (root, result)),
                    (self._validate_paths, (root, result)),
                    (self._check_optimizations, (root, result, svg_content))
                ]
                
                if self.check_accessibility:
                    validations.append((self._validate_accessibility, (root, result)))
                    
                if self.validate_rendering:
                    validations.append((self._validate_rendering, (root, result)))
                
                # Execute validations in parallel
                thread_pool = self._get_thread_pool()
                futures = []
                
                for func, args in validations:
                    futures.append(thread_pool.submit(func, *args))
                    
                # Wait for all validations to complete
                for future in futures:
                    # Collect any exceptions
                    try:
                        future.result()
                    except Exception as e:
                        logger.exception(f"Error during SVG validation: {e}")
                        result.add_error(f"Validation error: {str(e)}")
                        
                # Calculate complexity score
                self._calculate_complexity(root, result)
                
            except Exception as e:
                logger.exception(f"Unexpected error during SVG validation: {e}")
                result.add_error(f"Validation failed: {str(e)}")
                
            # Finalize and cache results
            result.finalize()
            self._update_cache(svg_content, result)
            return result

    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate SVG file.
        
        Args:
            file_path: Path to SVG file
            
        Returns:
            ValidationResult with validation details
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                result = ValidationResult()
                result.add_error(f"File not found: {file_path}")
                return result.finalize()
                
            # Check file size
            file_size_kb = os.path.getsize(file_path) / 1024
            if file_size_kb > self.max_file_size_kb:
                result = ValidationResult()
                result.add_error(
                    f"File size ({file_size_kb:.1f} KB) exceeds maximum allowed "
                    f"({self.max_file_size_kb} KB)"
                )
                return result.finalize()
                
            # Read file with proper encoding detection
            with open(file_path, 'rb') as f:
                content = f.read()
                
            # Try to decode as UTF-8
            try:
                svg_content = content.decode('utf-8')
            except UnicodeDecodeError:
                # Fall back to Latin-1
                try:
                    svg_content = content.decode('latin-1')
                except UnicodeDecodeError:
                    result = ValidationResult()
                    result.add_error("Unable to decode file content")
                    return result.finalize()
                    
            # Validate content
            result = self.validate(svg_content)
            result.update_metrics("size_bytes", len(content))
            
            return result
            
        except Exception as e:
            logger.exception(f"Error validating SVG file {file_path}: {e}")
            result = ValidationResult()
            result.add_error(f"File validation error: {str(e)}")
            return result.finalize()

    def _validate_basic(self, svg_content: str, result: ValidationResult) -> bool:
        """
        Perform basic validation checks.
        
        Args:
            svg_content: SVG content to validate
            result: ValidationResult to update
            
        Returns:
            Whether basic validation passed
        """
        # Check file size
        size_bytes = len(svg_content.encode('utf-8'))
        result.update_metrics("size_bytes", size_bytes)
        
        if size_bytes > self.max_file_size_kb * 1024:
            result.add_error(
                f"SVG size ({size_bytes / 1024:.1f} KB) exceeds maximum allowed "
                f"({self.max_file_size_kb} KB)"
            )
            return False
            
        # Check for SVG header
        if not re.search(r'<svg[^>]*>', svg_content):
            result.add_error("Missing SVG root element")
            return False
            
        # Check for XML declaration
        if not svg_content.strip().startswith('<?xml') and self.strict_mode:
            result.add_warning("Missing XML declaration")
            
        # Check for SVG namespace
        if SVG_NAMESPACE not in svg_content and self.strict_mode:
            result.add_warning("Missing SVG namespace declaration")
            
        # Check for script tags (potential security issue)
        if '<script' in svg_content:
            result.add_warning("SVG contains script tags which may pose security risks")
            
        # Check for external references
        if 'xlink:href="http' in svg_content:
            result.add_warning("SVG contains external references which may not load correctly")
            
        return True

    def _parse_svg(self, svg_content: str, result: ValidationResult) -> Optional[ET.Element]:
        """
        Parse SVG content into ElementTree.
        
        Args:
            svg_content: SVG content to parse
            result: ValidationResult to update
            
        Returns:
            ElementTree root element or None if parsing failed
        """
        try:
            # Register namespaces
            for prefix, uri in NAMESPACES.items():
                ET.register_namespace(prefix, uri)
                
            # Parse SVG
            try:
                tree = ET.fromstring(svg_content)
            except ET.ParseError as e:
                result.add_error(f"XML parsing error: {str(e)}")
                return None
                
            # Verify root element is SVG
            if tree.tag != f"{{{SVG_NAMESPACE}}}svg":
                result.add_error(f"Root element is not SVG, found: {tree.tag}")
                return None
                
            return tree
            
        except Exception as e:
            result.add_error(f"Failed to parse SVG: {str(e)}")
            return None

    def _validate_structure(self, root: ET.Element, result: ValidationResult) -> None:
        """
        Validate SVG structure.
        
        Args:
            root: Root SVG element
            result: ValidationResult to update
        """
        # Check SVG attributes
        if 'width' not in root.attrib and 'height' not in root.attrib and 'viewBox' not in root.attrib:
            result.add_warning("SVG lacks width, height, and viewBox attributes")
            
        # Count elements and validate
        element_count = 0
        element_types = set()
        max_depth = [0]
        
        def count_elements(element, depth=0):
            nonlocal element_count
            element_count += 1
            max_depth[0] = max(max_depth[0], depth)
            
            # Extract element name without namespace
            tag = element.tag
            if '}' in tag:
                tag = tag.split('}', 1)[1]
                
            element_types.add(tag)
            
            # Check invalid elements
            if tag not in VALID_SVG_ELEMENTS and not tag.startswith('fe'):
                result.add_warning(f"Unknown SVG element: {tag}")
                
            # Validate children
            for child in element:
                count_elements(child, depth + 1)
                
            # Check if element count exceeds limit
            if element_count > self.max_elements:
                result.add_error(
                    f"SVG has too many elements: {element_count} (max: {self.max_elements})"
                )
                return
                
        # Count elements
        count_elements(root)
        
        # Update metrics
        result.update_metrics("element_count", element_count)
        result.update_metrics("depth", max_depth[0])
        
        # Check for missing elements
        if 'path' not in element_types and 'rect' not in element_types and 'circle' not in element_types:
            result.add_warning("SVG lacks common shape elements (path, rect, circle)")

    def _validate_ids(self, root: ET.Element, result: ValidationResult) -> None:
        """
        Validate ID uniqueness and references.
        
        Args:
            root: Root SVG element
            result: ValidationResult to update
        """
        ids = set()
        references = set()
        
        # Find all IDs and references
        for elem in root.findall(".//*[@id]"):
            id_value = elem.get('id')
            if id_value in ids:
                result.add_error(f"Duplicate ID: {id_value}")
            ids.add(id_value)
            
        # Find all references
        for elem in root.findall(".//*[@*]"):
            for attr_name, attr_value in elem.attrib.items():
                # Check for references in xlink:href, href, or url()
                if (attr_name.endswith('}href') or attr_name == 'href') and attr_value.startswith('#'):
                    references.add(attr_value[1:])  # Remove leading #
                elif 'url(#' in attr_value:
                    # Extract ID from url(#id)
                    id_match = re.search(r'url\(#([^)]+)\)', attr_value)
                    if id_match:
                        references.add(id_match.group(1))
                        
        # Check for broken references
        for ref in references:
            if ref not in ids:
                result.add_warning(f"Reference to non-existent ID: {ref}")
                
        # Update metrics
        result.update_metrics("unique_ids", len(ids))

    def _validate_css(self, root: ET.Element, result: ValidationResult) -> None:
        """
        Validate CSS usage.
        
        Args:
            root: Root SVG element
            result: ValidationResult to update
        """
        # Check for style elements
        style_elements = root.findall(f".//{{{SVG_NAMESPACE}}}style")
        
        for style in style_elements:
            css_content = style.text or ""
            
            # Check for browser-specific CSS
            browser_prefixes = ['-webkit-', '-moz-', '-ms-', '-o-']
            for prefix in browser_prefixes:
                if prefix in css_content:
                    result.add_warning(f"Browser-specific CSS prefix found: {prefix}")
                    
            # Check for potential issues
            if '!important' in css_content:
                result.add_warning("Use of '!important' in CSS may cause rendering issues")
                
        # Check for inline styles
        inline_style_count = 0
        for elem in root.findall(".//*[@style]"):
            inline_style_count += 1
            
        if inline_style_count > 20:
            result.add_optimization(
                f"Consider using CSS classes instead of {inline_style_count} inline styles",
                impact="medium"
            )
            
        # Check for presentation attributes that could be in CSS
        presentation_attrs = ['fill', 'stroke', 'stroke-width', 'font-size', 'font-family']
        presentation_attr_count = 0
        
        for attr in presentation_attrs:
            elements = root.findall(f".//*[@{attr}]")
            presentation_attr_count += len(elements)
            
        if presentation_attr_count > 50:
            result.add_optimization(
                f"Consider using CSS for {presentation_attr_count} presentation attributes",
                impact="medium"
            )

    def _validate_paths(self, root: ET.Element, result: ValidationResult) -> None:
        """
        Validate SVG paths.
        
        Args:
            root: Root SVG element
            result: ValidationResult to update
        """
        path_elements = root.findall(f".//{{{SVG_NAMESPACE}}}path")
        
        # Check path data
        for path in path_elements:
            path_data = path.get('d', '')
            
            # Check for non-relative commands
            absolute_commands = re.findall(r'[MLHVCSQTA]', path_data)
            relative_commands = re.findall(r'[mlhvcsqta]', path_data)
            
            if len(absolute_commands) > 0 and len(relative_commands) == 0:
                result.add_optimization(
                    "Consider using relative path commands for smaller file size",
                    impact="low"
                )
                
            # Check for unnecessarily precise numbers
            decimal_points = re.findall(r'\d\.\d{5,}', path_data)
            if decimal_points:
                result.add_optimization(
                    "Path data contains unnecessarily precise numbers (>4 decimal places)",
                    impact="low"
                )
                
            # Check for potentially malformed path data
            if not re.match(r'^[mM]', path_data):
                result.add_warning(f"Path data should start with 'm' or 'M' command")

    def _validate_accessibility(self, root: ET.Element, result: ValidationResult) -> None:
        """
        Validate accessibility features.
        
        Args:
            root: Root SVG element
            result: ValidationResult to update
        """
        # Check for title element
        title_elements = root.findall(f".//{{{SVG_NAMESPACE}}}title")
        if not title_elements:
            result.add_warning("Missing <title> element (recommended for accessibility)")
            
        # Check for desc element
        desc_elements = root.findall(f".//{{{SVG_NAMESPACE}}}desc")
        if not desc_elements:
            result.add_warning("Missing <desc> element (recommended for accessibility)")
            
        # Check for ARIA attributes
        has_aria = False
        for elem in root.findall(".//*[@*]"):
            for attr_name in elem.attrib.keys():
                if attr_name.startswith('aria-') or attr_name == 'role':
                    has_aria = True
                    break
            if has_aria:
                break
                
        if not has_aria:
            result.add_warning("No ARIA attributes found (could improve accessibility)")
            
        # Check text elements for potential issues
        text_elements = root.findall(f".//{{{SVG_NAMESPACE}}}text")
        for text in text_elements:
            font_size = text.get('font-size')
            if font_size and font_size.endswith('px') and float(font_size[:-2]) < 12:
                result.add_warning(f"Text with font-size {font_size} may be too small for readability")

    def _validate_rendering(self, root: ET.Element, result: ValidationResult) -> None:
        """
        Validate rendering capabilities.
        
        Args:
            root: Root SVG element
            result: ValidationResult to update
        """
        # Check for filters
        filter_elements = root.findall(f".//{{{SVG_NAMESPACE}}}filter")
        if filter_elements:
            result.add_info(f"SVG uses {len(filter_elements)} filter elements")
            
        # Check for masks
        mask_elements = root.findall(f".//{{{SVG_NAMESPACE}}}mask")
        if mask_elements:
            result.add_info(f"SVG uses {len(mask_elements)} mask elements")
            
        # Check for patterns
        pattern_elements = root.findall(f".//{{{SVG_NAMESPACE}}}pattern")
        if pattern_elements:
            result.add_info(f"SVG uses {len(pattern_elements)} pattern elements")
            
        # Check for advanced features that might not render consistently
        advanced_features = {
            'clipPath': f".//{{{SVG_NAMESPACE}}}clipPath",
            'foreignObject': f".//{{{SVG_NAMESPACE}}}foreignObject",
            'marker': f".//{{{SVG_NAMESPACE}}}marker",
            'symbol': f".//{{{SVG_NAMESPACE}}}symbol"
        }
        
        for feature, xpath in advanced_features.items():
            elements = root.findall(xpath)
            if elements:
                result.add_info(f"SVG uses {len(elements)} {feature} elements")
                
        # Check for animation elements
        animation_elements = {
            'animate': f".//{{{SVG_NAMESPACE}}}animate",
            'animateTransform': f".//{{{SVG_NAMESPACE}}}animateTransform",
            'animateMotion': f".//{{{SVG_NAMESPACE}}}animateMotion",
            'set': f".//{{{SVG_NAMESPACE}}}set"
        }
        
        has_animation = False
        for feature, xpath in animation_elements.items():
            elements = root.findall(xpath)
            if elements:
                has_animation = True
                result.add_info(f"SVG uses {len(elements)} {feature} elements")
                
        if has_animation:
            result.add_warning("SVG uses animation which may not be supported in all contexts")

    def _check_optimizations(
        self, 
        root: ET.Element, 
        result: ValidationResult,
        svg_content: str
    ) -> None:
        """
        Check for potential optimizations.
        
        Args:
            root: Root SVG element
            result: ValidationResult to update
            svg_content: Original SVG content
        """
        # Check raw file size
        size_kb = len(svg_content.encode('utf-8')) / 1024
        
        # Check for trailing zeros
        trailing_zeros = re.findall(r'\d+\.0+(?=[^0-9]|$)', svg_content)
        if trailing_zeros:
            result.add_optimization(
                f"Remove {len(trailing_zeros)} unnecessary trailing zeros",
                impact="low"
            )
            
        # Check for unnecessary precision in transform attributes
        transform_precision = re.findall(r'transform="[^"]*\d\.\d{5,}[^"]*"', svg_content)
        if transform_precision:
            result.add_optimization(
                f"Reduce precision in {len(transform_precision)} transform attributes",
                impact="low"
            )
            
        # Check for redundant groups
        empty_groups = root.findall(f".//{{{SVG_NAMESPACE}}}g[not(*)]")
        if empty_groups:
            result.add_optimization(
                f"Remove {len(empty_groups)} empty group elements",
                impact="low"
            )
            
        single_child_groups = []
        for group in root.findall(f".//{{{SVG_NAMESPACE}}}g"):
            children = list(group)
            if len(children) == 1 and not group.attrib:
                single_child_groups.append(group)
                
        if single_child_groups:
            result.add_optimization(
                f"Simplify {len(single_child_groups)} groups with single child and no attributes",
                impact="low"
            )
            
        # Check for unnecessary whitespace and comments
        whitespace_ratio = len(re.findall(r'>\s+<', svg_content)) / max(1, len(svg_content))
        if whitespace_ratio > 0.05:
            result.add_optimization(
                "Minify SVG by removing unnecessary whitespace",
                impact="medium"
            )
            
        # Check for comments
        comment_count = len(re.findall(r'<!--.*?-->', svg_content, re.DOTALL))
        if comment_count > 0:
            result.add_optimization(
                f"Remove {comment_count} comments to reduce file size",
                impact="low"
            )
            
        # Check for unnecessary defaulted attributes
        default_attrs = [
            'x="0"', 'y="0"', 'dx="0"', 'dy="0"', 
            'fill-opacity="1"', 'stroke-opacity="1"',
            'stroke-width="1"', 'stroke-linecap="butt"',
            'stroke-linejoin="miter"'
        ]
        
        default_count = 0
        for attr in default_attrs:
            default_count += svg_content.count(attr)
            
        if default_count > 10:
            result.add_optimization(
                f"Remove {default_count} unnecessary default attributes",
                impact="medium"
            )

    def _calculate_complexity(self, root: ET.Element, result: ValidationResult) -> None:
        """
        Calculate complexity score for the SVG.
        
        Args:
            root: Root SVG element
            result: ValidationResult to update
        """
        # Factors affecting complexity
        element_count = result.metrics["element_count"]
        depth = result.metrics["depth"]
        unique_ids = result.metrics["unique_ids"]
        
        # Count special elements
        special_elements = 0
        special_tags = [
            'filter', 'mask', 'pattern', 'clipPath', 'symbol',
            'linearGradient', 'radialGradient'
        ]
        
        for tag in special_tags:
            elements = root.findall(f".//{{{SVG_NAMESPACE}}}{tag}")
            special_elements += len(elements)
            
        # Calculate complexity score
        complexity = element_count * 1.0 + depth * 5.0 + unique_ids * 2.0 + special_elements * 10.0
        
        # Normalize to 0-100 scale
        normalized_complexity = min(100, max(0, complexity / 1000 * 100))
        
        result.update_metrics("complexity_score", round(normalized_complexity, 1))
        
        # Add recommendation for complex SVGs
        if normalized_complexity > 70:
            result.add_optimization(
                "Consider simplifying SVG due to high complexity score",
                impact="high"
            )


# Batch validation utility for processing multiple SVGs efficiently
def batch_validate(
    svg_files: List[str],
    validator: Optional[SVGValidator] = None,
    parallel: bool = True
) -> Dict[str, ValidationResult]:
    """
    Batch validate multiple SVG files with parallelism.
    
    Args:
        svg_files: List of SVG file paths
        validator: SVGValidator instance or None to create default
        parallel: Whether to use parallel validation
        
    Returns:
        Dictionary mapping file paths to ValidationResults
    """
    # Create validator if not provided
    if validator is None:
        validator = SVGValidator()
        
    results = {}
    
    if parallel:
        # Use thread pool for parallel validation
        with ThreadPoolExecutor(max_workers=min(len(svg_files), validator.parallelism)) as executor:
            # Submit validation tasks
            futures = {
                executor.submit(validator.validate_file, file_path): file_path
                for file_path in svg_files
            }
            
            # Collect results
            for future in futures:
                file_path = futures[future]
                try:
                    results[file_path] = future.result()
                except Exception as e:
                    logger.exception(f"Error validating {file_path}: {e}")
                    result = ValidationResult()
                    result.add_error(f"Validation error: {str(e)}")
                    results[file_path] = result.finalize()
    else:
        # Sequential validation
        for file_path in svg_files:
            try:
                results[file_path] = validator.validate_file(file_path)
            except Exception as e:
                logger.exception(f"Error validating {file_path}: {e}")
                result = ValidationResult()
                result.add_error(f"Validation error: {str(e)}")
                results[file_path] = result.finalize()
                
    return results


# Utility function to fix common SVG issues
def fix_common_issues(svg_content: str) -> Tuple[str, List[str]]:
    """
    Fix common SVG issues automatically.
    
    Args:
        svg_content: SVG content to fix
        
    Returns:
        Tuple of (fixed SVG content, list of applied fixes)
    """
    fixes = []
    fixed_content = svg_content
    
    # Add XML declaration if missing
    if not fixed_content.strip().startswith('<?xml'):
        fixed_content = '<?xml version="1.0" encoding="UTF-8"?>\n' + fixed_content
        fixes.append("Added XML declaration")
        
    # Add SVG namespace if missing
    svg_ns_pattern = r'<svg[^>]*\s+xmlns=[\'"](http://www\.w3\.org/2000/svg)[\'"]'
    if not re.search(svg_ns_pattern, fixed_content):
        fixed_content = re.sub(
            r'<svg',
            f'<svg xmlns="{SVG_NAMESPACE}"',
            fixed_content
        )
        fixes.append("Added SVG namespace")
        
    # Fix missing viewBox if width and height are present
    width_height_pattern = r'<svg[^>]*\s+width=[\'"](\d+)(?:px)?[\'"][^>]*\s+height=[\'"](\d+)(?:px)?[\'"]'
    viewbox_pattern = r'<svg[^>]*\s+viewBox=[\'"][^\'"]*[\'"]'
    
    if not re.search(viewbox_pattern, fixed_content):
        width_height_match = re.search(width_height_pattern, fixed_content)
        if width_height_match:
            width, height = width_height_match.groups()
            fixed_content = re.sub(
                r'<svg',
                f'<svg viewBox="0 0 {width} {height}"',
                fixed_content
            )
            fixes.append(f"Added viewBox='0 0 {width} {height}' based on width/height")
            
    # Remove unnecessary decimal places
    original_length = len(fixed_content)
    fixed_content = re.sub(r'(\d+\.\d{4})\d+', r'\1', fixed_content)
    if len(fixed_content) < original_length:
        fixes.append("Reduced decimal precision to 4 places")
        
    # Remove empty groups
    original_length = len(fixed_content)
    fixed_content = re.sub(r'<g[^>]*>\s*</g>', '', fixed_content)
    if len(fixed_content) < original_length:
        fixes.append("Removed empty groups")
        
    # Remove comments
    original_length = len(fixed_content)
    fixed_content = re.sub(r'<!--.*?-->', '', fixed_content, flags=re.DOTALL)
    if len(fixed_content) < original_length:
        fixes.append("Removed comments")
        
    # Remove default attribute values
    default_attrs = [
        (r'\s+x="0"', ''),
        (r'\s+y="0"', ''),
        (r'\s+dx="0"', ''),
        (r'\s+dy="0"', ''),
        (r'\s+fill-opacity="1"', ''),
        (r'\s+stroke-opacity="1"', ''),
        (r'\s+stroke-width="1"', ''),
        (r'\s+stroke-linecap="butt"', ''),
        (r'\s+stroke-linejoin="miter"', '')
    ]
    
    for pattern, replacement in default_attrs:
        original_length = len(fixed_content)
        fixed_content = re.sub(pattern, replacement, fixed_content)
        if len(fixed_content) < original_length:
            attr_name = pattern.split('"')[0].strip()
            fixes.append(f"Removed default {attr_name} attributes")
            
    # Remove unnecessary whitespace
    original_length = len(fixed_content)
    fixed_content = re.sub(r'>\s+<', '><', fixed_content)
    if len(fixed_content) < original_length:
        fixes.append("Removed unnecessary whitespace")
        
    return fixed_content, fixes


# Export main validation function for direct use
def validate_svg(svg_content: str, **kwargs) -> Dict[str, Any]:
    """
    Validate SVG content and return results as a dictionary.
    
    Args:
        svg_content: SVG content to validate
        **kwargs: Additional arguments for SVGValidator
        
    Returns:
        Validation results as a dictionary
    """
    validator = SVGValidator(**kwargs)
    result = validator.validate(svg_content)
    return result.to_dict()