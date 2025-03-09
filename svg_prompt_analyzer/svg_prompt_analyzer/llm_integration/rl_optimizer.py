"""
RL Optimizer Module
================
This module provides reinforcement learning optimization for SVG generation.
It iteratively improves SVG quality through exploration and exploitation of
generation parameters.
"""

import os
import time
import random
import json
import logging
import threading
import heapq
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Set, Callable
from pathlib import Path

from svg_prompt_analyzer.llm_integration.llm_manager import LLMManager
from svg_prompt_analyzer.llm_integration.clip_evaluator import CLIPEvaluator
from svg_prompt_analyzer.models.scene import Scene

logger = logging.getLogger(__name__)


@dataclass
class OptimizationCandidate:
    """A candidate solution in the optimization process."""
    svg_code: str
    score: float
    generation_params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def __lt__(self, other):
        """Compare candidates based on score for heapq operations."""
        return self.score > other.score  # Use > for max-heap (highest score first)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "score": self.score,
            "generation_params": self.generation_params,
            "metadata": self.metadata,
            # Don't include SVG code in serialization to save space
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any], svg_code: str) -> 'OptimizationCandidate':
        """Create from dictionary and SVG code."""
        return cls(
            svg_code=svg_code,
            score=data["score"],
            generation_params=data["generation_params"],
            metadata=data["metadata"],
            id=data["id"]
        )


class RLOptimizer:
    """
    Reinforcement Learning Optimizer for SVG generation.
    Uses exploration and exploitation to iteratively improve SVG quality.
    """
    
    def __init__(self, 
                 llm_manager: Optional[LLMManager] = None,
                 clip_evaluator: Optional[CLIPEvaluator] = None,
                 max_iterations: int = 3,
                 population_size: int = 5,
                 exploration_rate: float = 0.3,
                 hall_of_fame_size: int = 50,
                 hall_of_fame_dir: str = "hall_of_fame",
                 cache_dir: str = ".cache/optimization",
                 use_caching: bool = True):
        """
        Initialize the RL optimizer.
        
        Args:
            llm_manager: LLM manager instance for model access
            clip_evaluator: CLIP evaluation instance for similarity scoring
            max_iterations: Maximum number of optimization iterations
            population_size: Size of candidate population per iteration
            exploration_rate: Rate of parameter exploration (0.0-1.0)
            hall_of_fame_size: Number of top candidates to keep in hall of fame
            hall_of_fame_dir: Directory to store hall of fame candidates
            cache_dir: Directory for caching optimization results
            use_caching: Whether to use caching
        """
        self.llm_manager = llm_manager or LLMManager()
        self.clip_evaluator = clip_evaluator or CLIPEvaluator()
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.exploration_rate = exploration_rate
        self.hall_of_fame_size = hall_of_fame_size
        self.hall_of_fame_dir = hall_of_fame_dir
        self.cache_dir = cache_dir
        self.use_caching = use_caching
        
        # Hall of Fame (top scoring candidates across all optimizations)
        self.hall_of_fame: List[OptimizationCandidate] = []
        self._hall_of_fame_lock = threading.Lock()
        
        # Create directories
        os.makedirs(hall_of_fame_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing Hall of Fame if available
        self._load_hall_of_fame()
        
    def optimize(self, 
                scene: Scene, 
                base_svg: str, 
                optimization_level: int = 1,
                callback: Optional[Callable[[int, float, str], None]] = None) -> Tuple[str, float]:
        """
        Optimize SVG generation using reinforcement learning.
        
        Args:
            scene: Scene object to optimize for
            base_svg: Initial SVG code to start optimization from
            optimization_level: Level of optimization effort (1-3)
            callback: Optional callback function called with (iteration, score, svg) after each improvement
            
        Returns:
            Tuple of (best_svg_code, best_score)
        """
        # Check if we can use the CLIP evaluator
        if not self.clip_evaluator.model_loaded and not self.clip_evaluator.load_model():
            logger.warning("CLIP evaluator not available, optimization will be limited")
            return base_svg, 0.0
            
        # Check cache if enabled
        if self.use_caching:
            cache_key = self._get_cache_key(scene)
            cached_result = self._load_from_cache(cache_key)
            if cached_result and cached_result["optimization_level"] >= optimization_level:
                logger.info(f"Using cached optimization result for scene {scene.id}")
                return cached_result["svg_code"], cached_result["score"]
        
        # Adjust optimization parameters based on level
        iterations, population = self._get_optimization_params(optimization_level)
        
        logger.info(f"Starting optimization for scene {scene.id} at level {optimization_level} "
                   f"({iterations} iterations, {population} candidates)")
        
        start_time = time.time()
        
        # Initial evaluation of base SVG
        base_score = self.clip_evaluator.compute_similarity(base_svg, scene.prompt)
        
        # Initialize best solution with base SVG
        best_svg = base_svg
        best_score = base_score
        best_params = self._get_default_parameters()
        
        logger.info(f"Initial score for scene {scene.id}: {base_score:.4f}")
        
        # Create initial population of candidates
        current_population = [
            OptimizationCandidate(
                svg_code=base_svg,
                score=base_score,
                generation_params=self._get_default_parameters(),
                metadata={"iteration": 0, "source": "base"}
            )
        ]
        
        # Keep track of all candidates for diversity
        all_candidates: List[OptimizationCandidate] = list(current_population)
        
        # Optimization loop
        for iteration in range(1, iterations + 1):
            iteration_start = time.time()
            logger.info(f"Starting iteration {iteration}/{iterations} for scene {scene.id}")
            
            # Create new candidates through exploration and exploitation
            new_candidates = self._generate_candidates(
                scene=scene,
                current_population=current_population,
                all_candidates=all_candidates,
                population_size=population,
                iteration=iteration
            )
            
            # Score new candidates
            self._score_candidates(new_candidates, scene.prompt)
            
            # Update all candidates pool
            all_candidates.extend(new_candidates)
            
            # Select best candidates for next iteration
            current_population = self._select_candidates(all_candidates, population)
            
            # Update best solution if improved
            if current_population and current_population[0].score > best_score:
                best_svg = current_population[0].svg_code
                best_score = current_population[0].score
                best_params = current_population[0].generation_params
                
                logger.info(f"New best score at iteration {iteration}: {best_score:.4f} "
                          f"(improvement: +{best_score - base_score:.4f})")
                
                # Call callback if provided
                if callback:
                    callback(iteration, best_score, best_svg)
                    
            # Update Hall of Fame
            self._update_hall_of_fame(current_population)
            
            iteration_time = time.time() - iteration_start
            logger.info(f"Completed iteration {iteration} in {iteration_time:.2f}s, "
                       f"current best score: {best_score:.4f}")
        
        optimization_time = time.time() - start_time
        improvement = best_score - base_score
        improvement_percent = improvement / base_score * 100 if base_score > 0 else 0
        
        logger.info(f"Optimization completed for scene {scene.id} in {optimization_time:.2f}s. "
                   f"Final score: {best_score:.4f} (improvement: +{improvement:.4f}, +{improvement_percent:.2f}%)")
        
        # Cache the result if improved and caching is enabled
        if self.use_caching and best_score > base_score:
            self._save_to_cache(
                cache_key=self._get_cache_key(scene),
                svg_code=best_svg,
                score=best_score,
                params=best_params,
                optimization_level=optimization_level
            )
            
        return best_svg, best_score
        
    def _get_optimization_params(self, optimization_level: int) -> Tuple[int, int]:
        """
        Get optimization parameters based on level.
        
        Args:
            optimization_level: Optimization level (1-3)
            
        Returns:
            Tuple of (iterations, population_size)
        """
        # Define parameter sets for each level
        params = {
            1: (1, 3),  # Level 1: Quick optimization (1 iteration, 3 candidates)
            2: (2, 5),  # Level 2: Standard optimization (2 iterations, 5 candidates)
            3: (3, 8)   # Level 3: Thorough optimization (3 iterations, 8 candidates)
        }
        
        # Default to level 1 if invalid
        level = max(1, min(3, optimization_level))
        return params.get(level, params[1])
        
    def _get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default generation parameters.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            "temperature": 0.7,         # Creative temperature
            "detail_level": 0.5,        # Level of visual detail (0.0-1.0)
            "style_weight": 0.5,        # Weight for style vs. content (0.0-1.0)
            "use_gradients": True,      # Use color gradients
            "use_patterns": True,       # Use patterns for textures
            "use_effects": True,        # Use special visual effects
            "semantic_focus": [],       # List of semantic elements to emphasize
            "color_palette": "default"  # Color palette to use
        }
        
    def _generate_candidates(self, 
                            scene: Scene,
                            current_population: List[OptimizationCandidate],
                            all_candidates: List[OptimizationCandidate],
                            population_size: int,
                            iteration: int) -> List[OptimizationCandidate]:
        """
        Generate new candidates through exploration and exploitation.
        
        Args:
            scene: Scene to generate for
            current_population: Current population of candidates
            all_candidates: All candidates seen so far
            population_size: Size of population to generate
            iteration: Current iteration number
            
        Returns:
            List of new candidate solutions
        """
        new_candidates = []
        
        # Use best candidate as parent
        parent = current_population[0] if current_population else None
        
        # 1. Try candidates from Hall of Fame with similar prompts
        if self.hall_of_fame:
            similar_candidates = self._find_similar_candidates(scene.prompt, max_candidates=2)
            for candidate in similar_candidates:
                # Adapt hall of fame candidate to current scene
                new_params = candidate.generation_params.copy()
                
                # Create adapted LLM prompt
                llm_input = self._create_adaptation_prompt(scene, candidate)
                
                # Generate new SVG based on adaptation
                new_svg = self._generate_svg_with_llm(scene, llm_input, new_params)
                
                if new_svg:
                    new_candidates.append(
                        OptimizationCandidate(
                            svg_code=new_svg,
                            score=0.0,  # Will be scored later
                            generation_params=new_params,
                            metadata={
                                "iteration": iteration,
                                "source": "hall_of_fame_adaptation",
                                "parent_id": candidate.id
                            }
                        )
                    )
        
        # 2. Generate candidates with varying parameters
        remaining_slots = population_size - len(new_candidates)
        explore_count = max(1, int(remaining_slots * self.exploration_rate))
        exploit_count = remaining_slots - explore_count
        
        # Exploration: Create candidates with random parameter variations
        for i in range(explore_count):
            new_params = self._mutate_parameters(
                parent.generation_params if parent else self._get_default_parameters(),
                mutation_strength=0.3
            )
            
            new_svg = self._generate_svg_with_llm(scene, None, new_params)
            
            if new_svg:
                new_candidates.append(
                    OptimizationCandidate(
                        svg_code=new_svg,
                        score=0.0,  # Will be scored later
                        generation_params=new_params,
                        metadata={
                            "iteration": iteration,
                            "source": "exploration"
                        }
                    )
                )
        
        # Exploitation: Refine best candidates
        if parent:
            for i in range(exploit_count):
                # Create a refined version with smaller parameter changes
                new_params = self._mutate_parameters(
                    parent.generation_params,
                    mutation_strength=0.1
                )
                
                # Create refinement prompt
                llm_input = self._create_refinement_prompt(scene, parent.svg_code)
                
                # Generate refined SVG
                new_svg = self._generate_svg_with_llm(scene, llm_input, new_params)
                
                if new_svg:
                    new_candidates.append(
                        OptimizationCandidate(
                            svg_code=new_svg,
                            score=0.0,  # Will be scored later
                            generation_params=new_params,
                            metadata={
                                "iteration": iteration,
                                "source": "exploitation"
                            }
                        )
                    )
        
        # If we failed to generate enough candidates, fill in with direct LLM generation
        if len(new_candidates) < population_size:
            remaining = population_size - len(new_candidates)
            logger.info(f"Generating {remaining} additional candidates via direct LLM generation")
            
            for i in range(remaining):
                # Create new parameters
                new_params = self._get_default_parameters()
                # Slightly adjust temperature for diversity
                new_params["temperature"] = 0.6 + (i * 0.1)
                
                new_svg = self._generate_svg_with_llm(scene, None, new_params)
                
                if new_svg:
                    new_candidates.append(
                        OptimizationCandidate(
                            svg_code=new_svg,
                            score=0.0,  # Will be scored later
                            generation_params=new_params,
                            metadata={
                                "iteration": iteration,
                                "source": "direct_generation"
                            }
                        )
                    )
                    
        return new_candidates
        
    def _find_similar_candidates(self, prompt: str, max_candidates: int = 3) -> List[OptimizationCandidate]:
        """
        Find candidates from Hall of Fame with similar prompts.
        
        Args:
            prompt: Prompt to find similar candidates for
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of similar candidates
        """
        # Simple heuristic: Look for keyword overlap
        prompt_keywords = set(prompt.lower().split())
        
        scored_candidates = []
        for candidate in self.hall_of_fame:
            if "original_prompt" in candidate.metadata:
                candidate_prompt = candidate.metadata["original_prompt"]
                candidate_keywords = set(candidate_prompt.lower().split())
                
                # Calculate Jaccard similarity (overlap / union)
                overlap = len(prompt_keywords.intersection(candidate_keywords))
                union = len(prompt_keywords.union(candidate_keywords))
                similarity = overlap / union if union > 0 else 0
                
                scored_candidates.append((similarity, candidate))
                
        # Sort by similarity score descending
        scored_candidates.sort(reverse=True)
        
        # Return top candidates
        return [candidate for _, candidate in scored_candidates[:max_candidates]]
        
    def _mutate_parameters(self, params: Dict[str, Any], mutation_strength: float) -> Dict[str, Any]:
        """
        Create a mutated version of generation parameters.
        
        Args:
            params: Original parameters
            mutation_strength: Strength of mutation (0.0-1.0)
            
        Returns:
            Mutated parameters
        """
        new_params = params.copy()
        
        # Randomly mutate parameters
        if random.random() < mutation_strength:
            # Modify temperature
            new_params["temperature"] = min(1.0, max(0.1, params["temperature"] + 
                                                  random.uniform(-0.2, 0.2) * mutation_strength))
                                                  
        if random.random() < mutation_strength:
            # Modify detail level
            new_params["detail_level"] = min(1.0, max(0.0, params["detail_level"] + 
                                                   random.uniform(-0.3, 0.3) * mutation_strength))
                                                   
        if random.random() < mutation_strength:
            # Modify style weight
            new_params["style_weight"] = min(1.0, max(0.0, params["style_weight"] + 
                                                   random.uniform(-0.3, 0.3) * mutation_strength))
                                                   
        if random.random() < mutation_strength * 0.5:
            # Toggle boolean features (less frequently)
            new_params["use_gradients"] = not params["use_gradients"]
            
        if random.random() < mutation_strength * 0.5:
            new_params["use_patterns"] = not params["use_patterns"]
            
        if random.random() < mutation_strength * 0.5:
            new_params["use_effects"] = not params["use_effects"]
            
        # Modify color palette
        if random.random() < mutation_strength:
            palettes = ["default", "vibrant", "pastel", "monochrome", "complementary"]
            new_params["color_palette"] = random.choice(palettes)
            
        return new_params
        
    def _create_adaptation_prompt(self, scene: Scene, candidate: OptimizationCandidate) -> str:
        """
        Create a prompt for adapting a Hall of Fame candidate to the current scene.
        
        Args:
            scene: Current scene
            candidate: Hall of Fame candidate to adapt
            
        Returns:
            LLM prompt for adaptation
        """
        original_prompt = candidate.metadata.get("original_prompt", "unknown")
        
        prompt = f"""You are an expert SVG illustrator. Adapt the following SVG design for a new prompt.

Original prompt: "{original_prompt}"
New prompt: "{scene.prompt}"

The SVG below was successful for the original prompt. Modify it to match the new prompt while
preserving the successful visual style and techniques.

Changes needed:
1. Update colors, shapes, and elements to match the new prompt
2. Keep the successful visual techniques from the original
3. Ensure the adaptation maintains high visual quality
4. Make sure the result is semantically appropriate for the new prompt

Original SVG:
```xml
{candidate.svg_code}
```

Create an adapted SVG that maintains the successful elements but matches the new prompt:
```xml
"""
        return prompt
        
    def _create_refinement_prompt(self, scene: Scene, svg_code: str) -> str:
        """
        Create a prompt for refining an SVG.
        
        Args:
            scene: Scene to refine for
            svg_code: Original SVG code
            
        Returns:
            LLM prompt for refinement
        """
        prompt = f"""You are an expert SVG illustrator. Refine the following SVG to improve its visual quality
and similarity to the prompt.

Prompt: "{scene.prompt}"

Current SVG:
```xml
{svg_code}
```

Refine this SVG to:
1. Improve visual fidelity to the prompt
2. Enhance details and clarity
3. Make colors more vibrant and appropriate
4. Add any missing elements or details
5. Optimize for CLIP similarity scoring

Create a refined version while maintaining the same basic structure:
```xml
"""
        return prompt
        
    def _generate_svg_with_llm(self, scene: Scene, 
                             prompt: Optional[str] = None, 
                             params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Generate SVG code using LLM with given parameters.
        
        Args:
            scene: Scene to generate for
            prompt: Optional custom prompt (if None, creates standard prompt)
            params: Generation parameters
            
        Returns:
            Generated SVG code or None if generation failed
        """
        try:
            # Ensure LLM manager is available
            if not self.llm_manager.load_model("svg_generator"):
                logger.error("Failed to load SVG generator model")
                return None
                
            # Create default parameters if not provided
            if params is None:
                params = self._get_default_parameters()
                
            # Create standard prompt if not provided
            if prompt is None:
                prompt = self._create_standard_generation_prompt(scene, params)
                
            # Generate SVG with LLM
            llm_response = self.llm_manager.generate(
                role="svg_generator",
                prompt=prompt,
                max_tokens=10000,
                temperature=params["temperature"],
                stop_sequences=["```"]
            )
            
            # Extract SVG code
            svg_code = self._extract_svg_code(llm_response)
            
            return svg_code
            
        except Exception as e:
            logger.error(f"Error generating SVG with LLM: {str(e)}")
            return None
            
    def _create_standard_generation_prompt(self, scene: Scene, params: Dict[str, Any]) -> str:
        """
        Create a standard prompt for SVG generation based on parameters.
        
        Args:
            scene: Scene to generate for
            params: Generation parameters
            
        Returns:
            LLM prompt for SVG generation
        """
        # Extract visual style guidance based on parameters
        style_guidance = []
        
        if params["detail_level"] < 0.3:
            style_guidance.append("Use a minimal, simple style with clean lines.")
        elif params["detail_level"] > 0.7:
            style_guidance.append("Create a highly detailed, intricate illustration.")
        else:
            style_guidance.append("Use a balanced level of detail.")
            
        if params["use_gradients"]:
            style_guidance.append("Use color gradients for depth and visual interest.")
            
        if params["use_patterns"]:
            style_guidance.append("Use patterns for textures and materials.")
            
        if params["use_effects"]:
            style_guidance.append("Add visual effects like shadows, glows, or highlights.")
            
        # Color palette guidance
        palette_guidance = {
            "default": "Use a natural, balanced color palette.",
            "vibrant": "Use vibrant, saturated colors for visual impact.",
            "pastel": "Use soft, pastel colors for a gentle look.",
            "monochrome": "Use variations of a single color for a cohesive look.",
            "complementary": "Use complementary colors for visual contrast."
        }
        
        style_guidance.append(palette_guidance.get(params["color_palette"], palette_guidance["default"]))
        
        # Create the prompt
        prompt = f"""You are an expert SVG illustrator. Create a high-quality SVG illustration for the following prompt.

Prompt: "{scene.prompt}"

Visual style guidance:
{' '.join(f'{i+1}. {guidance}' for i, guidance in enumerate(style_guidance))}

Focus on creating an SVG that:
1. Directly corresponds to the prompt
2. Has high visual fidelity and clarity
3. Contains appropriate level of detail
4. Uses effective colors and composition
5. Is optimized for CLIP similarity scoring

The SVG should be semantically meaningful and directly represent the elements in the prompt.

Create complete SVG code:
```xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{scene.width}" height="{scene.height}" viewBox="0 0 {scene.width} {scene.height}"
    xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <title>SVG illustration of {scene.prompt}</title>
    <desc>Generated from prompt: {scene.prompt}</desc>
    
    <!-- Definitions for patterns, gradients, etc. -->
    <defs>
"""
        return prompt
        
    def _extract_svg_code(self, llm_response: str) -> Optional[str]:
        """
        Extract SVG code from LLM response.
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            Extracted SVG code or None if extraction failed
        """
        # Look for SVG code in code blocks
        import re
        
        svg_match = re.search(r'```(?:xml|svg)?\s*((?:<\?xml|<svg).*?</svg>)', llm_response, re.DOTALL)
        if svg_match:
            return svg_match.group(1).strip()
            
        # If no explicit code block, try to find SVG tags directly
        svg_match = re.search(r'(?:<\?xml|<svg).*?</svg>', llm_response, re.DOTALL)
        if svg_match:
            return svg_match.group(0).strip()
            
        # If still no match, check if response starts with XML declaration or SVG tag
        if llm_response.strip().startswith(('<?xml', '<svg')):
            # Find the closing SVG tag
            if '</svg>' in llm_response:
                end_index = llm_response.rindex('</svg>') + 6
                return llm_response[:end_index].strip()
                
        logger.warning("Could not extract SVG code from LLM response")
        return None
        
    def _score_candidates(self, candidates: List[OptimizationCandidate], prompt: str) -> None:
        """
        Score candidates using CLIP similarity.
        
        Args:
            candidates: List of candidates to score
            prompt: Prompt for scoring
        """
        if not candidates:
            return
            
        try:
            # Prepare SVG-prompt pairs
            pairs = [(c.svg_code, prompt) for c in candidates]
            
            # Evaluate in batch
            scores = self.clip_evaluator.evaluate_batch(pairs)
            
            # Update candidate scores
            for i, score in enumerate(scores):
                candidates[i].score = score
                
        except Exception as e:
            logger.error(f"Error scoring candidates: {str(e)}")
            
            # Fall back to sequential scoring
            for candidate in candidates:
                try:
                    score = self.clip_evaluator.compute_similarity(candidate.svg_code, prompt)
                    candidate.score = score
                except:
                    candidate.score = 0.0
                    
    def _select_candidates(self, 
                          candidates: List[OptimizationCandidate], 
                          population_size: int) -> List[OptimizationCandidate]:
        """
        Select best candidates for next iteration.
        
        Args:
            candidates: Pool of all candidates
            population_size: Number of candidates to select
            
        Returns:
            List of selected candidates
        """
        # Sort candidates by score (descending)
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        
        # Simple elitism: keep best candidates
        return sorted_candidates[:population_size]
        
    def _update_hall_of_fame(self, candidates: List[OptimizationCandidate]) -> None:
        """
        Update Hall of Fame with new candidates.
        
        Args:
            candidates: List of candidates to consider
        """
        with self._hall_of_fame_lock:
            # Only consider candidates with scores above threshold
            threshold = 0.5  # Minimum score to be considered for Hall of Fame
            
            for candidate in candidates:
                if candidate.score > threshold:
                    # Generate a unique ID if not present
                    if not candidate.id:
                        candidate.id = f"hof_{int(time.time())}_{len(self.hall_of_fame)}"
                        
                    # Add metadata about original prompt if not present
                    if "original_prompt" not in candidate.metadata and "prompt" in candidate.metadata:
                        candidate.metadata["original_prompt"] = candidate.metadata["prompt"]
                        
                    # Add to Hall of Fame
                    self.hall_of_fame.append(candidate)
                    
                    # Save candidate to disk
                    self._save_candidate_to_disk(candidate)
            
            # Keep only top candidates
            if len(self.hall_of_fame) > self.hall_of_fame_size:
                self.hall_of_fame = heapq.nlargest(self.hall_of_fame_size, self.hall_of_fame, key=lambda c: c.score)
                
    def _save_candidate_to_disk(self, candidate: OptimizationCandidate) -> None:
        """
        Save Hall of Fame candidate to disk.
        
        Args:
            candidate: Candidate to save
        """
        try:
            # Create directory structure
            hof_dir = Path(self.hall_of_fame_dir)
            hof_dir.mkdir(exist_ok=True)
            
            # Save SVG file
            svg_path = hof_dir / f"{candidate.id}.svg"
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(candidate.svg_code)
                
            # Save metadata
            metadata_path = hof_dir / f"{candidate.id}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(candidate.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving candidate to disk: {str(e)}")
            
    def _load_hall_of_fame(self) -> None:
        """Load Hall of Fame from disk."""
        try:
            hof_dir = Path(self.hall_of_fame_dir)
            if not hof_dir.exists():
                return
                
            # Find all JSON metadata files
            metadata_files = list(hof_dir.glob("*.json"))
            
            loaded_candidates = []
            for metadata_path in metadata_files:
                try:
                    # Load metadata
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        
                    # Check for corresponding SVG file
                    svg_path = hof_dir / f"{metadata['id']}.svg"
                    if not svg_path.exists():
                        continue
                        
                    # Load SVG content
                    with open(svg_path, 'r', encoding='utf-8') as f:
                        svg_code = f.read()
                        
                    # Create candidate
                    candidate = OptimizationCandidate.from_dict(metadata, svg_code)
                    loaded_candidates.append(candidate)
                    
                except Exception as e:
                    logger.error(f"Error loading candidate {metadata_path}: {str(e)}")
                    
            # Update Hall of Fame with loaded candidates
            if loaded_candidates:
                # Keep top candidates by score
                self.hall_of_fame = heapq.nlargest(
                    self.hall_of_fame_size, 
                    loaded_candidates, 
                    key=lambda c: c.score
                )
                
                logger.info(f"Loaded {len(self.hall_of_fame)} candidates to Hall of Fame")
                
        except Exception as e:
            logger.error(f"Error loading Hall of Fame: {str(e)}")
            
    def _get_cache_key(self, scene: Scene) -> str:
        """Generate a cache key for the scene."""
        import hashlib
        # Create a deterministic hash of the scene information
        scene_info = f"{scene.id}:{scene.prompt}"
        return hashlib.md5(scene_info.encode()).hexdigest()
        
    def _save_to_cache(self, 
                      cache_key: str, 
                      svg_code: str, 
                      score: float,
                      params: Dict[str, Any],
                      optimization_level: int) -> None:
        """
        Save optimization result to cache.
        
        Args:
            cache_key: Cache key
            svg_code: Optimized SVG code
            score: Optimization score
            params: Generation parameters
            optimization_level: Optimization level
        """
        try:
            if not self.use_caching:
                return
                
            # Prepare cache data
            cache_data = {
                "svg_code": svg_code,
                "score": score,
                "params": params,
                "optimization_level": optimization_level,
                "timestamp": time.time()
            }
            
            # Save JSON metadata
            metadata_path = Path(self.cache_dir) / f"{cache_key}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                # Don't include SVG code in JSON to keep it small
                metadata = {k: v for k, v in cache_data.items() if k != "svg_code"}
                json.dump(metadata, f)
                
            # Save SVG content
            svg_path = Path(self.cache_dir) / f"{cache_key}.svg"
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg_code)
                
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
            
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Load optimization result from cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        try:
            if not self.use_caching:
                return None
                
            # Check for metadata file
            metadata_path = Path(self.cache_dir) / f"{cache_key}.json"
            if not metadata_path.exists():
                return None
                
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # Check for SVG file
            svg_path = Path(self.cache_dir) / f"{cache_key}.svg"
            if not svg_path.exists():
                return None
                
            # Load SVG content
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_code = f.read()
                
            # Combine metadata and SVG code
            result = metadata.copy()
            result["svg_code"] = svg_code
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading from cache: {str(e)}")
            return None