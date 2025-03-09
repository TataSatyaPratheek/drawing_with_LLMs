"""
Multi-Objective Optimizer Module
=========================
This module provides functionality for multi-objective optimization of
SVG generation, balancing quality, size, performance, and visual complexity.
"""

import os
import time
import logging
import json
import random
import numpy as np
import threading
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from heapq import nlargest

from svg_prompt_analyzer.llm_integration.llm_manager import LLMManager
from svg_prompt_analyzer.llm_integration.clip_evaluator import CLIPEvaluator
from svg_prompt_analyzer.models.scene import Scene

logger = logging.getLogger(__name__)


@dataclass
class Solution:
    """Class representing a solution in multi-objective optimization."""
    svg_code: str
    objectives: Dict[str, float]
    parameters: Dict[str, Any]
    dominated_by: Set[int] = field(default_factory=set)
    dominates: Set[int] = field(default_factory=set)
    rank: int = 0
    crowding_distance: float = 0.0
    id: Optional[int] = None
    
    def __lt__(self, other: 'Solution') -> bool:
        """
        Compare solutions for sorting.
        
        Args:
            other: Solution to compare with
            
        Returns:
            True if self is better than other
        """
        # Primary sort by rank (lower is better)
        if self.rank != other.rank:
            return self.rank < other.rank
        
        # Secondary sort by crowding distance (higher is better)
        return self.crowding_distance > other.crowding_distance


class MultiObjectiveOptimizer:
    """
    Optimizer that balances multiple objectives for SVG generation.
    Implements NSGA-II for finding Pareto-optimal solutions.
    """
    
    def __init__(self,
                 llm_manager: Optional[LLMManager] = None,
                 clip_evaluator: Optional[CLIPEvaluator] = None,
                 objectives: Optional[Dict[str, Tuple[float, bool]]] = None,
                 reference_point: Optional[Dict[str, float]] = None,
                 population_size: int = 20,
                 archive_size: int = 50,
                 max_generations: int = 5,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.7,
                 adaptation_rate: float = 0.1,
                 scalarization_method: str = "weighted_sum",
                 use_reference_adaptation: bool = True,
                 cache_dir: str = ".cache/multi_objective",
                 use_caching: bool = True):
        """
        Initialize the multi-objective optimizer.
        
        Args:
            llm_manager: LLM manager for SVG generation
            clip_evaluator: CLIP evaluator for similarity scoring
            objectives: Dictionary mapping objective names to (weight, is_higher_better) tuples
            reference_point: Reference point for hypervolume calculation
            population_size: Size of population per generation
            archive_size: Size of non-dominated solution archive
            max_generations: Maximum number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            adaptation_rate: Rate for adapting reference point
            scalarization_method: Method for scalarizing multi-objective values
            use_reference_adaptation: Whether to adapt reference point
            cache_dir: Directory for caching results
            use_caching: Whether to use caching
        """
        self.llm_manager = llm_manager or LLMManager()
        self.clip_evaluator = clip_evaluator or CLIPEvaluator()
        
        # Set default objectives if not provided
        if objectives is None:
            self.objectives = {
                "clip_similarity": (0.6, True),    # CLIP similarity (higher is better)
                "svg_size": (0.2, False),          # SVG size (lower is better)
                "visual_complexity": (0.1, True),  # Visual complexity (higher is better)
                "rendering_performance": (0.1, False)  # Rendering performance (lower is better)
            }
        else:
            self.objectives = objectives
            
        # Set default reference point if not provided
        if reference_point is None:
            self.reference_point = {
                "clip_similarity": 0.0,           # Minimum possible similarity
                "svg_size": 20000,                # Maximum reasonable size in bytes
                "visual_complexity": 0.0,         # Minimum complexity
                "rendering_performance": 1000     # Maximum reasonable rendering time in ms
            }
        else:
            self.reference_point = reference_point
            
        self.population_size = population_size
        self.archive_size = archive_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.adaptation_rate = adaptation_rate
        self.scalarization_method = scalarization_method
        self.use_reference_adaptation = use_reference_adaptation
        
        # Create cache directory
        self.cache_dir = cache_dir
        self.use_caching = use_caching
        if self.use_caching:
            os.makedirs(cache_dir, exist_ok=True)
            
        # Initialize solution archive
        self.archive: List[Solution] = []
        
        # Initialize parameter ranges
        self.parameter_ranges = {
            "temperature": (0.1, 1.0),
            "detail_level": (0.1, 1.0),
            "style_weight": (0.1, 1.0),
            "use_gradients": (0, 1),  # Boolean
            "use_patterns": (0, 1),   # Boolean
            "use_effects": (0, 1),    # Boolean
            "palette_index": (0, 4),  # Index into color palette options
            "layout_index": (0, 3)    # Index into layout options
        }
        
        # Define color palettes
        self.color_palettes = [
            "natural",      # Natural/balanced colors
            "vibrant",      # Vibrant/saturated colors
            "monochrome",   # Monochromatic colors
            "pastel",       # Soft pastel colors
            "complementary" # Complementary colors
        ]
        
        # Define layout options
        self.layout_options = [
            "balanced",     # Balanced composition
            "centered",     # Centered focal point
            "rule_of_thirds", # Rule of thirds composition
            "dynamic"       # Dynamic/asymmetric composition
        ]
        
        logger.info(f"Multi-Objective Optimizer initialized with {len(self.objectives)} objectives")
    
    def optimize(self, 
                scene: Scene, 
                base_svg: str,
                callback: Optional[Callable[[int, Dict[str, float], str], None]] = None) -> Tuple[str, Dict[str, float]]:
        """
        Optimize SVG generation using multi-objective optimization.
        
        Args:
            scene: Scene to optimize
            base_svg: Initial SVG code
            callback: Optional callback function called with (generation, objectives, svg)
            
        Returns:
            Tuple of (best_svg_code, objective_values)
        """
        logger.info(f"Starting multi-objective optimization for scene {scene.id}")
        
        # Check cache if enabled
        if self.use_caching:
            cache_key = self._get_cache_key(scene)
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached optimization result for scene {scene.id}")
                return cached_result["svg_code"], cached_result["objectives"]
        
        # Ensure CLIP evaluator is loaded
        if not self.clip_evaluator.model_loaded:
            self.clip_evaluator.load_model()
        
        # Initialize timing
        start_time = time.time()
        
        # Evaluate base SVG
        base_objectives = self._evaluate_objectives(base_svg, scene.prompt)
        
        # Create initial population
        population = self._initialize_population(scene, base_svg, base_objectives)
        
        # Initialize best solution
        best_solution = self._select_best_solution(population)
        
        # Run optimization for multiple generations
        for generation in range(self.max_generations):
            generation_start = time.time()
            logger.info(f"Starting generation {generation+1}/{self.max_generations}")
            
            # Select parents for next generation
            parents = self._select_parents(population)
            
            # Create offspring through crossover and mutation
            offspring = self._create_offspring(scene, parents)
            
            # Evaluate offspring
            self._evaluate_population(offspring, scene.prompt)
            
            # Combine parents and offspring
            combined = population + offspring
            
            # Non-dominated sorting and crowding distance
            ranked_population = self._non_dominated_sort(combined)
            
            # Select new population
            population = self._select_population(ranked_population)
            
            # Update archive with non-dominated solutions
            self._update_archive(population)
            
            # Update reference point if enabled
            if self.use_reference_adaptation:
                self._adapt_reference_point(population)
            
            # Find best solution in this generation
            current_best = self._select_best_solution(population)
            
            # Update best overall solution if better
            if self._is_better_solution(current_best, best_solution):
                best_solution = current_best
                
                # Call callback if provided
                if callback:
                    callback(generation + 1, best_solution.objectives, best_solution.svg_code)
            
            generation_time = time.time() - generation_start
            logger.info(f"Completed generation {generation+1} in {generation_time:.2f}s")
            
            # Log objective values of best solution
            objectives_str = ", ".join([f"{name}: {value:.4f}" for name, value in best_solution.objectives.items()])
            logger.info(f"Current best solution: {objectives_str}")
        
        total_time = time.time() - start_time
        logger.info(f"Optimization completed in {total_time:.2f}s")
        
        # Cache result if enabled
        if self.use_caching:
            self._save_to_cache(cache_key, best_solution.svg_code, best_solution.objectives)
        
        return best_solution.svg_code, best_solution.objectives
    
    def _initialize_population(self, scene: Scene, base_svg: str, base_objectives: Dict[str, float]) -> List[Solution]:
        """
        Initialize population with random solutions.
        
        Args:
            scene: Scene to optimize
            base_svg: Initial SVG code
            base_objectives: Objective values for base SVG
            
        Returns:
            List of initial solutions
        """
        population = []
        
        # Add base solution to population
        base_solution = Solution(
            svg_code=base_svg,
            objectives=base_objectives,
            parameters=self._get_default_parameters(),
            id=0
        )
        population.append(base_solution)
        
        # Create random solutions for the rest of the population
        for i in range(1, self.population_size):
            # Generate random parameters
            params = self._generate_random_parameters()
            
            # Generate SVG with these parameters
            svg_code = self._generate_svg(scene, params)
            
            # Evaluate objectives
            objectives = self._evaluate_objectives(svg_code, scene.prompt)
            
            # Create solution
            solution = Solution(
                svg_code=svg_code,
                objectives=objectives,
                parameters=params,
                id=i
            )
            
            population.append(solution)
        
        return population
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default generation parameters.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            "temperature": 0.7,
            "detail_level": 0.5,
            "style_weight": 0.5,
            "use_gradients": True,
            "use_patterns": True,
            "use_effects": True,
            "palette_index": 0,
            "layout_index": 0
        }
    
    def _generate_random_parameters(self) -> Dict[str, Any]:
        """
        Generate random parameters within valid ranges.
        
        Returns:
            Dictionary of random parameters
        """
        params = {}
        
        for param, (min_val, max_val) in self.parameter_ranges.items():
            if param in ["use_gradients", "use_patterns", "use_effects"]:
                # Boolean parameters
                params[param] = random.choice([True, False])
            elif param in ["palette_index", "layout_index"]:
                # Integer parameters
                params[param] = random.randint(min_val, max_val)
            else:
                # Continuous parameters
                params[param] = min_val + random.random() * (max_val - min_val)
        
        return params
    
    def _generate_svg(self, scene: Scene, params: Dict[str, Any]) -> str:
        """
        Generate SVG using the provided parameters.
        
        Args:
            scene: Scene to generate for
            params: Generation parameters
            
        Returns:
            Generated SVG code
        """
        # Create generation prompt
        prompt = self._create_generation_prompt(scene, params)
        
        # Generate SVG using LLM
        svg_code = self.llm_manager.generate(
            role="svg_generator",
            prompt=prompt,
            max_tokens=10000,
            temperature=params["temperature"],
            stop_sequences=["```"]
        )
        
        # Extract SVG code
        return self._extract_svg_code(svg_code)
    
    def _create_generation_prompt(self, scene: Scene, params: Dict[str, Any]) -> str:
        """
        Create a generation prompt based on parameters.
        
        Args:
            scene: Scene to generate for
            params: Generation parameters
            
        Returns:
            Generation prompt
        """
        # Extract style guidance based on parameters
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
        
        # Add color palette guidance
        palette_index = min(int(params["palette_index"]), len(self.color_palettes) - 1)
        palette = self.color_palettes[palette_index]
        if palette == "natural":
            style_guidance.append("Use a natural, balanced color palette.")
        elif palette == "vibrant":
            style_guidance.append("Use vibrant, saturated colors for visual impact.")
        elif palette == "monochrome":
            style_guidance.append("Use variations of a single color for a cohesive look.")
        elif palette == "pastel":
            style_guidance.append("Use soft, pastel colors for a gentle look.")
        elif palette == "complementary":
            style_guidance.append("Use complementary colors for visual contrast.")
        
        # Add layout guidance
        layout_index = min(int(params["layout_index"]), len(self.layout_options) - 1)
        layout = self.layout_options[layout_index]
        if layout == "balanced":
            style_guidance.append("Create a balanced, symmetric composition.")
        elif layout == "centered":
            style_guidance.append("Create a centered composition with a clear focal point.")
        elif layout == "rule_of_thirds":
            style_guidance.append("Apply the rule of thirds for a dynamic composition.")
        elif layout == "dynamic":
            style_guidance.append("Create a dynamic, asymmetric composition.")
        
        # Style weight affects the balance of style vs. content
        if params["style_weight"] < 0.3:
            style_guidance.append("Prioritize accurate representation over stylistic elements.")
        elif params["style_weight"] > 0.7:
            style_guidance.append("Prioritize visual style and aesthetic appeal.")
        
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
    
    def _extract_svg_code(self, llm_response: str) -> str:
        """
        Extract SVG code from LLM response.
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            Extracted SVG code
        """
        import re
        
        # Look for SVG code in code blocks
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
        
        # Return a minimal valid SVG as fallback
        return f"""
<svg width="{800}" height="{600}" viewBox="0 0 800 600"
    xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#F5F5F5" />
    <text x="400" y="300" text-anchor="middle" font-size="24" fill="#333">
        {scene.prompt if hasattr(scene, 'prompt') else "Fallback SVG"}
    </text>
</svg>
"""
    
    def _evaluate_objectives(self, svg_code: str, prompt: str) -> Dict[str, float]:
        """
        Evaluate objective functions for an SVG.
        
        Args:
            svg_code: SVG code to evaluate
            prompt: Prompt text
            
        Returns:
            Dictionary of objective values
        """
        objectives = {}
        
        # Evaluate CLIP similarity
        try:
            if self.clip_evaluator.model_loaded:
                clip_score = self.clip_evaluator.compute_similarity(svg_code, prompt)
            else:
                clip_score = random.uniform(0.3, 0.7)  # Mock score if CLIP not available
            objectives["clip_similarity"] = clip_score
        except Exception as e:
            logger.error(f"Error evaluating CLIP similarity: {str(e)}")
            objectives["clip_similarity"] = 0.3  # Default value on error
        
        # Evaluate SVG size
        objectives["svg_size"] = len(svg_code) / 1000  # Size in KB
        
        # Evaluate visual complexity
        objectives["visual_complexity"] = self._calculate_visual_complexity(svg_code)
        
        # Evaluate rendering performance (simplified)
        objectives["rendering_performance"] = self._estimate_rendering_performance(svg_code)
        
        return objectives
    
    def _calculate_visual_complexity(self, svg_code: str) -> float:
        """
        Calculate visual complexity of an SVG.
        
        Args:
            svg_code: SVG code to analyze
            
        Returns:
            Complexity score (0.0-1.0)
        """
        # Simple heuristics for complexity
        complexity = 0.0
        
        # Count SVG elements
        element_count = svg_code.count("<path") + svg_code.count("<rect") + \
                       svg_code.count("<circle") + svg_code.count("<ellipse") + \
                       svg_code.count("<polygon") + svg_code.count("<polyline") + \
                       svg_code.count("<line")
        
        # Count gradients
        gradient_count = svg_code.count("<linearGradient") + svg_code.count("<radialGradient")
        
        # Count filters
        filter_count = svg_code.count("<filter")
        
        # Count patterns
        pattern_count = svg_code.count("<pattern")
        
        # Calculate weighted complexity
        complexity = min(1.0, (
            element_count / 50 * 0.5 +   # Up to 50 elements (50% of score)
            gradient_count / 5 * 0.2 +   # Up to 5 gradients (20% of score)
            filter_count / 3 * 0.2 +     # Up to 3 filters (20% of score)
            pattern_count / 3 * 0.1      # Up to 3 patterns (10% of score)
        ))
        
        return complexity
    
    def _estimate_rendering_performance(self, svg_code: str) -> float:
        """
        Estimate rendering performance of an SVG.
        
        Args:
            svg_code: SVG code to analyze
            
        Returns:
            Performance score (higher is worse)
        """
        # Simple heuristics for performance
        performance = 0.0
        
        # Count complex elements that impact performance
        path_count = svg_code.count("<path")
        filter_count = svg_code.count("<filter")
        gradient_count = svg_code.count("<linearGradient") + svg_code.count("<radialGradient")
        
        # Estimate performance impact
        # Scale from 0 to 1, where 0 is best performance and 1 is worst
        performance = min(1.0, (
            path_count / 50 * 0.4 +      # Path complexity
            filter_count / 3 * 0.4 +     # Filter complexity
            gradient_count / 5 * 0.2     # Gradient complexity
        ))
        
        return performance
    
    def _evaluate_population(self, population: List[Solution], prompt: str) -> None:
        """
        Evaluate objective functions for a population.
        
        Args:
            population: List of solutions to evaluate
            prompt: Prompt text
        """
        for solution in population:
            # Skip if already evaluated
            if solution.objectives:
                continue
                
            # Evaluate objectives
            solution.objectives = self._evaluate_objectives(solution.svg_code, prompt)
    
    def _select_parents(self, population: List[Solution]) -> List[Solution]:
        """
        Select parents for the next generation using tournament selection.
        
        Args:
            population: Current population
            
        Returns:
            Selected parents
        """
        parents = []
        
        # Use tournament selection
        tournament_size = 3
        
        for _ in range(self.population_size):
            # Select random candidates for tournament
            candidates = random.sample(population, min(tournament_size, len(population)))
            
            # Select the best candidate as parent
            best_candidate = min(candidates)  # Solution.__lt__ handles comparison
            parents.append(best_candidate)
        
        return parents
    
    def _create_offspring(self, scene: Scene, parents: List[Solution]) -> List[Solution]:
        """
        Create offspring through crossover and mutation.
        
        Args:
            scene: Scene to generate for
            parents: Parent solutions
            
        Returns:
            Offspring solutions
        """
        offspring = []
        
        # Generate offspring until we have enough
        while len(offspring) < self.population_size:
            # Select two parents
            parent1, parent2 = random.sample(parents, 2)
            
            # Apply crossover with probability
            if random.random() < self.crossover_rate:
                child_params = self._crossover(parent1.parameters, parent2.parameters)
            else:
                # No crossover, use parameters from one parent
                child_params = parent1.parameters.copy()
                
            # Apply mutation with probability
            if random.random() < self.mutation_rate:
                child_params = self._mutate(child_params)
                
            # Generate SVG using parameters
            child_svg = self._generate_svg(scene, child_params)
            
            # Create solution (objectives will be evaluated later)
            child = Solution(
                svg_code=child_svg,
                objectives={},
                parameters=child_params,
                id=len(offspring) + self.population_size  # Assign unique ID
            )
            
            offspring.append(child)
            
            # Stop if we have enough offspring
            if len(offspring) >= self.population_size:
                break
        
        return offspring
    
    def _crossover(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform crossover between two parameter sets.
        
        Args:
            params1: First parameter set
            params2: Second parameter set
            
        Returns:
            New parameter set
        """
        child_params = {}
        
        for param in params1.keys():
            # Randomly select from either parent
            if random.random() < 0.5:
                child_params[param] = params1[param]
            else:
                child_params[param] = params2[param]
        
        return child_params
    
    def _mutate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate parameters randomly.
        
        Args:
            params: Parameter set to mutate
            
        Returns:
            Mutated parameter set
        """
        mutated_params = params.copy()
        
        # Select a random parameter to mutate
        param = random.choice(list(params.keys()))
        
        if param in self.parameter_ranges:
            min_val, max_val = self.parameter_ranges[param]
            
            if param in ["use_gradients", "use_patterns", "use_effects"]:
                # Toggle boolean parameter
                mutated_params[param] = not params[param]
            elif param in ["palette_index", "layout_index"]:
                # Mutate integer parameter
                current_val = params[param]
                # Either increment or decrement, with wrapping
                if random.random() < 0.5:
                    mutated_params[param] = (current_val + 1) % (max_val + 1)
                else:
                    mutated_params[param] = (current_val - 1) % (max_val + 1)
            else:
                # Mutate continuous parameter
                # Add random noise, clamped to valid range
                noise = (random.random() * 2 - 1) * 0.2  # ±20% noise
                mutated_params[param] = max(min_val, min(max_val, params[param] + noise))
        
        return mutated_params
    
    def _non_dominated_sort(self, population: List[Solution]) -> List[Solution]:
        """
        Perform non-dominated sorting on the population.
        
        Args:
            population: Population to sort
            
        Returns:
            Sorted population with rank and crowding distance
        """
        # Reset dominance information
        for solution in population:
            solution.dominates = set()
            solution.dominated_by = set()
            solution.rank = 0
            solution.crowding_distance = 0.0
        
        # Compute dominance relations
        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i == j:
                    continue
                    
                if self._dominates(p, q):
                    p.dominates.add(j)
                    q.dominated_by.add(i)
        
        # Compute ranks
        current_rank = 0
        ranked_solutions = []
        
        # Initial front (rank 0)
        current_front = [i for i, p in enumerate(population) if len(p.dominated_by) == 0]
        
        while current_front:
            # Assign current rank to solutions in current front
            for i in current_front:
                population[i].rank = current_rank
                
            # Add current front to ranked solutions
            ranked_solutions.extend([population[i] for i in current_front])
            
            # Prepare next front
            next_front = []
            for i in current_front:
                for j in population[i].dominates:
                    population[j].dominated_by.remove(i)
                    if len(population[j].dominated_by) == 0:
                        next_front.append(j)
            
            # Move to next front
            current_front = next_front
            current_rank += 1
        
        # Calculate crowding distance for each rank
        for rank in range(current_rank):
            rank_solutions = [s for s in population if s.rank == rank]
            self._calculate_crowding_distance(rank_solutions)
        
        # Sort by rank and crowding distance
        return sorted(population)
    
    def _dominates(self, p: Solution, q: Solution) -> bool:
        """
        Check if solution p dominates solution q.
        
        Args:
            p: First solution
            q: Second solution
            
        Returns:
            True if p dominates q
        """
        better_in_any = False
        
        for obj_name, (weight, is_higher_better) in self.objectives.items():
            # Skip if objective not present
            if obj_name not in p.objectives or obj_name not in q.objectives:
                continue
                
            p_val = p.objectives[obj_name]
            q_val = q.objectives[obj_name]
            
            # Compare based on whether higher is better
            if is_higher_better:
                if p_val < q_val:
                    return False  # p is worse in this objective
                if p_val > q_val:
                    better_in_any = True
            else:
                if p_val > q_val:
                    return False  # p is worse in this objective
                if p_val < q_val:
                    better_in_any = True
        
        return better_in_any
    
    def _calculate_crowding_distance(self, solutions: List[Solution]) -> None:
        """
        Calculate crowding distance for a set of solutions.
        
        Args:
            solutions: List of solutions
        """
        if len(solutions) <= 2:
            # If only one or two solutions, assign infinite crowding distance
            for s in solutions:
                s.crowding_distance = float('inf')
            return
            
        # Initialize crowding distance
        for s in solutions:
            s.crowding_distance = 0.0
            
        # Calculate crowding distance for each objective
        for obj_name, (weight, is_higher_better) in self.objectives.items():
            # Skip if objective not present
            if obj_name not in solutions[0].objectives:
                continue
                
            # Sort solutions by this objective
            sorted_solutions = sorted(solutions, key=lambda s: s.objectives[obj_name])
            
            # Edge solutions get infinite distance
            sorted_solutions[0].crowding_distance = float('inf')
            sorted_solutions[-1].crowding_distance = float('inf')
            
            # Calculate objective range
            obj_range = sorted_solutions[-1].objectives[obj_name] - sorted_solutions[0].objectives[obj_name]
            
            # Skip if range is zero
            if obj_range == 0:
                continue
                
            # Calculate crowding distance for interior solutions
            for i in range(1, len(sorted_solutions) - 1):
                distance = (sorted_solutions[i+1].objectives[obj_name] - 
                           sorted_solutions[i-1].objectives[obj_name]) / obj_range
                sorted_solutions[i].crowding_distance += distance * weight
    
    def _select_population(self, ranked_population: List[Solution]) -> List[Solution]:
        """
        Select the next population based on ranks and crowding distance.
        
        Args:
            ranked_population: Population sorted by rank and crowding distance
            
        Returns:
            Selected population
        """
        # Simply take the top population_size solutions
        return ranked_population[:self.population_size]
    
    def _update_archive(self, population: List[Solution]) -> None:
        """
        Update the archive of non-dominated solutions.
        
        Args:
            population: Current population
        """
        # Add non-dominated solutions to archive
        non_dominated = [s for s in population if s.rank == 0]
        
        # Combine with existing archive
        combined = self.archive + non_dominated
        
        # Remove duplicates (based on parameters)
        unique_combined = []
        param_hashes = set()
        
        for solution in combined:
            # Create a parameter hash for deduplication
            param_str = json.dumps(solution.parameters, sort_keys=True)
            param_hash = hash(param_str)
            
            if param_hash not in param_hashes:
                param_hashes.add(param_hash)
                unique_combined.append(solution)
        
        # Sort by rank and crowding distance
        sorted_combined = self._non_dominated_sort(unique_combined)
        
        # Keep only non-dominated solutions
        self.archive = [s for s in sorted_combined if s.rank == 0]
        
        # Trim archive if it gets too large
        if len(self.archive) > self.archive_size:
            # Sort by crowding distance (higher is better)
            self.archive.sort(key=lambda s: -s.crowding_distance)
            self.archive = self.archive[:self.archive_size]
    
    def _adapt_reference_point(self, population: List[Solution]) -> None:
        """
        Adapt the reference point based on current population.
        
        Args:
            population: Current population
        """
        # Skip if adaptation is disabled
        if not self.use_reference_adaptation:
            return
            
        # Find non-dominated solutions
        non_dominated = [s for s in population if s.rank == 0]
        
        # Skip if no non-dominated solutions
        if not non_dominated:
            return
            
        # Adapt reference point for each objective
        for obj_name, (weight, is_higher_better) in self.objectives.items():
            # Skip if objective not present
            if obj_name not in non_dominated[0].objectives:
                continue
                
            # Find best value for this objective
            if is_higher_better:
                best_value = max(s.objectives[obj_name] for s in non_dominated)
                
                # Adapt reference point upwards
                current_ref = self.reference_point.get(obj_name, 0.0)
                new_ref = current_ref + (best_value - current_ref) * self.adaptation_rate
                self.reference_point[obj_name] = new_ref
            else:
                best_value = min(s.objectives[obj_name] for s in non_dominated)
                
                # Adapt reference point downwards
                current_ref = self.reference_point.get(obj_name, 1.0)
                new_ref = current_ref - (current_ref - best_value) * self.adaptation_rate
                self.reference_point[obj_name] = new_ref
    
    def _select_best_solution(self, population: List[Solution]) -> Solution:
        """
        Select the best solution based on scalarization.
        
        Args:
            population: Population to select from
            
        Returns:
            Best solution
        """
        if not population:
            raise ValueError("Cannot select from empty population")
            
        # Calculate scalarized values
        scalarized_values = [self._scalarize(s.objectives) for s in population]
        
        # Find best solution
        best_index = scalarized_values.index(max(scalarized_values))
        return population[best_index]
    
    def _scalarize(self, objectives: Dict[str, float]) -> float:
        """
        Scalarize multiple objectives into a single value.
        
        Args:
            objectives: Dictionary of objective values
            
        Returns:
            Scalarized value
        """
        if self.scalarization_method == "weighted_sum":
            return self._weighted_sum(objectives)
        elif self.scalarization_method == "tchebycheff":
            return self._tchebycheff(objectives)
        else:
            # Default to weighted sum
            return self._weighted_sum(objectives)
    
    def _weighted_sum(self, objectives: Dict[str, float]) -> float:
        """
        Calculate weighted sum of objectives.
        
        Args:
            objectives: Dictionary of objective values
            
        Returns:
            Weighted sum
        """
        total = 0.0
        
        for obj_name, (weight, is_higher_better) in self.objectives.items():
            # Skip if objective not present
            if obj_name not in objectives:
                continue
                
            value = objectives[obj_name]
            
            # Adjust value if lower is better
            if not is_higher_better:
                # Transform to maximize
                if obj_name in self.reference_point:
                    # Use reference point for normalization
                    ref = self.reference_point[obj_name]
                    value = max(0.0, ref - value) / ref if ref > 0 else 0.0
                else:
                    # Simple inversion (assuming values are positive)
                    value = 1.0 / (1.0 + value)
            
            # Add weighted value
            total += weight * value
        
        return total
    
    def _tchebycheff(self, objectives: Dict[str, float]) -> float:
        """
        Calculate Tchebycheff scalarization.
        
        Args:
            objectives: Dictionary of objective values
            
        Returns:
            Tchebycheff value (negated for maximization)
        """
        max_term = float('-inf')
        
        for obj_name, (weight, is_higher_better) in self.objectives.items():
            # Skip if objective not present
            if obj_name not in objectives or obj_name not in self.reference_point:
                continue
                
            value = objectives[obj_name]
            ref = self.reference_point[obj_name]
            
            # Calculate distance from reference point
            if is_higher_better:
                # For maximization, distance is reference - value
                dist = max(0.0, ref - value)
            else:
                # For minimization, distance is value - reference
                dist = max(0.0, value - ref)
            
            # Scale by weight
            weighted_dist = weight * dist
            
            # Update maximum term
            max_term = max(max_term, weighted_dist)
        
        # Negate for maximization (lower is better for Tchebycheff)
        return -max_term
    
    def _is_better_solution(self, solution1: Solution, solution2: Solution) -> bool:
        """
        Check if solution1 is better than solution2 based on scalarization.
        
        Args:
            solution1: First solution
            solution2: Second solution
            
        Returns:
            True if solution1 is better
        """
        # Calculate scalarized values
        value1 = self._scalarize(solution1.objectives)
        value2 = self._scalarize(solution2.objectives)
        
        # Higher scalarized value is better
        return value1 > value2
    
    def _get_cache_key(self, scene: Scene) -> str:
        """
        Get cache key for a scene.
        
        Args:
            scene: Scene to get key for
            
        Returns:
            Cache key
        """
        # Create a unique key based on prompt and scene properties
        key_parts = [
            scene.prompt,
            str(scene.width),
            str(scene.height)
        ]
        
        # Join and hash
        import hashlib
        key = hashlib.md5("_".join(key_parts).encode()).hexdigest()
        
        return key
    
    def _save_to_cache(self, cache_key: str, svg_code: str, objectives: Dict[str, float]) -> None:
        """
        Save result to cache.
        
        Args:
            cache_key: Cache key
            svg_code: SVG code
            objectives: Objective values
        """
        if not self.use_caching:
            return
            
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            # Create cache entry
            cache_entry = {
                "svg_code": svg_code,
                "objectives": objectives,
                "timestamp": time.time()
            }
            
            # Save to file
            with open(cache_file, "w") as f:
                json.dump(cache_entry, f, indent=2)
                
            logger.debug(f"Saved result to cache: {cache_file}")
            
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Load result from cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached result or None
        """
        if not self.use_caching:
            return None
            
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            # Check if file exists
            if not os.path.exists(cache_file):
                return None
                
            # Load from file
            with open(cache_file, "r") as f:
                cache_entry = json.load(f)
                
            logger.debug(f"Loaded result from cache: {cache_file}")
            
            return cache_entry
            
        except Exception as e:
            logger.error(f"Error loading from cache: {str(e)}")
            return None
    
    def get_pareto_front(self) -> List[Solution]:
        """
        Get the current Pareto front from the archive.
        
        Returns:
            List of non-dominated solutions
        """
        return self.archive
    
    def calculate_hypervolume(self) -> float:
        """
        Calculate hypervolume indicator for the archive.
        
        Returns:
            Hypervolume value
        """
        # Skip if archive is empty
        if not self.archive:
            return 0.0
            
        # Simplified hypervolume calculation for 2D case
        # In a real implementation, you'd use a library like pygmo for this
        
        # Currently only implements 2D case for clip_similarity and svg_size
        if len(self.archive) < 2:
            return 0.0
            
        # Check if we have the required objectives
        if not all(("clip_similarity" in s.objectives and "svg_size" in s.objectives) for s in self.archive):
            return 0.0
            
        # Sort by clip_similarity (ascending)
        sorted_archive = sorted(self.archive, key=lambda s: s.objectives["clip_similarity"])
        
        # Calculate hypervolume
        hv = 0.0
        prev_x = self.reference_point.get("clip_similarity", 0.0)
        prev_y = sorted_archive[0].objectives["svg_size"]
        
        for s in sorted_archive:
            x = s.objectives["clip_similarity"]
            y = s.objectives["svg_size"]
            
            # Add area of rectangle
            width = x - prev_x
            height = prev_y - y  # Assuming lower svg_size is better
            
            if width > 0 and height > 0:
                hv += width * height
                
            # Update previous point
            prev_x = x
            if y < prev_y:  # Only update if y is better
                prev_y = y
        
        return hv
    
    def print_statistics(self) -> None:
        """Print statistics about the optimization process."""
        if not self.archive:
            print("No solutions in archive")
            return
            
        print(f"Archive size: {len(self.archive)}")
        print(f"Reference point: {self.reference_point}")
        
        # Print objective ranges
        for obj_name in self.objectives:
            if all(obj_name in s.objectives for s in self.archive):
                values = [s.objectives[obj_name] for s in self.archive]
                print(f"{obj_name}: min={min(values):.4f}, max={max(values):.4f}, avg={sum(values)/len(values):.4f}")
        
        # Print hypervolume
        hv = self.calculate_hypervolume()
        print(f"Hypervolume: {hv:.4f}")
        
        # Print best solution for each objective
        for obj_name, (weight, is_higher_better) in self.objectives.items():
            if all(obj_name in s.objectives for s in self.archive):
                if is_higher_better:
                    best_sol = max(self.archive, key=lambda s: s.objectives[obj_name])
                    comp = "max"
                else:
                    best_sol = min(self.archive, key=lambda s: s.objectives[obj_name])
                    comp = "min"
                    
                print(f"Best {obj_name} ({comp}): {best_sol.objectives[obj_name]:.4f}")