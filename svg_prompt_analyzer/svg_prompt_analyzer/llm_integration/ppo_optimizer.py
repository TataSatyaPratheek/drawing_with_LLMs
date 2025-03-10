"""
PPO Optimizer Module
======================
This module provides a full implementation of the Proximal Policy Optimization
(PPO) algorithm for reinforcement learning optimization of SVG generation.
It includes policy and value networks, advantage estimation, and policy updates.
"""

import os
import time
import logging
import json
import random
import numpy as np
import threading
import math
import gc
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Iterator
from pathlib import Path

# Import core optimizations
from svg_prompt_analyzer.core import CONFIG, memoize, jit, Profiler
from svg_prompt_analyzer.core.batch_processor import BatchProcessor
from svg_prompt_analyzer.core.memory_manager import MemoryManager
from svg_prompt_analyzer.core.hardware_manager import HardwareManager

# Import LLM manager and CLIP evaluator
from svg_prompt_analyzer.llm_integration.llm_manager import LLMManager, extract_svg_from_text
from svg_prompt_analyzer.llm_integration.clip_evaluator import CLIPEvaluator

# Import models
from svg_prompt_analyzer.models.scene import Scene

# Configure logger
logger = logging.getLogger(__name__)

# Get core component instances
memory_manager = MemoryManager()
hardware_manager = HardwareManager()


@dataclass
class Experience:
    """Class for storing experience data for PPO training."""
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    log_prob: float = 0.0
    value: float = 0.0


@dataclass
class PPOBatch:
    """Class for batched experience data."""
    states: List[Any]
    actions: List[Any]
    rewards: List[float]
    next_states: List[Any]
    dones: List[bool]
    log_probs: List[float]
    values: List[float]


@memoize
def _load_torch() -> Dict[str, Any]:
    """
    Lazily load PyTorch modules to reduce startup time and memory usage.
    Returns PyTorch modules as a dictionary.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torch.distributions import Categorical
        
        return {
            'torch': torch,
            'nn': nn,
            'F': F,
            'optim': optim,
            'Categorical': Categorical,
            'available': True
        }
    except ImportError as e:
        logger.warning(f"PyTorch import error: {e}")
        return {'available': False}


class PolicyNetwork:
    """Neural network for the policy in PPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space
            hidden_dim: Dimension of hidden layers
        """
        torch_modules = _load_torch()
        if not torch_modules['available']:
            raise ImportError("PyTorch is required for PolicyNetwork")
            
        nn = torch_modules['nn']
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Action probabilities
        """
        return self.network(state)


class ValueNetwork:
    """Neural network for the value function in PPO."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        """
        Initialize the value network.
        
        Args:
            state_dim: Dimensionality of state space
            hidden_dim: Dimension of hidden layers
        """
        torch_modules = _load_torch()
        if not torch_modules['available']:
            raise ImportError("PyTorch is required for ValueNetwork")
            
        nn = torch_modules['nn']
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Value estimate
        """
        return self.network(state)


class FeatureExtractor:
    """Extract features from scene and SVG for RL state representation."""
    
    def __init__(self, feature_dim: int = 64):
        """
        Initialize the feature extractor.
        
        Args:
            feature_dim: Dimension of feature vector
        """
        self.feature_dim = feature_dim
    
    @memoize
    def extract_features(self, scene: Scene, svg_code: str) -> np.ndarray:
        """
        Extract features from scene and SVG.
        
        Args:
            scene: Scene object
            svg_code: SVG code
            
        Returns:
            Feature vector
        """
        with Profiler("feature_extraction"):
            features = np.zeros(self.feature_dim)
            
            try:
                # Extract basic features
                features[0] = len(scene.objects)  # Number of objects
                features[1] = len(svg_code) / 10000  # Normalized SVG size
                
                # Extract color features
                color_counts = {}
                for obj in scene.objects:
                    if hasattr(obj, 'color') and obj.color:
                        color_name = obj.color.name
                        color_counts[color_name] = color_counts.get(color_name, 0) + 1
                
                features[2] = len(color_counts)  # Number of unique colors
                
                # Extract spatial features
                if scene.objects:
                    if hasattr(scene.objects[0], 'position'):
                        avg_x = sum(obj.position[0] for obj in scene.objects) / len(scene.objects)
                        avg_y = sum(obj.position[1] for obj in scene.objects) / len(scene.objects)
                        features[3] = avg_x
                        features[4] = avg_y
                
                # Extract complexity features
                features[5] = svg_code.count("<") / 100  # Number of tags
                features[6] = svg_code.count("path") / 10  # Number of paths
                
                # Count common SVG features
                features[7] = svg_code.count("<g") / 10  # Number of groups
                features[8] = svg_code.count("<rect") / 10  # Number of rectangles
                features[9] = svg_code.count("<circle") / 10  # Number of circles
                
                # Detect advanced SVG features
                features[10] = 1.0 if "linear-gradient" in svg_code else 0.0
                features[11] = 1.0 if "radial-gradient" in svg_code else 0.0
                features[12] = 1.0 if "<filter" in svg_code else 0.0
                features[13] = 1.0 if "<mask" in svg_code else 0.0
                features[14] = 1.0 if "<clipPath" in svg_code else 0.0
                
                # Add prompt length as a feature
                if hasattr(scene, 'prompt'):
                    features[15] = len(scene.prompt) / 100  # Normalized prompt length
            
            except Exception as e:
                logger.warning(f"Error extracting features: {e}")
                
            return features


@dataclass
class HallOfFameEntry:
    """An entry in the Hall of Fame for successful SVGs."""
    svg_code: str
    score: float
    params: Dict[str, Any]
    prompt: str
    timestamp: float = field(default_factory=time.time)


class HallOfFame:
    """Maintains a collection of the most successful SVGs for knowledge transfer."""
    
    def __init__(self, max_size: int = 20, storage_path: str = ".cache/hall_of_fame"):
        """
        Initialize the Hall of Fame.
        
        Args:
            max_size: Maximum number of entries to keep
            storage_path: Path to store entries
        """
        self.max_size = max_size
        self.storage_path = storage_path
        self.entries: List[HallOfFameEntry] = []
        self._lock = threading.RLock()
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Try to load existing entries
        self._load_entries()
    
    def add_entry(self, svg_code: str, score: float, params: Dict[str, Any], prompt: str) -> bool:
        """
        Add a new entry to the Hall of Fame.
        
        Args:
            svg_code: SVG code
            score: CLIP similarity score
            params: Generation parameters
            prompt: Text prompt
            
        Returns:
            True if entry was added, False if it wasn't good enough
        """
        with self._lock:
            # Create new entry
            entry = HallOfFameEntry(
                svg_code=svg_code,
                score=score,
                params=params.copy(),
                prompt=prompt,
                timestamp=time.time()
            )
            
            # Check if it's good enough to be added
            if len(self.entries) < self.max_size:
                self.entries.append(entry)
                self.entries.sort(key=lambda x: x.score, reverse=True)
                self._save_entries()
                return True
            elif score > self.entries[-1].score:
                # Replace the worst entry
                self.entries[-1] = entry
                self.entries.sort(key=lambda x: x.score, reverse=True)
                self._save_entries()
                return True
                
            return False
    
    def get_best_entry(self, prompt: str = None) -> Optional[HallOfFameEntry]:
        """
        Get the best entry, optionally filtered by prompt similarity.
        
        Args:
            prompt: Optional prompt to filter by
            
        Returns:
            Best entry or None if no entries
        """
        with self._lock:
            if not self.entries:
                return None
                
            if prompt is None:
                return self.entries[0]
                
            # Find entry with most similar prompt
            # This is a simple implementation - in production we'd use
            # sentence embeddings or other similarity measures
            best_entry = None
            best_similarity = -1
            
            for entry in self.entries:
                # Simple word overlap similarity
                prompt_words = set(prompt.lower().split())
                entry_words = set(entry.prompt.lower().split())
                similarity = len(prompt_words.intersection(entry_words)) / len(prompt_words.union(entry_words))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry
                    
            return best_entry
    
    def get_random_entry(self) -> Optional[HallOfFameEntry]:
        """
        Get a random entry from the Hall of Fame.
        
        Returns:
            Random entry or None if no entries
        """
        with self._lock:
            if not self.entries:
                return None
                
            return random.choice(self.entries)
    
    def get_entries(self, top_k: int = None) -> List[HallOfFameEntry]:
        """
        Get entries from the Hall of Fame.
        
        Args:
            top_k: Number of top entries to get, or None for all
            
        Returns:
            List of entries
        """
        with self._lock:
            if top_k is None or top_k >= len(self.entries):
                return self.entries.copy()
                
            return self.entries[:top_k]
    
    def _save_entries(self) -> None:
        """Save entries to disk."""
        try:
            with self._lock:
                # Save metadata
                metadata = []
                for i, entry in enumerate(self.entries):
                    metadata.append({
                        "score": entry.score,
                        "params": entry.params,
                        "prompt": entry.prompt,
                        "timestamp": entry.timestamp,
                        "filename": f"svg_{i}.svg"
                    })
                    
                metadata_path = os.path.join(self.storage_path, "metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                    
                # Save SVGs
                for i, entry in enumerate(self.entries):
                    svg_path = os.path.join(self.storage_path, f"svg_{i}.svg")
                    with open(svg_path, "w") as f:
                        f.write(entry.svg_code)
        except Exception as e:
            logger.error(f"Error saving Hall of Fame entries: {e}")
    
    def _load_entries(self) -> None:
        """Load entries from disk."""
        try:
            metadata_path = os.path.join(self.storage_path, "metadata.json")
            if not os.path.exists(metadata_path):
                return
                
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                
            entries = []
            for entry_data in metadata:
                svg_path = os.path.join(self.storage_path, entry_data.get("filename", ""))
                if not os.path.exists(svg_path):
                    continue
                    
                with open(svg_path, "r") as f:
                    svg_code = f.read()
                    
                entry = HallOfFameEntry(
                    svg_code=svg_code,
                    score=entry_data.get("score", 0.0),
                    params=entry_data.get("params", {}),
                    prompt=entry_data.get("prompt", ""),
                    timestamp=entry_data.get("timestamp", 0.0)
                )
                entries.append(entry)
                
            self.entries = sorted(entries, key=lambda x: x.score, reverse=True)
            logger.info(f"Loaded {len(self.entries)} entries to Hall of Fame")
        except Exception as e:
            logger.error(f"Error loading Hall of Fame entries: {e}")


class PPOOptimizer:
    """
    Full implementation of Proximal Policy Optimization for SVG generation.
    Includes policy and value networks, advantage estimation, and policy updates.
    """
    
    def __init__(self,
                 llm_manager: Optional[LLMManager] = None,
                 clip_evaluator: Optional[CLIPEvaluator] = None,
                 state_dim: int = 64,
                 action_dim: int = 16,
                 hidden_dim: int = 128,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 update_epochs: int = 4,
                 batch_size: int = 64,
                 device: str = "auto",
                 cache_dir: str = ".cache/ppo_models"):
        """
        Initialize the PPO optimizer.
        
        Args:
            llm_manager: LLM manager for SVG generation
            clip_evaluator: CLIP evaluator for reward calculation
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            lr: Learning rate
            gamma: Discount factor
            epsilon: PPO clip parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm
            update_epochs: Number of epochs for PPO updates
            batch_size: Batch size for updates
            device: Device to run on
            cache_dir: Directory for model caching
        """
        self.llm_manager = llm_manager or LLMManager()
        self.clip_evaluator = clip_evaluator or CLIPEvaluator()
        
        # Check for PyTorch availability
        torch_modules = _load_torch()
        if not torch_modules['available']:
            raise ImportError("PyTorch is required for PPOOptimizer")
        
        # PPO parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        
        # Determine optimal batch size based on available memory and device
        if batch_size is None:
            # Estimate item memory (state + action + reward + next_state)
            item_size_estimate = state_dim * 8 * 2 + 8 + 8  # ~bytes per item
            model_size_estimate = state_dim * action_dim * hidden_dim * 4 * 2  # Rough size for both networks
            
            self.batch_size = memory_manager.calculate_optimal_batch_size(
                item_size_estimate=item_size_estimate,
                model_size_estimate=model_size_estimate,
                target_device=device if device != "auto" else hardware_manager.get_optimal_device()
            )
            logger.info(f"Calculated optimal batch size: {self.batch_size}")
        else:
            self.batch_size = batch_size
            
        # Determine device
        self.device = self._get_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Create cache directory
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(feature_dim=state_dim)
        
        # Initialize networks (lazy)
        self.policy_net = None
        self.value_net = None
        self.policy_optimizer = None
        self.value_optimizer = None
        
        # Batch processor for parallel experience collection
        self._experience_processor = BatchProcessor(
            process_func=self._process_experience_batch,
            optimal_batch_size=min(8, self.batch_size),
            max_batch_size=min(16, self.batch_size),
            min_batch_size=1,
            adaptive_batching=True,
            memory_manager=memory_manager
        )
        self._experience_processor.start()
        
        # Experience buffer
        self.experiences = []
        self._experiences_lock = threading.Lock()
        
        # Initialize action mapping
        self.action_mapping = self._create_action_mapping()
        
        # Define parameter ranges
        self.parameter_ranges = {
            "temperature": (0.1, 1.0),
            "detail_level": (0.1, 1.0),
            "style_weight": (0.1, 1.0),
            "use_gradients": (0, 1),  # Boolean
            "use_patterns": (0, 1),   # Boolean
            "use_effects": (0, 1),     # Boolean
            "focus_keyword": (0, 15)  # Index of keyword to focus on
        }
        
        # Hall of Fame for successful SVGs
        self.hall_of_fame = HallOfFame(
            max_size=20,
            storage_path=os.path.join(cache_dir, "hall_of_fame")
        )
        
        logger.info(f"PPO Optimizer initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def _initialize_networks(self):
        """Initialize neural networks lazily to reduce memory usage on startup."""
        if self.policy_net is not None and self.value_net is not None:
            return
            
        with Profiler("ppo_network_init"):
            torch_modules = _load_torch()
            torch = torch_modules['torch']
            nn = torch_modules['nn']
            optim = torch_modules['optim']
            
            try:
                # Initialize networks
                self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).network
                self.value_net = ValueNetwork(self.state_dim, self.hidden_dim).network
                
                # Move to appropriate device
                self.policy_net = self.policy_net.to(self.device)
                self.value_net = self.value_net.to(self.device)
                
                # Initialize optimizers
                self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
                self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
                
                logger.info(f"Networks initialized on device: {self.device}")
            except Exception as e:
                logger.error(f"Failed to initialize networks: {e}")
                raise
    
    def _create_action_mapping(self) -> Dict[int, Tuple[str, float]]:
        """
        Create mapping from action indices to parameter changes.
        
        Returns:
            Dictionary mapping action indices to (parameter, delta) tuples
        """
        action_mapping = {}
        action_idx = 0
        
        # Continuous parameters with different delta values
        for param in ["temperature", "detail_level", "style_weight"]:
            for delta in [-0.2, -0.1, 0.1, 0.2]:
                action_mapping[action_idx] = (param, delta)
                action_idx += 1
        
        # Boolean parameters
        for param in ["use_gradients", "use_patterns", "use_effects"]:
            # Toggle actions
            action_mapping[action_idx] = (param, "toggle")
            action_idx += 1
        
        # Focus on specific semantic element
        action_mapping[action_idx] = ("focus_keyword", 1)  # Will be handled separately
        
        return action_mapping

    @memory_manager.memory_efficient_function
    def optimize(self, 
                scene: Scene, 
                base_svg: str, 
                max_iterations: int = 10,
                population_size: int = 8,
                callback: Optional[Callable[[int, float, str], None]] = None) -> Tuple[str, float]:
        """
        Optimize SVG generation using PPO.
        
        Args:
            scene: Scene to optimize
            base_svg: Initial SVG code
            max_iterations: Maximum number of iterations
            population_size: Population size per iteration
            callback: Optional callback function called with (iteration, score, svg)
            
        Returns:
            Tuple of (best_svg_code, best_score)
        """
        logger.info(f"Starting PPO optimization for scene {scene.id}")
        
        # Initialize networks if not already done
        self._initialize_networks()
        
        # Ensure CLIP evaluator is loaded
        if not self.clip_evaluator.model_loaded:
            self.clip_evaluator.load_model()
        
        # Start with default parameters
        current_params = self._get_default_parameters()
        best_params = current_params.copy()
        
        # Evaluate base SVG
        base_score = self.clip_evaluator.compute_similarity(base_svg, scene.prompt)
        best_score = base_score
        best_svg = base_svg
        
        logger.info(f"Initial score: {base_score:.4f}")
        
        # Extract keywords from prompt
        keywords = self._extract_keywords(scene.prompt)
        
        # Clear experience buffer
        with self._experiences_lock:
            self.experiences = []
        
        torch_modules = _load_torch()
        torch = torch_modules['torch']
        
        # Run optimization iterations
        for iteration in range(max_iterations):
            with Profiler(f"ppo_iteration_{iteration}"):
                iteration_start = time.time()
                logger.info(f"Starting iteration {iteration+1}/{max_iterations}")
                
                # Generate population of SVGs using policy
                population = self._generate_population(scene, current_params, keywords, population_size)
                
                # Memory checkpoint after generation
                memory_manager.operation_checkpoint()
                
                # Collect experiences from population
                experiences = self._collect_experiences(scene, population, current_params, keywords)
                
                # Add to experience buffer
                with self._experiences_lock:
                    self.experiences.extend(experiences)
                
                # Find best SVG in this population
                best_in_population = max(population, key=lambda p: p["score"]) if population else None
                
                # Update best overall if we have a better one
                if best_in_population and best_in_population["score"] > best_score:
                    best_score = best_in_population["score"]
                    best_svg = best_in_population["svg_code"]
                    best_params = best_in_population["params"].copy()
                    
                    # Update Hall of Fame
                    self.hall_of_fame.add_entry(
                        svg_code=best_svg,
                        score=best_score,
                        params=best_params,
                        prompt=scene.prompt
                    )
                    
                    # Call callback if provided
                    if callback:
                        callback(iteration + 1, best_score, best_svg)
                    
                    logger.info(f"New best score: {best_score:.4f} (improvement: +{best_score - base_score:.4f})")
                
                # Update policy if we have enough experiences
                with self._experiences_lock:
                    if len(self.experiences) >= self.batch_size:
                        self._update_policy()
                    
                # Sample new parameters from policy
                state = self._get_state(scene, best_svg, best_params, keywords)
                current_params = self._sample_parameters(state, keywords)
                
                iteration_time = time.time() - iteration_start
                logger.info(f"Completed iteration {iteration+1} in {iteration_time:.2f}s. "
                           f"Current best score: {best_score:.4f}")
                
                # Memory checkpoint
                memory_manager.operation_checkpoint()
        
        # Free up resources
        self._experience_processor.stop(wait_complete=False)
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return best_svg, best_score
    
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
            "focus_keyword": 0
        }
    
    @memoize
    def _extract_keywords(self, prompt: str) -> List[str]:
        """
        Extract keywords from prompt for potential focus.
        
        Args:
            prompt: Prompt text
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - just split by spaces and keep non-stopwords
        stopwords = {"a", "an", "the", "of", "in", "on", "with", "and", "or", "by", "to", "at", "for"}
        words = [word.lower() for word in prompt.split() if word.lower() not in stopwords]
        
        # Deduplicate
        unique_words = []
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
        
        # Cap at 16 keywords (our focus_keyword parameter range)
        return unique_words[:16]
    
    def _generate_population(self, 
                           scene: Scene, 
                           base_params: Dict[str, Any],
                           keywords: List[str],
                           population_size: int) -> List[Dict[str, Any]]:
        """
        Generate a population of SVGs using the current policy.
        
        Args:
            scene: Scene to generate for
            base_params: Base parameters to modify
            keywords: Keywords from prompt
            population_size: Size of population to generate
            
        Returns:
            List of dictionaries with SVG code, parameters, and scores
        """
        with Profiler("generate_population"):
            # Get current state
            state = self._get_state(scene, None, base_params, keywords)
            
            # Use batch processor for parallel generation
            generation_tasks = []
            
            for i in range(population_size):
                # Sample parameters from policy
                params = self._sample_parameters(state, keywords)
                generation_tasks.append((scene, params, keywords, f"candidate_{i}"))
            
            # Process tasks in parallel
            futures = []
            for task in generation_tasks:
                self._experience_processor.add_item(
                    f"gen_task_{id(task)}",
                    task,
                    priority=2  # Higher priority for generation
                )
                futures.append(f"gen_task_{id(task)}")
            
            # Collect results
            population = []
            
            for future_id in futures:
                result = self._experience_processor.get_result(future_id, timeout=30.0)
                if result:
                    population.append(result)
            
            # Fill in any missing slots with sequential generation
            if len(population) < population_size:
                remaining = population_size - len(population)
                logger.info(f"Generating {remaining} additional candidates sequentially")
                
                for i in range(remaining):
                    params = self._sample_parameters(state, keywords)
                    result = self._generate_and_evaluate_svg(scene, params, keywords, f"seq_candidate_{i}")
                    if result:
                        population.append(result)
            
            logger.info(f"Generated population of {len(population)} candidates")
            return population
    
    def _generate_and_evaluate_svg(self, scene: Scene, params: Dict[str, Any], 
                                 keywords: List[str], item_id: str) -> Dict[str, Any]:
        """Generate and evaluate a single SVG (for parallel processing)."""
        try:
            # Generate SVG
            svg_code = self._generate_svg(scene, params, keywords)
            
            # Skip if generation failed
            if not svg_code:
                return None
                
            # Evaluate SVG
            score = self.clip_evaluator.compute_similarity(svg_code, scene.prompt)
            
            # Return candidate
            return {
                "svg_code": svg_code,
                "params": params,
                "score": score,
                "id": item_id
            }
        except Exception as e:
            logger.error(f"Error in SVG generation and evaluation: {e}")
            return None
    
    def _sample_parameters(self, state: np.ndarray, keywords: List[str]) -> Dict[str, Any]:
        """
        Sample parameters using the current policy.
        
        Args:
            state: Current state
            keywords: Available keywords for focus
            
        Returns:
            Dictionary of sampled parameters
        """
        # Start with default parameters
        params = self._get_default_parameters()
        
        # Try to use a Hall of Fame entry as inspiration occasionally
        if random.random() < 0.2:  # 20% chance
            hof_entry = self.hall_of_fame.get_random_entry()
            if hof_entry:
                # Copy some parameters from successful entry
                for key in ["detail_level", "style_weight", "use_gradients", "use_patterns", "use_effects"]:
                    if key in hof_entry.params:
                        params[key] = hof_entry.params[key]
                        
                logger.debug("Using Hall of Fame entry as parameter inspiration")
        
        # Lazy load torch modules
        torch_modules = _load_torch()
        torch = torch_modules['torch']
        Categorical = torch_modules['Categorical']
        
        # Convert state to tensor
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Sample action from policy
            with torch.no_grad():
                action_probs = self.policy_net(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample().item()
            
            # Apply action to modify parameters
            if action in self.action_mapping:
                param_name, delta = self.action_mapping[action]
                
                if param_name in ["temperature", "detail_level", "style_weight"]:
                    # Apply delta to continuous parameter
                    params[param_name] = min(1.0, max(0.1, params[param_name] + delta))
                elif param_name in ["use_gradients", "use_patterns", "use_effects"]:
                    # Toggle boolean parameter
                    params[param_name] = not params[param_name]
                elif param_name == "focus_keyword" and keywords:
                    # Set focus keyword index
                    params[param_name] = min(len(keywords) - 1, max(0, action % len(keywords)))
        except Exception as e:
            logger.error(f"Error sampling parameters: {e}")
            # Return default parameters on error
        
        return params
    
    def _generate_svg(self, 
                    scene: Scene, 
                    params: Dict[str, Any], 
                    keywords: List[str]) -> str:
        """
        Generate SVG code using the provided parameters.
        
        Args:
            scene: Scene to generate for
            params: Generation parameters
            keywords: Keywords from prompt
            
        Returns:
            Generated SVG code
        """
        with Profiler("svg_generation"):
            # Create generation prompt
            prompt = self._create_generation_prompt(scene, params, keywords)
            
            # Get inspiration from Hall of Fame occasionally
            if random.random() < 0.3:  # 30% chance
                hof_entry = self.hall_of_fame.get_best_entry(scene.prompt)
                if hof_entry:
                    prompt += f"\n\nHere's an example of a high-quality SVG for inspiration (you can use similar techniques but create a different design):\n```xml\n{hof_entry.svg_code}\n```"
            
            # Generate SVG using LLM
            try:
                svg_code = self.llm_manager.generate(
                    role="svg_generator",
                    prompt=prompt,
                    max_tokens=10000,
                    temperature=params["temperature"],
                    stop_sequences=["```"]
                )
                
                # Extract SVG code
                return self._extract_svg_code(svg_code)
            except Exception as e:
                logger.error(f"Error generating SVG: {e}")
                return None
    
    def _create_generation_prompt(self, 
                                scene: Scene, 
                                params: Dict[str, Any], 
                                keywords: List[str]) -> str:
        """
        Create a generation prompt based on parameters.
        
        Args:
            scene: Scene to generate for
            params: Generation parameters
            keywords: Keywords from prompt
            
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
        
        # Add keyword focus if applicable
        if keywords and "focus_keyword" in params and params["focus_keyword"] < len(keywords):
            focus_keyword = keywords[params["focus_keyword"]]
            style_guidance.append(f"Place special emphasis on '{focus_keyword}' in the illustration.")
        
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

Technical requirements:
1. All colors should have good contrast
2. Use vector shapes appropriate for the content
3. Ensure proper layering of elements
4. Optimize SVG code for performance and file size
5. Make all paths and shapes fully enclosed and valid

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
        # Use the utility function from LLM manager
        svg_code = extract_svg_from_text(llm_response)
        
        if svg_code:
            return svg_code
            
        # If extraction failed, return a minimal valid SVG as fallback
        logger.warning("Failed to extract SVG code from LLM response")
        
        return f"""
<svg width="800" height="600" viewBox="0 0 800 600"
    xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#F5F5F5" />
    <text x="400" y="300" text-anchor="middle" font-size="24" fill="#333">
        Fallback SVG
    </text>
</svg>
"""
    
    def _get_state(self, 
                 scene: Scene, 
                 svg_code: Optional[str], 
                 params: Dict[str, Any],
                 keywords: List[str]) -> np.ndarray:
        """
        Get state representation for RL.
        
        Args:
            scene: Scene object
            svg_code: SVG code or None
            params: Current parameters
            keywords: Keywords from prompt
            
        Returns:
            State vector
        """
        # If no SVG code provided, use empty state
        if svg_code is None:
            return np.zeros(self.state_dim)
        
        # Extract features from scene and SVG
        features = self.feature_extractor.extract_features(scene, svg_code)
        
        # Add parameter values to state
        param_indices = {
            "temperature": 16,
            "detail_level": 17,
            "style_weight": 18,
            "use_gradients": 19,
            "use_patterns": 20,
            "use_effects": 21,
            "focus_keyword": 22
        }
        
        for param, idx in param_indices.items():
            if idx < len(features) and param in params:
                if param in ["use_gradients", "use_patterns", "use_effects"]:
                    features[idx] = 1.0 if params[param] else 0.0
                else:
                    features[idx] = float(params[param])
        
        return features
    
    def _collect_experiences(self,
                           scene: Scene,
                           population: List[Dict[str, Any]],
                           current_params: Dict[str, Any],
                           keywords: List[str]) -> List[Experience]:
        """
        Collect experiences from population.
        
        Args:
            scene: Scene object
            population: Population of candidates
            current_params: Current parameters
            keywords: Keywords from prompt
            
        Returns:
            List of experiences
        """
        with Profiler("collect_experiences"):
            if not population:
                return []
                
            # Sort population by score
            sorted_population = sorted(population, key=lambda p: p["score"], reverse=True)
            
            # Process in batches using the batch processor
            batch_items = [(scene, candidate, current_params, keywords, i)
                         for i, candidate in enumerate(sorted_population)]
            
            # Add tasks to processor
            for i, item in enumerate(batch_items):
                self._experience_processor.add_item(
                    f"exp_task_{id(item)}",
                    item,
                    priority=1  # Standard priority
                )
            
            # Collect experiences
            experiences = []
            
            # Wait for all results with timeout
            timeout = max(30.0, len(batch_items) * 2.0)  # 2 seconds per item, min 30 seconds
            start_time = time.time()
            
            # Process sequentially for small batches, otherwise use parallel
            if len(batch_items) <= 2:
                # Direct processing for small batches
                for item in batch_items:
                    experience = self._process_experience(*item)
                    if experience:
                        experiences.append(experience)
            else:
                # Parallel processing for larger batches
                for i, item in enumerate(batch_items):
                    task_id = f"exp_task_{id(item)}"
                    
                    # Check time remaining
                    if time.time() - start_time > timeout:
                        logger.warning("Experience collection timeout reached")
                        break
                        
                    # Get result with short timeout
                    result = self._experience_processor.get_result(task_id, timeout=5.0)
                    
                    if result:
                        experiences.append(result)
                    else:
                        # Fall back to direct processing
                        experience = self._process_experience(*item)
                        if experience:
                            experiences.append(experience)
            
            logger.info(f"Collected {len(experiences)} experiences")
            return experiences
    
    @memory_manager.memory_efficient_function
    def _process_experience(self, scene: Scene, candidate: Dict[str, Any], 
                          current_params: Dict[str, Any], keywords: List[str], index: int) -> Experience:
        """Process a single experience (for parallel processing)."""
        try:
            # Lazy load torch modules
            torch_modules = _load_torch()
            torch = torch_modules['torch']
            
            # Calculate reward as CLIP score
            reward = candidate["score"]
            
            # Get state
            state = self._get_state(scene, candidate["svg_code"], candidate["params"], keywords)
            
            # Determine action that led to these parameters
            action = self._params_to_action(current_params, candidate["params"])
            
            # Get state tensor for policy and value prediction
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action probabilities and value from networks
            with torch.no_grad():
                action_probs = self.policy_net(state_tensor)
                value = self.value_net(state_tensor).item()
            
            # Calculate log probability
            log_prob = torch.log(action_probs[0][action] + 1e-10).item()
            
            # Create experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=state,  # In our case, next state is same as state
                done=True,         # Each experience is a complete episode
                log_prob=log_prob,
                value=value
            )
            
            return experience
        except Exception as e:
            logger.error(f"Error creating experience: {e}")
            return None
    
    def _process_experience_batch(self, batch: List[Tuple[Scene, Dict[str, Any], Dict[str, Any], List[str], int]]) -> List[Experience]:
        """Process a batch of experiences for the batch processor."""
        results = []
        for scene, candidate, current_params, keywords, index in batch:
            experience = self._process_experience(scene, candidate, current_params, keywords, index)
            if experience:
                results.append(experience)
        return results
    
    def _params_to_action(self, 
                         current_params: Dict[str, Any], 
                         new_params: Dict[str, Any]) -> int:
        """
        Determine action that led from current params to new params.
        
        Args:
            current_params: Current parameters
            new_params: New parameters
            
        Returns:
            Action index
        """
        # Default action (no change)
        default_action = 0
        
        # Check each parameter for changes
        for param in ["temperature", "detail_level", "style_weight"]:
            if param in current_params and param in new_params:
                delta = new_params[param] - current_params[param]
                
                # Find closest delta in our action mapping
                for action, (p, d) in self.action_mapping.items():
                    if p == param:
                        if isinstance(d, (int, float)) and abs(d - delta) < 0.05:
                            return action
        
        # Check boolean parameters
        for param in ["use_gradients", "use_patterns", "use_effects"]:
            if param in current_params and param in new_params:
                if current_params[param] != new_params[param]:
                    # Find toggle action
                    for action, (p, d) in self.action_mapping.items():
                        if p == param and d == "toggle":
                            return action
        
        # Check focus keyword
        if "focus_keyword" in current_params and "focus_keyword" in new_params:
            if current_params["focus_keyword"] != new_params["focus_keyword"]:
                # This is a change in focus keyword
                for action, (p, d) in self.action_mapping.items():
                    if p == "focus_keyword":
                        return action
        
        return default_action
    
    @memory_manager.memory_efficient_function
    def _update_policy(self) -> None:
        """Update policy and value networks using PPO."""
        # Skip update if not enough experiences
        with self._experiences_lock:
            if len(self.experiences) < self.batch_size:
                return
        
        with Profiler("update_policy"):
            logger.info(f"Updating policy with {len(self.experiences)} experiences")
            
            # Lazy load torch modules
            torch_modules = _load_torch()
            torch = torch_modules['torch']
            F = torch_modules['F']
            Categorical = torch_modules['Categorical']
            
            try:
                # Sample batch
                with self._experiences_lock:
                    batch_indices = random.sample(range(len(self.experiences)), self.batch_size)
                    batch = [self.experiences[i] for i in batch_indices]
                
                # Prepare batch data
                states = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
                actions = torch.LongTensor([exp.action for exp in batch]).to(self.device)
                rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
                log_probs_old = torch.FloatTensor([exp.log_prob for exp in batch]).to(self.device)
                values_old = torch.FloatTensor([exp.value for exp in batch]).to(self.device)
                
                # Normalize rewards for stability
                if len(rewards) > 1:
                    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                
                # Compute advantages
                advantages = rewards - values_old
                
                # Optimize for multiple epochs
                for epoch in range(self.update_epochs):
                    # Get action probabilities and values
                    action_probs = self.policy_net(states)
                    values = self.value_net(states).squeeze()
                    
                    # Calculate log probabilities
                    dist = Categorical(action_probs)
                    log_probs = dist.log_prob(actions)
                    
                    # Calculate entropy
                    entropy = dist.entropy().mean()
                    
                    # Calculate ratio for PPO
                    ratio = torch.exp(log_probs - log_probs_old)
                    
                    # Calculate surrogate losses
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
                    
                    # Calculate policy loss
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Calculate value loss
                    value_loss = F.mse_loss(values, rewards)
                    
                    # Total loss
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    
                    # Update policy network
                    self.policy_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                    self.policy_optimizer.step()
                    
                    # Update value network
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                    self.value_optimizer.step()
                
                # Clear experience buffer
                with self._experiences_lock:
                    self.experiences = []
                
                # Force garbage collection
                gc.collect()
                
                logger.info("Policy updated successfully")
                
            except Exception as e:
                logger.error(f"Error updating policy: {e}")
                # Clear experiences on error to avoid repeating problematic data
                with self._experiences_lock:
                    self.experiences = []
    
    def save_models(self, path: Optional[str] = None) -> None:
        """
        Save policy and value networks.
        
        Args:
            path: Path to save models (defaults to cache_dir)
        """
        if not self.policy_net or not self.value_net:
            logger.warning("No models to save - networks not initialized")
            return
            
        save_path = path or os.path.join(self.cache_dir, "models")
        os.makedirs(save_path, exist_ok=True)
        
        # Lazy load torch
        torch_modules = _load_torch()
        torch = torch_modules['torch']
        
        try:
            # Save policy network
            policy_path = os.path.join(save_path, "policy_net.pt")
            torch.save(self.policy_net.state_dict(), policy_path)
            
            # Save value network
            value_path = os.path.join(save_path, "value_net.pt")
            torch.save(self.value_net.state_dict(), value_path)
            
            # Save action mapping
            with open(os.path.join(save_path, "action_mapping.json"), "w") as f:
                # Convert tuples to lists for JSON serialization
                serialized_mapping = {
                    str(k): [v[0], float(v[1]) if isinstance(v[1], (int, float)) else v[1]] 
                    for k, v in self.action_mapping.items()
                }
                json.dump(serialized_mapping, f, indent=2)
            
            logger.info(f"Models saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, path: Optional[str] = None) -> bool:
        """
        Load policy and value networks.
        
        Args:
            path: Path to load models from (defaults to cache_dir)
            
        Returns:
            Whether loading was successful
        """
        # Initialize networks
        self._initialize_networks()
        
        # Lazy load torch
        torch_modules = _load_torch()
        torch = torch_modules['torch']
        
        load_path = path or os.path.join(self.cache_dir, "models")
        
        try:
            # Check for model files
            policy_path = os.path.join(load_path, "policy_net.pt")
            value_path = os.path.join(load_path, "value_net.pt")
            
            if not os.path.exists(policy_path) or not os.path.exists(value_path):
                logger.warning(f"Model files not found in {load_path}")
                return False
            
            # Load policy network
            self.policy_net.load_state_dict(torch.load(policy_path, map_location=self.device))
            
            # Load value network
            self.value_net.load_state_dict(torch.load(value_path, map_location=self.device))
            
            # Try to load action mapping
            mapping_path = os.path.join(load_path, "action_mapping.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, "r") as f:
                    serialized_mapping = json.load(f)
                    # Convert back to the expected format
                    self.action_mapping = {
                        int(k): (v[0], v[1]) for k, v in serialized_mapping.items()
                    }
            
            logger.info(f"Models loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _get_device(self, device: str) -> str:
        """
        Determine the optimal device.
        
        Args:
            device: Requested device
            
        Returns:
            Optimal device
        """
        if device != "auto":
            return device
            
        # Use hardware manager to get optimal device
        return hardware_manager.get_optimal_device()
            
    def cleanup(self) -> None:
        """Free resources when done."""
        logger.info("Cleaning up PPOOptimizer resources")
        
        # Stop batch processor
        if hasattr(self, '_experience_processor'):
            self._experience_processor.stop(wait_complete=True)
        
        # Clear experiences
        with self._experiences_lock:
            self.experiences = []
            
        # Free GPU memory if using CUDA
        if self.device == 'cuda' and hasattr(self, 'policy_net') and self.policy_net is not None:
            torch_modules = _load_torch()
            torch = torch_modules['torch']
            
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                
        # Force garbage collection
        gc.collect()


# Create singleton instance for easy import
_default_ppo_optimizer = None

def get_default_ppo_optimizer() -> PPOOptimizer:
    """Get default PPO optimizer instance."""
    global _default_ppo_optimizer
    if _default_ppo_optimizer is None:
        _default_ppo_optimizer = PPOOptimizer()
    return _default_ppo_optimizer


# Utility functions for direct use
def optimize_svg(scene: Scene, base_svg: str, max_iterations: int = 10, population_size: int = 8, callback: Optional[Callable] = None) -> Tuple[str, float]:
    """
    Optimize SVG generation using PPO.
    
    Args:
        scene: Scene to optimize
        base_svg: Initial SVG code
        max_iterations: Maximum number of iterations
        population_size: Population size per iteration
        callback: Optional callback function for progress updates
        
    Returns:
        Tuple of (best_svg_code, best_score)
    """
    return get_default_ppo_optimizer().optimize(scene, base_svg, max_iterations, population_size, callback)