"""
PPO Optimizer Module - Enhanced
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
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from svg_prompt_analyzer.llm_integration.llm_manager import LLMManager
from svg_prompt_analyzer.llm_integration.clip_evaluator import CLIPEvaluator
from svg_prompt_analyzer.models.scene import Scene

logger = logging.getLogger(__name__)


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


class PolicyNetwork(nn.Module):
    """Neural network for the policy in PPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space
            hidden_dim: Dimension of hidden layers
        """
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Action probabilities
        """
        return self.network(state)


class ValueNetwork(nn.Module):
    """Neural network for the value function in PPO."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        """
        Initialize the value network.
        
        Args:
            state_dim: Dimensionality of state space
            hidden_dim: Dimension of hidden layers
        """
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
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
    
    def extract_features(self, scene: Scene, svg_code: str) -> np.ndarray:
        """
        Extract features from scene and SVG.
        
        Args:
            scene: Scene object
            svg_code: SVG code
            
        Returns:
            Feature vector
        """
        features = np.zeros(self.feature_dim)
        
        # Extract basic features
        features[0] = len(scene.objects)  # Number of objects
        features[1] = len(svg_code) / 10000  # Normalized SVG size
        
        # Extract color features
        color_counts = {}
        for obj in scene.objects:
            if obj.color:
                color_name = obj.color.name
                color_counts[color_name] = color_counts.get(color_name, 0) + 1
        
        features[2] = len(color_counts)  # Number of unique colors
        
        # Extract spatial features
        if scene.objects:
            avg_x = sum(obj.position[0] for obj in scene.objects) / len(scene.objects)
            avg_y = sum(obj.position[1] for obj in scene.objects) / len(scene.objects)
            features[3] = avg_x
            features[4] = avg_y
        
        # Extract complexity features
        features[5] = svg_code.count("<") / 100  # Number of tags
        features[6] = svg_code.count("path") / 10  # Number of paths
        
        # Extract semantic features - these would be calculated in a real implementation
        # E.g., by using embeddings from a language model
        
        return features


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
        self.batch_size = batch_size
        
        # Determine device
        self.device = self._get_device(device)
        
        # Create cache directory
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(feature_dim=state_dim)
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Experience buffer
        self.experiences = []
        
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
        
        logger.info(f"PPO Optimizer initialized with state_dim={state_dim}, action_dim={action_dim}")
    
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
        self.experiences = []
        
        # Run optimization iterations
        for iteration in range(max_iterations):
            iteration_start = time.time()
            logger.info(f"Starting iteration {iteration+1}/{max_iterations}")
            
            # Generate population of SVGs using policy
            population = self._generate_population(scene, current_params, keywords, population_size)
            
            # Collect experiences from population
            experiences = self._collect_experiences(scene, population, current_params, keywords)
            self.experiences.extend(experiences)
            
            # Find best SVG in this population
            best_in_population = max(population, key=lambda p: p["score"])
            
            # Update best overall
            if best_in_population["score"] > best_score:
                best_score = best_in_population["score"]
                best_svg = best_in_population["svg_code"]
                best_params = best_in_population["params"].copy()
                
                # Call callback if provided
                if callback:
                    callback(iteration + 1, best_score, best_svg)
                
                logger.info(f"New best score: {best_score:.4f} (improvement: +{best_score - base_score:.4f})")
            
            # Update policy if we have enough experiences
            if len(self.experiences) >= self.batch_size:
                self._update_policy()
                
            # Sample new parameters from policy
            state = self._get_state(scene, best_svg, best_params, keywords)
            current_params = self._sample_parameters(state)
            
            iteration_time = time.time() - iteration_start
            logger.info(f"Completed iteration {iteration+1} in {iteration_time:.2f}s. "
                       f"Current best score: {best_score:.4f}")
        
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
        population = []
        
        # Get current state
        state = self._get_state(scene, None, base_params, keywords)
        
        # Generate candidates using policy
        for i in range(population_size):
            # Sample parameters from policy
            params = self._sample_parameters(state)
            
            # Generate SVG
            svg_code = self._generate_svg(scene, params, keywords)
            
            # Evaluate SVG
            score = self.clip_evaluator.compute_similarity(svg_code, scene.prompt)
            
            # Add to population
            population.append({
                "svg_code": svg_code,
                "params": params,
                "score": score
            })
        
        return population
    
    def _sample_parameters(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Sample parameters using the current policy.
        
        Args:
            state: Current state
            
        Returns:
            Dictionary of sampled parameters
        """
        # Start with default parameters
        params = self._get_default_parameters()
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Sample action from policy
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample().item()
        
        # Apply action to modify parameters
        param_name, delta = self.action_mapping[action]
        
        if param_name in ["temperature", "detail_level", "style_weight"]:
            # Apply delta to continuous parameter
            params[param_name] = min(1.0, max(0.1, params[param_name] + delta))
        elif param_name in ["use_gradients", "use_patterns", "use_effects"]:
            # Toggle boolean parameter
            params[param_name] = not params[param_name]
        elif param_name == "focus_keyword":
            # Set focus keyword index
            if keywords:
                params[param_name] = min(len(keywords) - 1, max(0, action % len(keywords)))
        
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
        # Create generation prompt
        prompt = self._create_generation_prompt(scene, params, keywords)
        
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
        {scene.prompt}
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
            "temperature": 10,
            "detail_level": 11,
            "style_weight": 12,
            "use_gradients": 13,
            "use_patterns": 14,
            "use_effects": 15,
            "focus_keyword": 16
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
        experiences = []
        
        # Sort population by score
        sorted_population = sorted(population, key=lambda p: p["score"], reverse=True)
        
        # Convert to tensors for batch processing
        state_tensors = []
        for candidate in sorted_population:
            state = self._get_state(scene, candidate["svg_code"], candidate["params"], keywords)
            state_tensors.append(torch.FloatTensor(state))
        
        states_tensor = torch.stack(state_tensors).to(self.device)
        
        # Get action probabilities and values
        with torch.no_grad():
            action_probs = self.policy_net(states_tensor)
            values = self.value_net(states_tensor).squeeze()
        
        # Create experiences
        for i, candidate in enumerate(sorted_population):
            # Calculate reward as CLIP score
            reward = candidate["score"]
            
            # Get state
            state = self._get_state(scene, candidate["svg_code"], candidate["params"], keywords)
            
            # Determine action that led to these parameters
            action = self._params_to_action(current_params, candidate["params"])
            
            # Calculate log probability
            log_prob = torch.log(action_probs[i][action] + 1e-10).item()
            
            # Get value
            value = values[i].item()
            
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
            
            experiences.append(experience)
        
        return experiences
    
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
    
    def _update_policy(self) -> None:
        """Update policy and value networks using PPO."""
        # Skip update if not enough experiences
        if len(self.experiences) < self.batch_size:
            return
        
        logger.info(f"Updating policy with {len(self.experiences)} experiences")
        
        # Sample batch
        batch_indices = random.sample(range(len(self.experiences)), self.batch_size)
        batch = [self.experiences[i] for i in batch_indices]
        
        # Prepare batch data
        states = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        log_probs_old = torch.FloatTensor([exp.log_prob for exp in batch]).to(self.device)
        values_old = torch.FloatTensor([exp.value for exp in batch]).to(self.device)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Compute advantages
        advantages = rewards - values_old
        
        # Optimize for multiple epochs
        for _ in range(self.update_epochs):
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
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
        
        # Clear experience buffer
        self.experiences = []
        
        logger.info("Policy updated successfully")
    
    def save_models(self, path: str) -> None:
        """
        Save policy and value networks.
        
        Args:
            path: Path to save models
        """
        os.makedirs(path, exist_ok=True)
        
        # Save policy network
        policy_path = os.path.join(path, "policy_net.pt")
        torch.save(self.policy_net.state_dict(), policy_path)
        
        # Save value network
        value_path = os.path.join(path, "value_net.pt")
        torch.save(self.value_net.state_dict(), value_path)
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str) -> bool:
        """
        Load policy and value networks.
        
        Args:
            path: Path to load models from
            
        Returns:
            Whether loading was successful
        """
        try:
            # Load policy network
            policy_path = os.path.join(path, "policy_net.pt")
            self.policy_net.load_state_dict(torch.load(policy_path, map_location=self.device))
            
            # Load value network
            value_path = os.path.join(path, "value_net.pt")
            self.value_net.load_state_dict(torch.load(value_path, map_location=self.device))
            
            logger.info(f"Models loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
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
            
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, 'mps') and torch.mps.is_available():
            return "mps"
        else:
            return "cpu"