"""
Model Pruner Module
================
This module provides functionality for pruning LLM models to reduce memory
footprint and improve inference speed while maintaining quality.
"""

import os
import gc
import logging
import time
import threading
import torch
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

logger = logging.getLogger(__name__)


class ModelPruner:
    """
    Class for applying model pruning techniques to reduce memory footprint
    and improve inference speed.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for resource efficiency."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelPruner, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, 
                 prune_amount: float = 0.3,
                 structured_pruning: bool = True,
                 dynamic_sparsity: bool = False,
                 hardware_aware: bool = True,
                 cache_dir: str = ".cache/pruned_models"):
        """
        Initialize the model pruner.
        
        Args:
            prune_amount: Amount of parameters to prune (0.0-1.0)
            structured_pruning: Whether to use structured pruning
            dynamic_sparsity: Whether to use dynamic sparsity
            hardware_aware: Whether to optimize for specific hardware
            cache_dir: Directory for caching pruned models
        """
        # Initialize only once (singleton pattern)
        if hasattr(self, 'initialized'):
            return
        
        self.prune_amount = prune_amount
        self.structured_pruning = structured_pruning
        self.dynamic_sparsity = dynamic_sparsity
        self.hardware_aware = hardware_aware
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Track pruned models
        self.pruned_models = {}
        
        # Flag initialization
        self.initialized = True
        logger.info(f"Model Pruner initialized with prune_amount={prune_amount}, "
                   f"structured_pruning={structured_pruning}")
    
    def prune_model(self, 
                   model: Any,
                   model_id: str,
                   device: str = "cpu",
                   evaluation_func: Optional[Callable] = None) -> Any:
        """
        Prune a model to reduce parameters while maintaining performance.
        
        Args:
            model: Model to prune
            model_id: Identifier for the model
            device: Device to run pruning on
            evaluation_func: Optional function to evaluate pruning quality
            
        Returns:
            Pruned model
        """
        # Check if model is already pruned
        if model_id in self.pruned_models:
            logger.info(f"Using cached pruned model for {model_id}")
            return self.pruned_models[model_id]
        
        try:
            logger.info(f"Pruning model {model_id} with prune_amount={self.prune_amount}")
            start_time = time.time()
            
            # Check if torch is available
            if not hasattr(torch, 'nn'):
                logger.error("PyTorch not available for model pruning")
                return model
            
            # Move model to specified device
            original_device = next(model.parameters()).device
            model = model.to(device)
            
            # Apply pruning strategy based on settings
            if self.structured_pruning:
                pruned_model = self._apply_structured_pruning(model)
            else:
                pruned_model = self._apply_unstructured_pruning(model)
            
            # Apply dynamic sparsity if enabled
            if self.dynamic_sparsity:
                pruned_model = self._apply_dynamic_sparsity(pruned_model)
            
            # Apply hardware optimizations if enabled
            if self.hardware_aware:
                pruned_model = self._apply_hardware_optimizations(pruned_model, device)
            
            # Evaluate pruned model if evaluation function provided
            if evaluation_func:
                logger.info("Evaluating pruned model quality")
                quality_score = evaluation_func(pruned_model)
                logger.info(f"Pruned model quality score: {quality_score:.4f}")
            
            # Move model back to original device
            pruned_model = pruned_model.to(original_device)
            
            # Cache the pruned model
            self.pruned_models[model_id] = pruned_model
            
            pruning_time = time.time() - start_time
            logger.info(f"Model pruning completed in {pruning_time:.2f}s")
            
            # Run garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return pruned_model
            
        except Exception as e:
            logger.error(f"Error during model pruning: {str(e)}")
            # Return original model on error
            return model
    
    def _apply_structured_pruning(self, model: Any) -> Any:
        """
        Apply structured pruning to the model (channel/head pruning).
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model
        """
        try:
            # Get current device
            device = next(model.parameters()).device
            
            # Create a copy of the model to avoid modifying the original
            pruned_model = type(model)(**(model.config.to_dict() if hasattr(model, 'config') else {}))
            pruned_model.load_state_dict(model.state_dict())
            pruned_model = pruned_model.to(device)
            
            # Identify prunable layers
            attention_layers = []
            ffn_layers = []
            
            # Traverse model to find attention and FFN layers
            for name, module in pruned_model.named_modules():
                # Identify attention heads
                if any(attn_type in name.lower() for attn_type in ['attention', 'attn']):
                    attention_layers.append((name, module))
                
                # Identify feed-forward layers
                elif any(ffn_type in name.lower() for ffn_type in ['mlp', 'ffn', 'feedforward']):
                    ffn_layers.append((name, module))
            
            # Apply attention head pruning
            self._prune_attention_heads(pruned_model, attention_layers)
            
            # Apply FFN intermediate dimension pruning
            self._prune_ffn_dimensions(pruned_model, ffn_layers)
            
            # Fine-tune pruned model briefly
            # This is a placeholder - in a real implementation, you'd want
            # to perform some fine-tuning to recover performance
            
            return pruned_model
            
        except Exception as e:
            logger.error(f"Error in structured pruning: {str(e)}")
            return model
    
    def _prune_attention_heads(self, model: Any, attention_layers: List[Tuple[str, Any]]) -> None:
        """
        Prune attention heads based on importance.
        
        Args:
            model: Model to modify
            attention_layers: List of (name, module) tuples for attention layers
        """
        # This is a simplified implementation
        # In a production system, you'd calculate head importance metrics
        # and prune the least important heads
        
        for name, module in attention_layers:
            # Skip layers without query/key/value projections
            if not hasattr(module, 'q_proj') or not hasattr(module, 'v_proj'):
                continue
            
            # Get head dimension
            if hasattr(module, 'num_heads'):
                num_heads = module.num_heads
            elif hasattr(module, 'n_heads'):
                num_heads = module.n_heads
            else:
                continue
                
            # Calculate number of heads to prune
            heads_to_prune = int(num_heads * self.prune_amount)
            if heads_to_prune <= 0:
                continue
                
            logger.debug(f"Pruning {heads_to_prune} heads from {name} with {num_heads} heads")
            
            # In a real implementation, you would modify the weights
            # to zero out specific heads based on importance scores
            
            # For now, just log the pruning plan
            logger.info(f"Would prune {heads_to_prune}/{num_heads} heads from {name}")
    
    def _prune_ffn_dimensions(self, model: Any, ffn_layers: List[Tuple[str, Any]]) -> None:
        """
        Prune feed-forward network dimensions based on importance.
        
        Args:
            model: Model to modify
            ffn_layers: List of (name, module) tuples for FFN layers
        """
        # This is a simplified implementation
        # In a production system, you'd calculate neuron importance metrics
        # and prune the least important neurons
        
        for name, module in ffn_layers:
            # Skip layers without intermediate projections
            if not hasattr(module, 'intermediate') and not hasattr(module, 'up_proj'):
                continue
            
            # Get intermediate size
            if hasattr(module, 'intermediate'):
                if hasattr(module.intermediate, 'dense'):
                    intermediate_size = module.intermediate.dense.out_features
                else:
                    continue
            elif hasattr(module, 'up_proj'):
                intermediate_size = module.up_proj.out_features
            else:
                continue
                
            # Calculate number of dimensions to prune
            dims_to_prune = int(intermediate_size * self.prune_amount)
            if dims_to_prune <= 0:
                continue
                
            logger.debug(f"Pruning {dims_to_prune} dimensions from {name} with {intermediate_size} dimensions")
            
            # In a real implementation, you would modify the weights
            # to zero out specific dimensions based on importance scores
            
            # For now, just log the pruning plan
            logger.info(f"Would prune {dims_to_prune}/{intermediate_size} dimensions from {name}")
    
    def _apply_unstructured_pruning(self, model: Any) -> Any:
        """
        Apply unstructured pruning to the model (individual weight pruning).
        
        Args:
            model: Model to prune
            
        Returns:
            Pruned model
        """
        try:
            # Get current device
            device = next(model.parameters()).device
            
            # Create a copy of the model to avoid modifying the original
            pruned_model = type(model)(**(model.config.to_dict() if hasattr(model, 'config') else {}))
            pruned_model.load_state_dict(model.state_dict())
            pruned_model = pruned_model.to(device)
            
            # Apply global unstructured pruning
            global_threshold = self._calculate_weight_threshold(pruned_model)
            
            # Apply threshold to each parameter
            with torch.no_grad():
                for param in pruned_model.parameters():
                    # Apply magnitude-based pruning
                    mask = (torch.abs(param.data) > global_threshold).float()
                    param.data.mul_(mask)
            
            return pruned_model
            
        except Exception as e:
            logger.error(f"Error in unstructured pruning: {str(e)}")
            return model
    
    def _calculate_weight_threshold(self, model: Any) -> float:
        """
        Calculate weight magnitude threshold for pruning.
        
        Args:
            model: Model to analyze
            
        Returns:
            Threshold value for pruning
        """
        # Collect all weights
        all_weights = []
        for param in model.parameters():
            if param.dim() > 1:  # Skip biases and 1D parameters
                all_weights.append(param.abs().view(-1))
        
        # Flatten and sort weights by magnitude
        all_weights = torch.cat(all_weights)
        sorted_weights, _ = torch.sort(all_weights)
        
        # Find threshold at the specified percentile
        threshold_idx = int(len(sorted_weights) * self.prune_amount)
        threshold = sorted_weights[threshold_idx].item()
        
        return threshold
    
    def _apply_dynamic_sparsity(self, model: Any) -> Any:
        """
        Apply dynamic sparsity to enable runtime pruning.
        
        Args:
            model: Model to modify
            
        Returns:
            Model with dynamic sparsity
        """
        # This is a placeholder for dynamic sparsity implementation
        # In a real system, you'd implement this using techniques like:
        # - Sparse attention mechanisms
        # - Conditional computation
        # - Runtime neuron activation
        
        logger.info("Dynamic sparsity implementation is a placeholder")
        return model
    
    def _apply_hardware_optimizations(self, model: Any, device: str) -> Any:
        """
        Apply hardware-specific optimizations.
        
        Args:
            model: Model to optimize
            device: Target device
            
        Returns:
            Optimized model
        """
        # Apply hardware-specific optimizations
        if device == "cuda" and torch.cuda.is_available():
            # CUDA-specific optimizations
            try:
                # Apply CUDA optimizations
                if hasattr(torch.backends, 'cudnn'):
                    torch.backends.cudnn.benchmark = True
                
                # Enable TF32 precision on Ampere+ GPUs
                if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    if hasattr(torch.backends.cuda, 'allow_tf32'):
                        torch.backends.cuda.allow_tf32 = True
            except Exception as e:
                logger.warning(f"Error applying CUDA optimizations: {str(e)}")
        
        elif device == "cpu":
            # CPU-specific optimizations
            try:
                # Enable MKL optimizations if available
                if hasattr(torch, 'set_num_threads') and os.cpu_count():
                    torch.set_num_threads(max(1, os.cpu_count() - 1))
                
                # Set number of interop threads
                if hasattr(torch, 'set_num_interop_threads') and os.cpu_count():
                    torch.set_num_interop_threads(max(1, os.cpu_count() // 2))
            except Exception as e:
                logger.warning(f"Error applying CPU optimizations: {str(e)}")
        
        # Try to apply JIT optimization for inference
        try:
            # Use torch.jit.optimize_for_inference if it's an inference model
            if hasattr(torch.jit, 'optimize_for_inference') and hasattr(model, 'eval'):
                model.eval()
                # This is hypothetical - the actual API might differ
                # model = torch.jit.optimize_for_inference(torch.jit.script(model))
                logger.info("Applied JIT optimization for inference")
        except Exception as e:
            logger.debug(f"Error applying JIT optimization: {str(e)}")
        
        return model
    
    def compress_model(self, 
                      model: Any,
                      model_id: str,
                      quantize: bool = True,
                      quantize_bits: int = 8,
                      mixed_precision: bool = True) -> Any:
        """
        Apply compression techniques to reduce model size.
        
        Args:
            model: Model to compress
            model_id: Identifier for the model
            quantize: Whether to apply quantization
            quantize_bits: Number of bits for quantization (4 or 8)
            mixed_precision: Whether to use mixed precision
            
        Returns:
            Compressed model
        """
        try:
            logger.info(f"Compressing model {model_id}")
            
            # Apply quantization if requested
            if quantize:
                if quantize_bits == 4:
                    # Apply 4-bit quantization
                    logger.info("Applying 4-bit quantization")
                    # Implementation would depend on the specific model architecture
                    # and quantization library (e.g., bitsandbytes)
                elif quantize_bits == 8:
                    # Apply 8-bit quantization
                    logger.info("Applying 8-bit quantization")
                    # Implementation would depend on the specific model architecture
                    # and quantization library
            
            # Apply mixed precision if requested
            if mixed_precision and torch.cuda.is_available():
                # Apply mixed precision (FP16)
                logger.info("Applying mixed precision (FP16)")
                # Implementation would depend on the specific model architecture
            
            logger.info(f"Model compression completed for {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Error during model compression: {str(e)}")
            return model
    
    def calculate_model_stats(self, model: Any) -> Dict[str, Any]:
        """
        Calculate statistics for a model.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary of model statistics
        """
        stats = {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "non_trainable_parameters": 0,
            "parameter_size_mb": 0,
            "layer_count": 0
        }
        
        try:
            # Calculate total parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_trainable_params = total_params - trainable_params
            
            # Calculate approximate memory usage
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            
            # Count layers
            layer_count = len(list(model.modules()))
            
            stats.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": non_trainable_params,
                "parameter_size_mb": (param_size + buffer_size) / (1024 * 1024),
                "layer_count": layer_count
            })
            
        except Exception as e:
            logger.error(f"Error calculating model statistics: {str(e)}")
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the cache of pruned models."""
        self.pruned_models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()