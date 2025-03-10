"""
Model pruning and optimization utilities for production deployment.
Provides tools for reducing model size and improving inference speed.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# Import optimized core module components
from svg_prompt_analyzer.core import CONFIG, memoize, jit, Profiler, get_thread_pool

class ModelPruner:
    """
    Production-grade model pruning utility that optimizes models for inference
    while maintaining accuracy thresholds.
    """
    
    def __init__(
        self,
        sparsity_target: float = 0.5,
        accuracy_threshold: float = 0.02,
        quantize: bool = True,
        quantize_precision: str = "int8",
        cache_pruned_models: bool = True,
        cache_dir: Optional[str] = None,
        use_distributed: bool = False
    ):
        """
        Initialize the model pruner with production configuration.
        
        Args:
            sparsity_target: Target sparsity level (0.0-1.0)
            accuracy_threshold: Maximum acceptable accuracy drop
            quantize: Whether to quantize the model after pruning
            quantize_precision: Quantization precision (int8, fp16, etc.)
            cache_pruned_models: Whether to cache pruned models to disk
            cache_dir: Directory to cache pruned models
            use_distributed: Whether to use distributed processing
        """
        self.sparsity_target = sparsity_target
        self.accuracy_threshold = accuracy_threshold
        self.quantize = quantize
        self.quantize_precision = quantize_precision
        self.cache_pruned_models = cache_pruned_models
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "model_cache")
        self.use_distributed = use_distributed and CONFIG["distributed_mode"]
        
        # Create cache directory if needed
        if self.cache_pruned_models and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Optimize based on available hardware
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.num_gpus = torch.cuda.device_count()
            self.mixed_precision = True
        else:
            self.num_gpus = 0
            self.mixed_precision = False
            # Adjust settings for CPU-only environment
            if self.quantize_precision == "int8" and not hasattr(torch, 'quantization'):
                self.quantize_precision = "fp16"
                
        # Precompute some optimization settings
        self._calculate_optimal_batch_size()

    def _calculate_optimal_batch_size(self) -> None:
        """Calculate the optimal batch size based on hardware."""
        if self.device == "cuda":
            # Estimate based on available GPU memory
            try:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                # Use ~70% of available memory for safe operation
                self.batch_size = max(1, int((gpu_mem * 0.7) / (1024**3 * 2)))
            except Exception:
                self.batch_size = 16  # Fallback default
        else:
            # CPU-based heuristic
            self.batch_size = max(1, os.cpu_count() or 4)

    def _get_model_cache_path(self, model: torch.nn.Module, config_hash: str) -> str:
        """Generate a unique cache path for the pruned model."""
        model_name = type(model).__name__
        return os.path.join(
            self.cache_dir,
            f"{model_name}_{config_hash}_s{int(self.sparsity_target*100)}_"
            f"q{self.quantize_precision}.pt"
        )

    def _calculate_config_hash(self, model: torch.nn.Module) -> str:
        """Generate a hash of the model configuration for caching."""
        import hashlib
        import json
        
        # Get model parameters summary
        param_sizes = [p.numel() for p in model.parameters()]
        layer_names = [name for name, _ in model.named_modules()]
        
        # Create a deterministic representation
        config_dict = {
            "architecture": type(model).__name__,
            "param_count": sum(param_sizes),
            "layer_count": len(layer_names),
            "layer_sizes": param_sizes[:10],  # Use first 10 layers as fingerprint
        }
        
        # Generate hash
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:10]

    @memoize
    def analyze_model(
        self, 
        model: torch.nn.Module
    ) -> Dict[str, Any]:
        """
        Analyze model for pruning opportunities with memory optimization.
        
        Args:
            model: The PyTorch model to analyze
            
        Returns:
            Dictionary with analysis metrics and recommendations
        """
        with Profiler("model_analysis"):
            # Move model to appropriate device
            model.to(self.device)
            
            # Collect layer information
            layers = []
            total_params = 0
            
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and not isinstance(
                    module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)
                ):
                    # Skip analyzing batch norm layers as they shouldn't be pruned
                    weight = module.weight
                    params = weight.numel()
                    total_params += params
                    
                    # Calculate current sparsity
                    if weight.data is not None:
                        zeros = (weight.data.abs() < 1e-6).sum().item()
                        sparsity = zeros / params if params > 0 else 0
                    else:
                        sparsity = 0
                        
                    layers.append({
                        'name': name,
                        'type': module.__class__.__name__,
                        'params': params,
                        'sparsity': sparsity,
                        'prunable': True
                    })
            
            # Calculate potential memory savings
            potential_saving = total_params * self.sparsity_target * (
                4 if not self.quantize else (
                    2 if self.quantize_precision == 'fp16' else 
                    1 if self.quantize_precision == 'int8' else 4
                )
            )
            
            return {
                'total_parameters': total_params,
                'prunable_layers': len(layers),
                'current_sparsity': sum(l['sparsity'] * l['params'] for l in layers) / total_params,
                'potential_memory_saving_bytes': potential_saving,
                'layer_analysis': layers,
                'recommended_layers': [
                    l['name'] for l in layers 
                    if l['params'] > 1000 and l['sparsity'] < self.sparsity_target
                ]
            }

    def prune_model(
        self, 
        model: torch.nn.Module,
        calibration_fn: Optional[Callable] = None,
        pruning_method: str = 'magnitude',
        custom_pruning_fn: Optional[Callable] = None,
    ) -> torch.nn.Module:
        """
        Prune a model to reduce its size and improve inference speed.
        
        Args:
            model: The PyTorch model to prune
            calibration_fn: Function to run model calibration (for quantization)
            pruning_method: Pruning method to use ('magnitude', 'structured', 'custom')
            custom_pruning_fn: Custom pruning function if pruning_method='custom'
            
        Returns:
            Pruned and optimized model
        """
        # Check for cached model first
        config_hash = self._calculate_config_hash(model)
        cache_path = self._get_model_cache_path(model, config_hash)
        
        if self.cache_pruned_models and os.path.exists(cache_path):
            try:
                return torch.load(cache_path, map_location=self.device)
            except Exception as e:
                import logging
                logging.warning(f"Failed to load cached model: {e}")
        
        with Profiler("model_pruning"):
            # Get model analysis
            analysis = self.analyze_model(model)
            
            # Apply pruning
            if pruning_method == 'magnitude':
                self._apply_magnitude_pruning(model, analysis)
            elif pruning_method == 'structured':
                self._apply_structured_pruning(model, analysis)
            elif pruning_method == 'custom' and custom_pruning_fn is not None:
                custom_pruning_fn(model, analysis)
            else:
                raise ValueError(f"Unsupported pruning method: {pruning_method}")
            
            # Apply quantization if enabled
            if self.quantize:
                model = self._quantize_model(model, calibration_fn)
                
            # Cache the pruned model
            if self.cache_pruned_models:
                try:
                    torch.save(model, cache_path)
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to cache pruned model: {e}")
                    
            return model

    def _apply_magnitude_pruning(
        self, 
        model: torch.nn.Module, 
        analysis: Dict[str, Any]
    ) -> None:
        """Apply magnitude-based pruning to the model."""
        import torch.nn.utils.prune as prune
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and name in analysis['recommended_layers']:
                prune.l1_unstructured(
                    module, 
                    name='weight', 
                    amount=self.sparsity_target
                )
                # Make pruning permanent to save memory
                prune.remove(module, 'weight')

    def _apply_structured_pruning(
        self, 
        model: torch.nn.Module, 
        analysis: Dict[str, Any]
    ) -> None:
        """Apply structured pruning to the model."""
        import torch.nn.utils.prune as prune
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and name in analysis['recommended_layers']:
                # Different structured pruning based on layer type
                if isinstance(module, torch.nn.Conv2d):
                    # Channel pruning for Conv2d
                    prune.ln_structured(
                        module, 
                        name='weight', 
                        amount=self.sparsity_target, 
                        n=2, 
                        dim=0  # Prune output channels
                    )
                elif isinstance(module, torch.nn.Linear):
                    # Structured row pruning for Linear
                    prune.ln_structured(
                        module, 
                        name='weight', 
                        amount=self.sparsity_target, 
                        n=2, 
                        dim=0  # Prune output features
                    )
                # Make pruning permanent
                prune.remove(module, 'weight')

    def _quantize_model(
        self, 
        model: torch.nn.Module, 
        calibration_fn: Optional[Callable] = None
    ) -> torch.nn.Module:
        """Quantize the model to reduce memory footprint."""
        # Skip if PyTorch quantization not available
        if not hasattr(torch, 'quantization') and self.quantize_precision == 'int8':
            return model
            
        if self.quantize_precision == 'int8':
            # Int8 quantization
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm' if self.device == 'cpu' else 'qnnpack')
            model_prepared = torch.quantization.prepare(model)
            
            # Run calibration if provided
            if calibration_fn is not None:
                calibration_fn(model_prepared)
                
            # Convert to quantized model
            quantized_model = torch.quantization.convert(model_prepared)
            return quantized_model
            
        elif self.quantize_precision == 'fp16':
            # Half-precision (FP16)
            if self.device == 'cuda':
                return model.half()
            else:
                # FP16 not well supported on CPU, return original
                return model
                
        # Return original model if quantization not supported
        return model
        
    def optimize_for_inference(
        self, 
        model: torch.nn.Module,
        optimize_memory: bool = True,
        optimize_speed: bool = True,
        export_format: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Apply additional optimizations for inference.
        
        Args:
            model: The pruned PyTorch model
            optimize_memory: Whether to apply memory optimizations
            optimize_speed: Whether to apply speed optimizations
            export_format: Export format (None, 'torchscript', 'onnx')
            
        Returns:
            Optimized model for inference
        """
        with Profiler("inference_optimization"):
            # Basic inference prep
            model.eval()
            
            # Memory optimizations
            if optimize_memory:
                # Remove dropout layers
                for module in model.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = 0
                        
                # Freeze parameters
                for param in model.parameters():
                    param.requires_grad = False
            
            # Speed optimizations
            if optimize_speed and self.device == 'cuda':
                # Use CUDA graphs for repeated inference if available
                if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'make_graphed_callables'):
                    # Note: This is advanced and may need sample inputs specific to the model
                    # Implementation would depend on the specific model architecture
                    pass
                    
            # Export optimized model
            if export_format == 'torchscript':
                try:
                    return torch.jit.script(model)
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to export to TorchScript: {e}")
            elif export_format == 'onnx':
                try:
                    import io
                    import onnx
                    import onnxruntime
                    
                    # This is a placeholder - actual implementation would need sample inputs
                    # and proper export configuration
                    pass
                except ImportError:
                    import logging
                    logging.warning("ONNX export requested but onnx/onnxruntime not available")
            
            return model


# Utility functions for batched operations
def batch_process_tensors(
    tensors: List[torch.Tensor],
    process_fn: Callable,
    batch_size: Optional[int] = None
) -> List[torch.Tensor]:
    """
    Process a large list of tensors in batches to optimize memory usage.
    
    Args:
        tensors: List of input tensors
        process_fn: Function to apply to each batch
        batch_size: Batch size (auto-calculated if None)
        
    Returns:
        List of processed tensors
    """
    if not tensors:
        return []
        
    # Auto-calculate batch size if not provided
    if batch_size is None:
        # Heuristic: Larger tensors = smaller batches
        avg_size = sum(t.numel() for t in tensors) / len(tensors)
        batch_size = max(1, min(128, int(1e7 / avg_size)))
    
    # Process in batches
    results = []
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i+batch_size]
        # Process batch
        if len(batch) == 1:
            # Handle single item case
            results.append(process_fn(batch[0]))
        else:
            # Stack batch and process
            stacked = torch.stack(batch)
            processed = process_fn(stacked)
            # Unstack results
            results.extend(torch.unbind(processed))
            
    return results


# Memory-efficient weight transfer between models
def transfer_weights(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    layer_mapping: Optional[Dict[str, str]] = None
) -> None:
    """
    Transfer weights between models with memory optimization.
    
    Args:
        source_model: Source model to transfer weights from
        target_model: Target model to transfer weights to
        layer_mapping: Optional mapping between layer names
    """
    # Default to identity mapping
    if layer_mapping is None:
        # Create mapping based on matching parameter names
        layer_mapping = {}
        target_state_dict = dict(target_model.named_parameters())
        
        for name, param in source_model.named_parameters():
            if name in target_state_dict:
                layer_mapping[name] = name
    
    # Transfer weights
    with torch.no_grad():
        for source_name, target_name in layer_mapping.items():
            try:
                source_tensor = None
                target_tensor = None
                
                # Locate source parameter
                for name, param in source_model.named_parameters():
                    if name == source_name:
                        source_tensor = param
                        break
                
                # Locate target parameter
                for name, param in target_model.named_parameters():
                    if name == target_name:
                        target_tensor = param
                        break
                
                # Transfer weights if shapes match
                if source_tensor is not None and target_tensor is not None:
                    if source_tensor.shape == target_tensor.shape:
                        target_tensor.copy_(source_tensor)
                    else:
                        import logging
                        logging.warning(
                            f"Shape mismatch: {source_name} {source_tensor.shape} -> "
                            f"{target_name} {target_tensor.shape}"
                        )
            except Exception as e:
                import logging
                logging.warning(f"Error transferring weights for {source_name}: {e}")