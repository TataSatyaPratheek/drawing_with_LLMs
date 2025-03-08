"""
Command-line interface for the SVG Generation Evaluator.
"""
import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type

from svg_gen_evaluator.core.evaluator import SVGEvaluator
from svg_gen_evaluator.models.base import BaseSVGModel
from svg_gen_evaluator.models.simple import SimpleModel
from svg_gen_evaluator.utils.io import load_config, save_config

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_model_class(model_spec: str) -> Type[BaseSVGModel]:
    """
    Load a model class from a module path or use a built-in model.
    
    Args:
        model_spec: Model specification, either a built-in name or module path
        
    Returns:
        SVG model class
    """
    # Check for built-in models
    if model_spec.lower() == "simple":
        return SimpleModel
    
    # Try to load from module path (e.g., "mypackage.models.MyModel")
    try:
        module_path, class_name = model_spec.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        # Verify it's a proper model class
        if not issubclass(model_class, BaseSVGModel):
            raise ValueError(
                f"Class {class_name} from {module_path} does not implement BaseSVGModel"
            )
        
        return model_class
    except (ImportError, AttributeError, ValueError) as e:
        logger.error(f"Failed to load model class {model_spec}: {e}")
        raise


def load_model_kwargs(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load model configuration from a JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary of keyword arguments for model initialization
    """
    if not config_path:
        return {}
    
    try:
        return load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate SVG generation models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="simple",
        help="Model to evaluate (built-in name or module path)",
    )
    
    parser.add_argument(
        "--test-data", "-t",
        type=str,
        required=True,
        help="Path to test data CSV with 'id' and 'description' columns",
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to model configuration JSON file",
    )
    
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save rendered SVG images",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the CLI.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    args = parse_args()
    setup_logging(args.verbose)
    
    try:
        # Load model and configuration
        model_class = load_model_class(args.model)
        model_kwargs = load_model_kwargs(args.config)
        
        # Create evaluator
        evaluator = SVGEvaluator(
            model_class=model_class,
            model_kwargs=model_kwargs,
            output_dir=args.output_dir,
        )
        
        # Run evaluation
        results_df, avg_similarity = evaluator.evaluate(
            test_csv_path=args.test_data,
            save_images=args.save_images,
        )
        
        # Save configuration used for this run
        config = {
            "model": args.model,
            "model_kwargs": model_kwargs,
            "test_data": args.test_data,
            "average_similarity": float(avg_similarity),
            "valid_svgs": int(results_df["valid"].sum()),
            "total_svgs": len(results_df),
        }
        save_config(config, Path(args.output_dir) / "evaluation_config.json")
        
        logger.info(f"Evaluation complete. Results saved to {args.output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())