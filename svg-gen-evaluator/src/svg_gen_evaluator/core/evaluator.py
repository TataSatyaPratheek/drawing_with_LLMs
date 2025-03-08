"""
Core evaluation functionality for SVG generation models.
"""
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import pandas as pd
from tqdm import tqdm

from svg_gen_evaluator.core.clip import CLIPSimilarityCalculator
from svg_gen_evaluator.core.renderer import SVGRenderer
from svg_gen_evaluator.core.validator import SVGValidator
from svg_gen_evaluator.models.base import BaseSVGModel
from svg_gen_evaluator.utils.io import save_results
from svg_gen_evaluator.utils.visualization import create_visualizations

logger = logging.getLogger(__name__)


class SVGEvaluator:
    """
    Evaluates SVG generation models according to competition rules.
    
    This class handles the complete evaluation process:
    1. Loading the model
    2. Processing test descriptions
    3. Generating SVGs
    4. Validating those SVGs against the competition constraints
    5. Calculating CLIP similarity scores
    6. Generating reports and visualizations
    """
    
    def __init__(
        self,
        model_class: Type[BaseSVGModel],
        model_kwargs: Optional[Dict] = None,
        output_dir: Union[str, Path] = "evaluation_results",
        max_svg_size: int = 10000,
        generation_timeout: int = 300,  # 5 minutes per SVG in seconds
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_class: Your SVG generation model class (must implement BaseSVGModel)
            model_kwargs: Optional arguments to pass to your model's constructor
            output_dir: Directory to save evaluation results
            max_svg_size: Maximum allowed SVG size in bytes
            generation_timeout: Maximum time allowed for generating a single SVG in seconds
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize core components
        self.validator = SVGValidator(max_svg_size=max_svg_size)
        self.renderer = SVGRenderer()
        self.clip_calculator = CLIPSimilarityCalculator()
        
        # Configuration
        self.generation_timeout = generation_timeout
        self.model = None
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "evaluation.log"),
                logging.StreamHandler(),
            ],
        )
    
    def load_model(self) -> None:
        """Load the model to be evaluated."""
        logger.info("Loading SVG generation model...")
        self.model = self.model_class(**self.model_kwargs)
        logger.info("Model loaded successfully.")
    
    def load_test_data(self, test_csv_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load test descriptions from a CSV file.
        
        Args:
            test_csv_path: Path to the CSV file containing test data
            
        Returns:
            DataFrame containing test data
        """
        logger.info(f"Loading test data from {test_csv_path}...")
        path = Path(test_csv_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Test file not found: {path}")
        
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} test examples.")
        return df
    
    def evaluate(
        self, test_csv_path: Union[str, Path], save_images: bool = True
    ) -> Tuple[pd.DataFrame, float]:
        """
        Run evaluation on the test set.
        
        Args:
            test_csv_path: Path to CSV file with test descriptions
            save_images: Whether to save rendered images
            
        Returns:
            Tuple of (results DataFrame, average similarity score)
        """
        if self.model is None:
            self.load_model()
        
        test_df = self.load_test_data(test_csv_path)
        results = []
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating SVGs"):
            description = row["description"]
            id_val = row["id"]
            
            # Time the SVG generation
            start_time = time.time()
            try:
                # Generate SVG with timeout
                svg_code = self._generate_with_timeout(description)
                generation_time = time.time() - start_time
                
                # Validate SVG
                is_valid, validation_error = self.validator.validate(svg_code)
                
                if is_valid:
                    # Render SVG to image
                    image = self.renderer.render_svg(svg_code)
                    
                    if save_images:
                        # Save the rendered image
                        image_path = self.output_dir / f"{id_val}.png"
                        image.save(image_path)
                        
                        # Save the SVG code
                        svg_path = self.output_dir / f"{id_val}.svg"
                        with open(svg_path, "w", encoding="utf-8") as f:
                            f.write(svg_code)
                    
                    # Compute CLIP similarity
                    similarity = self.clip_calculator.compute_similarity(description, image)
                    
                    results.append({
                        "id": id_val,
                        "description": description,
                        "svg_size": len(svg_code.encode("utf-8")),
                        "valid": True,
                        "similarity": similarity,
                        "generation_time": generation_time
                    })
                else:
                    logger.warning(f"Invalid SVG for '{description}': {validation_error}")
                    results.append({
                        "id": id_val,
                        "description": description,
                        "svg_size": len(svg_code.encode("utf-8")) if svg_code else 0,
                        "valid": False,
                        "error": validation_error,
                        "similarity": 0.0,
                        "generation_time": generation_time
                    })
            except Exception as e:
                logger.error(f"Error processing '{description}': {e}")
                results.append({
                    "id": id_val,
                    "description": description,
                    "svg_size": 0,
                    "valid": False,
                    "error": str(e),
                    "similarity": 0.0,
                    "generation_time": time.time() - start_time
                })
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Save results and generate reports
        self._generate_report(results_df)
        
        # Calculate overall metrics
        valid_results = results_df[results_df["valid"]]
        avg_similarity = valid_results["similarity"].mean() if len(valid_results) > 0 else 0.0
        
        return results_df, avg_similarity
    
    def _generate_with_timeout(self, description: str) -> str:
        """
        Generate SVG with a timeout.
        
        Args:
            description: The text description to generate SVG from
            
        Returns:
            Generated SVG code
            
        Raises:
            TimeoutError: If generation takes too long
        """
        # In a real implementation, you'd need to implement actual timeout logic
        # This could be done with the 'timeout' library, multiprocessing, or other approaches
        
        # Simple placeholder implementation
        start_time = time.time()
        svg_code = self.model.predict(description)
        
        if time.time() - start_time > self.generation_timeout:
            raise TimeoutError(f"SVG generation exceeded the {self.generation_timeout}s limit")
            
        return svg_code
    
    def _generate_report(self, results_df: pd.DataFrame) -> None:
        """
        Generate evaluation report and visualizations.
        
        Args:
            results_df: DataFrame with evaluation results
        """
        # Calculate metrics
        valid_count = results_df["valid"].sum()
        total_count = len(results_df)
        valid_percentage = (valid_count / total_count) * 100 if total_count > 0 else 0
        
        valid_results = results_df[results_df["valid"]]
        avg_similarity = valid_results["similarity"].mean() if len(valid_results) > 0 else 0
        avg_generation_time = results_df["generation_time"].mean()
        
        # Print summary
        logger.info("\n=== Evaluation Results ===")
        logger.info(f"Total examples: {total_count}")
        logger.info(f"Valid SVGs: {valid_count} ({valid_percentage:.2f}%)")
        logger.info(f"Average CLIP similarity: {avg_similarity:.4f}")
        logger.info(f"Average generation time: {avg_generation_time:.2f} seconds")
        
        # Save detailed results
        save_results(results_df, self.output_dir / "evaluation_results.csv")
        
        # Create visualizations
        if valid_count > 0:
            create_visualizations(valid_results, self.output_dir)
            logger.info(f"Visualizations saved to {self.output_dir}")