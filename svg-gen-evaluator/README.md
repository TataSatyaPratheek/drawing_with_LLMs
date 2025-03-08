# SVG Generation Evaluator

A toolkit for evaluating models that generate Scalable Vector Graphics (SVG) from text descriptions, with a focus on Kaggle's Drawing with LLMs competition.

## Features

- Full-featured evaluation framework for SVG generation models
- CLIP-based similarity scoring between text descriptions and rendered SVGs
- SVG validation against competition constraints
- Visualization tools for analyzing model performance
- Simple example model implementation
- Command-line interface for easy evaluation
- Extensible architecture for implementing custom models

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/svg-gen-evaluator.git
cd svg-gen-evaluator

# Install in development mode
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

## Quick Start

### Evaluating the Built-in Model

```bash
# Evaluate the simple model on test data
svg-evaluate --model simple --test-data data/test_descriptions.csv
```

### Using the Library in Your Code

```python
from svg_gen_evaluator.core.evaluator import SVGEvaluator
from svg_gen_evaluator.models.simple import SimpleModel

# Create an evaluator with a model
evaluator = SVGEvaluator(
    model_class=SimpleModel,
    output_dir="evaluation_results"
)

# Run evaluation on test data
results_df, avg_similarity = evaluator.evaluate(
    test_csv_path="data/test_descriptions.csv",
    save_images=True
)

print(f"Average CLIP similarity: {avg_similarity:.4f}")
```

### Creating Your Own Model

```python
from svg_gen_evaluator.models.base import BaseSVGModel

class MyAdvancedModel(BaseSVGModel):
    def __init__(self, param1=0.5, param2="default"):
        self.param1 = param1
        self.param2 = param2
        # Initialize your model here
        
    def predict(self, description: str) -> str:
        # Your SVG generation logic here
        # Must return a valid SVG string
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
            <!-- Your generated SVG content -->
        </svg>"""

# Evaluate your model
evaluator = SVGEvaluator(
    model_class=MyAdvancedModel,
    model_kwargs={"param1": 0.8, "param2": "custom"},
    output_dir="my_model_results"
)
```

## Command Line Interface

The package provides a command-line interface for evaluation:

```
usage: svg-evaluate [-h] --test-data TEST_DATA [--model MODEL] [--output-dir OUTPUT_DIR] 
                    [--config CONFIG] [--save-images] [--verbose]

Evaluate SVG generation models.

options:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Model to evaluate (built-in name or module path)
  --test-data TEST_DATA, -t TEST_DATA
                        Path to test data CSV with 'id' and 'description' columns
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory to save evaluation results
  --config CONFIG, -c CONFIG
                        Path to model configuration JSON file
  --save-images         Save rendered SVG images
  --verbose, -v         Enable verbose logging
```

## Project Structure

```
svg-gen-evaluator/
├── LICENSE
├── README.md
├── pyproject.toml          # Modern Python project configuration
├── .gitignore
├── src/
│   └── svg_gen_evaluator/  # Main package
│       ├── __init__.py
│       ├── core/           # Core functionality modules
│       │   ├── __init__.py
│       │   ├── evaluator.py   # Base evaluation system
│       │   ├── clip.py        # CLIP similarity calculation
│       │   ├── renderer.py    # SVG rendering utilities 
│       │   └── validator.py   # SVG validation utilities
│       ├── models/         # Model implementations
│       │   ├── __init__.py
│       │   ├── base.py     # Base model interface
│       │   └── simple.py   # Simple example model
│       ├── utils/          # Utility functions and helpers
│       │   ├── __init__.py
│       │   ├── visualization.py  # Results visualization
│       │   └── io.py       # File loading and saving
│       ├── config/         # Configuration management
│       │   ├── __init__.py
│       │   └── default.py  # Default configuration
│       └── cli.py          # Command line interface
├── tests/                  # Tests for the package
│   ├── __init__.py
│   ├── test_evaluator.py
│   ├── test_renderer.py
│   └── test_validator.py
├── examples/               # Example usage scripts
│   ├── basic_evaluation.py
│   └── advanced_evaluation.py
└── data/                   # Example data for testing
    ├── test_descriptions.csv
    └── example_svgs/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.