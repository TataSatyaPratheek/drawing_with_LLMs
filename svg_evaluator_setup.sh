#!/bin/bash

# Exit script if any command fails
set -e

# Print commands as they are executed
set -x

# Create project directory and navigate into it
mkdir -p svg-gen-evaluator
cd svg-gen-evaluator

# Create basic project files
touch LICENSE
touch README.md
touch pyproject.toml
touch .gitignore
touch .pre-commit-config.yaml

# Create directory structure
mkdir -p src/svg_gen_evaluator/core
mkdir -p src/svg_gen_evaluator/models
mkdir -p src/svg_gen_evaluator/utils
mkdir -p src/svg_gen_evaluator/config
mkdir -p tests
mkdir -p examples
mkdir -p data/example_svgs

# Create __init__.py files
touch src/svg_gen_evaluator/__init__.py
touch src/svg_gen_evaluator/core/__init__.py
touch src/svg_gen_evaluator/models/__init__.py
touch src/svg_gen_evaluator/utils/__init__.py
touch src/svg_gen_evaluator/config/__init__.py
touch tests/__init__.py

# Create core module files
touch src/svg_gen_evaluator/core/evaluator.py
touch src/svg_gen_evaluator/core/clip.py
touch src/svg_gen_evaluator/core/renderer.py
touch src/svg_gen_evaluator/core/validator.py

# Create model files
touch src/svg_gen_evaluator/models/base.py
touch src/svg_gen_evaluator/models/simple.py

# Create utility files
touch src/svg_gen_evaluator/utils/visualization.py
touch src/svg_gen_evaluator/utils/io.py

# Create config files
touch src/svg_gen_evaluator/config/default.py

# Create CLI file
touch src/svg_gen_evaluator/cli.py

# Create test files
touch tests/test_evaluator.py
touch tests/test_renderer.py
touch tests/test_validator.py

# Create example files
touch examples/basic_evaluation.py
touch examples/advanced_evaluation.py

# Create sample data file
touch data/test_descriptions.csv

# Make example scripts executable
chmod +x examples/basic_evaluation.py
chmod +x examples/advanced_evaluation.py

# Make setup script executable
chmod +x svg_evaluator_setup.sh

echo "Project structure created successfully!"

# Optional: Initialize git repository
git init
echo "# SVG Generation Evaluator" > README.md
echo "__pycache__/" > .gitignore
echo "*.pyc" >> .gitignore
echo "*.pyo" >> .gitignore
echo "*.pyd" >> .gitignore
echo "*.so" >> .gitignore
echo "*.egg" >> .gitignore
echo "*.egg-info/" >> .gitignore
echo "dist/" >> .gitignore
echo "build/" >> .gitignore
echo ".pytest_cache/" >> .gitignore
echo ".coverage" >> .gitignore
echo "htmlcov/" >> .gitignore
echo ".env" >> .gitignore
echo ".venv" >> .gitignore
echo "env/" >> .gitignore
echo "venv/" >> .gitignore
echo "ENV/" >> .gitignore
echo ".DS_Store" >> .gitignore

# Create minimal pyproject.toml
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "svg-gen-evaluator"
version = "0.1.0"
description = "Evaluation toolkit for SVG generation models in Kaggle competitions"
readme = "README.md"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
license = {text = "MIT"}
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
requires-python = ">=3.8"
dependencies = [
    "numpy",
    "pandas",
    "torch>=1.10.0",
    "transformers>=4.20.0",
    "cairosvg",
    "Pillow",
    "matplotlib",
    "tqdm",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black",
    "isort",
    "flake8",
    "pre-commit",
]

[project.scripts]
svg-evaluate = "svg_gen_evaluator.cli:main"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.black]
line-length = 88
target-version = ["py38"]

[tool.isort]
profile = "black"
line_length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
EOF

# Create pre-commit configuration
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings]
EOF

# Create a sample test descriptions file
cat > data/test_descriptions.csv << 'EOF'
"id","description"
"01","a starlit night over snow-covered peaks"
"02","black and white checkered pants"
"03","crimson rectangles forming a chaotic grid"
"04","burgundy corduroy pants with patch pockets and silver buttons"
"05","orange corduroy overalls"
"06","a lighthouse overlooking the ocean"
"07","a green lagoon under a cloudy sky"
"08","a snowy plain"
"09","a maroon dodecahedron interwoven with teal threads"
"10","a purple silk scarf with tassel trim"
EOF

# Create an MIT license file
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# Print completion message
echo "Project structure created successfully! Next steps:"
echo "1. Navigate to the project directory: cd svg-gen-evaluator"
echo "2. Create a virtual environment: python -m venv venv"
echo "3. Activate the environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
echo "4. Install in development mode: pip install -e ."
echo "5. Start implementing the modules with your code"