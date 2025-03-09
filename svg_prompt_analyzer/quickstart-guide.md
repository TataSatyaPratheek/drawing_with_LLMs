# Quickstart Guide: LLM-Enhanced SVG Generation

This guide will help you quickly set up and run the LLM-enhanced SVG generator system.

## 1. Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- Recommended: CUDA-compatible GPU with 8GB+ VRAM

### Basic Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/svg-prompt-analyzer.git
cd svg-prompt-analyzer

# Option 1: Use the automated installation script
python install_dependencies.py

# Option 2: Manual installation with pip
pip install -e ".[full]"
python -m spacy download en_core_web_sm
```

## 2. Verify Installation

Run a quick test to verify everything is working:

```bash
# Run integration test
python tests/integration_test.py
```

You should see output indicating that the test completed successfully.

## 3. Basic Usage

### Process a CSV file of prompts

Your CSV file should have at least these columns:
- `id`: Unique identifier for each prompt
- `description`: Text description of the image to create

```bash
# Using the standard mode (no LLM)
svg-analyzer test.csv --output-dir output

# Using LLM-enhanced mode
svg-analyzer test.csv --output-dir output --use-llm

# Using LLM with optimization
svg-analyzer test.csv --output-dir output --use-llm --optimize
```

### Examine the results

```bash
# Open the output directory
cd output

# Preview an SVG file
# On macOS:
open 011af1.svg

# On Linux:
xdg-open 011af1.svg

# On Windows:
start 011af1.svg
```

## 4. Advanced Usage

### Customizing with configuration file

Create a file named `config.json`:

```json
{
  "llm": {
    "prompt_analyzer_model": "mistralai/Mistral-7B-Instruct-v0.2",
    "svg_generator_model": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "device": "auto",
    "use_8bit": true
  },
  "optimization": {
    "max_iterations": [1, 2, 3],
    "population_size": [3, 5, 8],
    "exploration_rate": 0.3
  }
}
```

Then run with this config:

```bash
svg-analyzer test.csv --output-dir output --use-llm --config config.json
```

### Parallel processing

For faster processing of multiple prompts:

```bash
svg-analyzer test.csv --output-dir output --use-llm --parallel --workers 4
```

### Optimize for best quality

For the highest quality results (but slower processing):

```bash
svg-analyzer test.csv --output-dir output --use-llm --optimize --optimization-level 3
```

## 5. Memory Optimization

If you encounter memory issues:

### Enable 8-bit quantization

Edit your config.json:

```json
{
  "llm": {
    "use_8bit": true,
    "use_4bit": false
  }
}
```

### Enable 4-bit quantization (even lower memory usage)

```json
{
  "llm": {
    "use_8bit": false,
    "use_4bit": true
  }
}
```

### Reduce batch size

```bash
svg-analyzer test.csv --output-dir output --use-llm --batch-size 2
```

## 6. Troubleshooting

### LLM models fail to load

Check:
- You have sufficient RAM (16GB+ recommended)
- If using GPU, verify CUDA is properly installed
- You have internet access for the first run to download models

### SVG generation errors

- Enable debug logging: `--log-level DEBUG`
- Check if the SVG code exceeds 10KB size limit
- Verify your CSV file has the correct format

### CLIP evaluation issues

- Ensure you have cairosvg installed: `pip install cairosvg`
- Ensure pillow is installed: `pip install pillow`

## 7. Next Steps

- Explore the "hall_of_fame" directory to see the best generated SVGs
- Review `output/summary.json` for statistics on your generation run
- Try different LLM models by editing the configuration file