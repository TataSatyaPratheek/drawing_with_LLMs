# LLM-Enhanced SVG Prompt Analyzer and Generator

A robust Python system that uses large language models (LLMs) and reinforcement learning to generate high-quality SVG images from text prompts.

## Overview

This system enhances the original SVG Prompt Analyzer with:

1. **LLM-powered prompt analysis** for deeper semantic understanding of text descriptions
2. **LLM-powered SVG generation** to create higher quality, more accurate SVG code
3. **CLIP-based evaluation** to measure similarity between prompts and generated images
4. **Reinforcement learning optimization** to iteratively improve SVG generation quality
5. **Fallback mechanisms** to guarantee reliable output even when LLM processing fails

The architecture maintains the original modular design while adding new components for enhanced capabilities.

## Features

- **Enhanced Semantic Understanding**: LLM-powered analysis extracts nuanced visual details from prompts
- **High-Quality SVG Generation**: Uses specialized code LLMs to create sophisticated SVG graphics
- **CLIP Similarity Optimization**: Directly optimizes for the competition evaluation metric  
- **Reinforcement Learning**: Iteratively improves generation through feedback loops
- **Flexible Configuration**: Easy customization via config files and command-line options
- **Parallel Processing**: Support for multi-threaded operation for batch processing
- **Memory Efficiency**: Optimizations for handling large batches with constrained resources
- **Robust Fallback System**: Graceful degradation when components encounter issues

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/svg-prompt-analyzer.git
cd svg-prompt-analyzer

# Install basic dependencies
pip install -e .
```

### Full Installation (with LLM and CLIP support)

```bash
# Install with all optional dependencies
pip install -e ".[full]"

# Download required language models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
```

### Automated Installation

For convenience, an installation script is provided:

```bash
# Full installation
python install_dependencies.py

# Minimal installation (without LLM/CLIP)
python install_dependencies.py --minimal

# CUDA-enabled installation
python install_dependencies.py --cuda
```

## Requirements

- **Basic Mode**: Python 3.8+ with minimal dependencies
- **LLM-Enhanced Mode**:
  - 16GB+ RAM for 7B parameter models
  - CUDA-compatible GPU with 8GB+ VRAM recommended
  - Internet access for initial model downloads

## Usage

### Command Line Interface

```bash
# Basic mode (original functionality)
svg-analyzer input.csv --output-dir output

# LLM-enhanced mode
svg-analyzer input.csv --output-dir output --use-llm

# With optimization
svg-analyzer input.csv --output-dir output --use-llm --optimize

# With custom configuration
svg-analyzer input.csv --output-dir output --use-llm --config my_config.json

# Advanced options
svg-analyzer input.csv --output-dir output --use-llm --optimize --optimization-level 3 --parallel --workers 4
```

### Python API

```python
from svg_prompt_analyzer.llm_integration.llm_enhanced_app import LLMEnhancedSVGApp

# Initialize the application
app = LLMEnhancedSVGApp(
    input_file="input.csv",
    output_dir="output",
    use_llm=True,
    use_optimization=True
)

# Process all prompts
results = app.run()
```

## Configuration

The system can be configured via a JSON configuration file:

```json
{
  "llm": {
    "prompt_analyzer_model": "mistralai/Mistral-7B-Instruct-v0.2",
    "svg_generator_model": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "device": "auto",
    "use_8bit": true,
    "use_4bit": false
  },
  "clip": {
    "model_name": "SigLIP-SoViT-400m",
    "device": "auto"
  },
  "optimization": {
    "max_iterations": [1, 2, 3],
    "population_size": [3, 5, 8],
    "exploration_rate": 0.3
  }
}
```

## Architecture

The enhanced system architecture includes these key components:

- **LLMManager**: Handles loading and inference with LLM models
- **LLMPromptAnalyzer**: Uses LLMs to extract detailed scene information from prompts
- **LLMSVGGenerator**: Uses code-specialized LLMs to generate high-quality SVG code
- **CLIPEvaluator**: Measures similarity between text prompts and generated SVGs
- **RLOptimizer**: Implements reinforcement learning to improve generation quality

## Optimization Levels

The system supports three optimization levels:

1. **Level 1**: Basic optimization (1 iteration, 3 candidates)
2. **Level 2**: Standard optimization (2 iterations, 5 candidates)
3. **Level 3**: Thorough optimization (3 iterations, 8 candidates)

Higher levels produce better results but require more processing time.

## Performance Considerations

- **Memory Usage**: LLM models require significant RAM (at least 16GB recommended)
- **GPU Acceleration**: CUDA-compatible GPU highly recommended for reasonable speed
- **Quantization**: 8-bit and 4-bit quantization options available to reduce memory usage
- **Batch Processing**: Adjust batch size based on available memory
- **Parallel Processing**: Useful for multi-prompt processing on multi-core systems

## License

[MIT License](LICENSE)