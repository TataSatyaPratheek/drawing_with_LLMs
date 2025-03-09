from setuptools import setup, find_packages

# Core dependencies
core_requirements = [
    "spacy>=3.5.0",
    "nltk>=3.8.1",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "typing-extensions>=4.4.0",
    "cairosvg>=2.5.2",
    "python-Levenshtein>=0.21.0"
]

# Optional LLM dependencies
llm_requirements = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "accelerate>=0.21.0",
    "sentencepiece>=0.1.99"
]

# Optional CLIP evaluation dependencies
clip_requirements = [
    "pillow>=9.3.0"
]

# Optional quantization dependencies
quant_requirements = [
    "bitsandbytes>=0.40.0"
]

setup(
    name="svg_prompt_analyzer",
    version="0.2.0",
    description="A tool to analyze text prompts and generate SVG images with LLM enhancement",
    author="Claude",
    packages=find_packages(),
    install_requires=core_requirements,
    extras_require={
        "llm": llm_requirements,
        "clip": clip_requirements,
        "quant": quant_requirements,
        "full": llm_requirements + clip_requirements + quant_requirements
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "svg-analyzer=svg_prompt_analyzer.main:main",
            "llm-svg-analyzer=svg_prompt_analyzer.llm_integration.llm_enhanced_app:run_cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)