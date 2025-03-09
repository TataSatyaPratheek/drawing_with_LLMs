from setuptools import setup, find_packages

setup(
    name="svg_prompt_analyzer",
    version="0.1.0",
    description="A tool to analyze text prompts and generate SVG images",
    author="Claude",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.5.0",
        "nltk>=3.8.1",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "typing-extensions>=4.4.0",
        "cairosvg>=2.5.2"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "svg-analyzer=svg_prompt_analyzer.main:run_cli",
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
