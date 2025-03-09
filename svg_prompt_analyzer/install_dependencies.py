#!/usr/bin/env python3
"""
Dependency Installation Script for LLM-Enhanced SVG Generator
=========================================================
This script installs all required dependencies for the LLM-enhanced SVG generation system.
"""

import os
import sys
import subprocess
import argparse
import platform

def main():
    parser = argparse.ArgumentParser(description="Install dependencies for LLM-Enhanced SVG Generator")
    parser.add_argument("--minimal", action="store_true", help="Install only core dependencies (no LLM/CLIP)")
    parser.add_argument("--cuda", action="store_true", help="Install CUDA-enabled versions")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only installations")
    args = parser.parse_args()
    
    print("Installing dependencies for LLM-Enhanced SVG Generator...")
    
    # Define dependency groups
    core_deps = [
        "spacy>=3.5.0",
        "nltk>=3.8.1",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "typing-extensions>=4.4.0",
        "cairosvg>=2.5.2",
        "python-Levenshtein>=0.21.0"
    ]
    
    llm_deps_cpu = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.21.0"
    ]
    
    llm_deps_cuda = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.21.0",
        "bitsandbytes>=0.40.0"
    ]
    
    clip_deps = [
        "pillow>=9.3.0"
    ]
    
    # Determine which deps to install
    to_install = core_deps
    
    if not args.minimal:
        if args.cuda and not args.cpu_only:
            to_install.extend(llm_deps_cuda)
        else:
            to_install.extend(llm_deps_cpu)
        to_install.extend(clip_deps)
    
    # Install spaCy English model
    print("\nInstalling dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + to_install)
    
    if not args.minimal:
        print("\nInstalling spaCy English model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
        print("\nInstalling NLTK data...")
        subprocess.check_call([sys.executable, "-c", 
                             "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"])
    
    # Install the package in development mode
    print("\nInstalling svg_prompt_analyzer package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    
    print("\nAll dependencies installed successfully!")
    
    if not args.minimal:
        print("\nTo use LLM features, you will need:")
        print("1. At least 16GB RAM for 7B parameter models")
        print("2. CUDA-compatible GPU with 8GB+ VRAM for hardware acceleration")
        print("3. Internet access for first-time model downloads")
        
    print("\nTo test the installation, try:")
    print("python -m svg_prompt_analyzer.main test.csv -o output")

if __name__ == "__main__":
    main()