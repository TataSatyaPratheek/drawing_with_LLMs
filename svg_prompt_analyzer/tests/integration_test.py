#!/usr/bin/env python3
"""
Integration Test for LLM-Enhanced SVG Generator
===========================================
This script runs a simple integration test to validate the LLM-enhanced SVG generation system.
"""

import os
import sys
import argparse
import logging
import tempfile
import json
from pathlib import Path

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from svg_prompt_analyzer.llm_integration.llm_manager import LLMManager
from svg_prompt_analyzer.llm_integration.llm_prompt_analyzer import LLMPromptAnalyzer
from svg_prompt_analyzer.llm_integration.llm_svg_generator import LLMSVGGenerator
from svg_prompt_analyzer.llm_integration.clip_evaluator import CLIPEvaluator
from svg_prompt_analyzer.utils.logger import setup_logger

# Configure logging
setup_logger("DEBUG")
logger = logging.getLogger(__name__)

def run_test(args):
    """Run the integration test."""
    test_prompts = [
        {"id": "test_1", "description": "a red circle on a blue background"},
        {"id": "test_2", "description": "a green triangle inside a yellow square"}
    ]
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = temp_dir if not args.output_dir else args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Using output directory: {output_dir}")
        
        # Initialize components
        logger.info("Initializing LLM Manager...")
        llm_manager = LLMManager()
        
        logger.info("Initializing LLM Prompt Analyzer...")
        analyzer = LLMPromptAnalyzer(llm_manager=llm_manager)
        
        logger.info("Initializing LLM SVG Generator...")
        generator = LLMSVGGenerator(llm_manager=llm_manager, output_dir=output_dir)
        
        # Test prompt analysis and SVG generation
        results = []
        
        for prompt in test_prompts:
            prompt_id = prompt["id"]
            prompt_text = prompt["description"]
            
            logger.info(f"Testing prompt: {prompt_text}")
            
            # 1. Analyze prompt
            logger.info("Analyzing prompt...")
            scene = analyzer.analyze_prompt(prompt_id, prompt_text)
            
            # 2. Generate SVG
            logger.info("Generating SVG...")
            svg_path = generator.save_svg(scene)
            
            # 3. Verify output
            if os.path.exists(svg_path):
                with open(svg_path, 'r') as f:
                    svg_content = f.read()
                    svg_size = len(svg_content)
                    is_valid = '<svg' in svg_content and '</svg>' in svg_content
                    
                logger.info(f"Generated SVG: {svg_path} (size: {svg_size} bytes, valid: {is_valid})")
                
                if args.clip_eval:
                    # 4. Evaluate with CLIP
                    logger.info("Evaluating with CLIP...")
                    clip_evaluator = CLIPEvaluator()
                    score = clip_evaluator.compute_similarity(svg_content, prompt_text)
                    logger.info(f"CLIP similarity score: {score:.4f}")
                    
                    results.append({
                        "prompt_id": prompt_id,
                        "prompt": prompt_text,
                        "svg_path": svg_path,
                        "svg_size": svg_size,
                        "is_valid": is_valid,
                        "clip_score": score
                    })
                else:
                    results.append({
                        "prompt_id": prompt_id,
                        "prompt": prompt_text,
                        "svg_path": svg_path,
                        "svg_size": svg_size,
                        "is_valid": is_valid
                    })
            else:
                logger.error(f"SVG generation failed for prompt: {prompt_text}")
                results.append({
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "error": "SVG generation failed"
                })
        
        # Save results
        results_path = os.path.join(output_dir, "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Test results saved to: {results_path}")
        
        # Print summary
        print("\nIntegration Test Summary:")
        print(f"- Prompts tested: {len(test_prompts)}")
        print(f"- Successful generations: {sum(1 for r in results if 'error' not in r)}")
        print(f"- Failed generations: {sum(1 for r in results if 'error' in r)}")
        
        if args.clip_eval:
            avg_score = sum(r.get("clip_score", 0) for r in results if "clip_score" in r) / len(results)
            print(f"- Average CLIP score: {avg_score:.4f}")
            
        print(f"\nOutput directory: {output_dir}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Run integration test for LLM-Enhanced SVG Generator")
    parser.add_argument("--output-dir", "-o", help="Directory for output files (default: temporary directory)")
    parser.add_argument("--clip-eval", "-c", action="store_true", help="Evaluate with CLIP similarity")
    args = parser.parse_args()
    
    try:
        run_test(args)
        logger.info("Integration test completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())