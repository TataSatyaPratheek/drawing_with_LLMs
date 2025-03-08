"""
Input/output utilities for loading and saving data.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the CSV data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}")
        raise


def save_results(
    results: Union[pd.DataFrame, List[Dict[str, Any]]], 
    output_path: Union[str, Path],
    format: str = 'csv'
) -> None:
    """
    Save evaluation results to a file.
    
    Args:
        results: DataFrame or list of dictionaries with results
        output_path: Path to save the results
        format: File format ('csv' or 'json')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Convert list to DataFrame if needed
    if isinstance(results, list):
        results = pd.DataFrame(results)
    
    try:
        if format.lower() == 'csv':
            results.to_csv(output_path, index=False)
            logger.info(f"Results saved to CSV: {output_path}")
        elif format.lower() == 'json':
            if output_path.suffix != '.json':
                output_path = output_path.with_suffix('.json')
            
            results.to_json(output_path, orient='records', indent=2)
            logger.info(f"Results saved to JSON: {output_path}")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


def load_svg(file_path: Union[str, Path]) -> str:
    """
    Load SVG code from a file.
    
    Args:
        file_path: Path to the SVG file
        
    Returns:
        SVG code as a string
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"SVG file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            svg_code = f.read()
        return svg_code
    except Exception as e:
        logger.error(f"Error loading SVG file {file_path}: {e}")
        raise


def save_svg(
    svg_code: str, 
    output_path: Union[str, Path],
    create_dirs: bool = True
) -> None:
    """
    Save SVG code to a file.
    
    Args:
        svg_code: SVG code as a string
        output_path: Path to save the SVG
        create_dirs: Whether to create parent directories if they don't exist
    """
    output_path = Path(output_path)
    
    if create_dirs:
        output_path.parent.mkdir(exist_ok=True, parents=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_code)
        logger.info(f"SVG saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving SVG to {output_path}: {e}")
        raise


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def save_config(
    config: Dict[str, Any], 
    output_path: Union[str, Path],
    create_dirs: bool = True
) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Dictionary containing configuration
        output_path: Path to save the configuration
        create_dirs: Whether to create parent directories if they don't exist
    """
    output_path = Path(output_path)
    
    if create_dirs:
        output_path.parent.mkdir(exist_ok=True, parents=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving configuration to {output_path}: {e}")
        raise