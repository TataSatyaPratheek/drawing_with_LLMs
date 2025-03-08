"""
Default configuration settings for SVG generation evaluation.
"""

DEFAULT_CONFIG = {
    # SVG constraints
    "max_svg_size": 10000,  # Maximum SVG size in bytes
    
    # CLIP model settings
    "clip_model_name": "google/siglip-so400m-patch14-384",  # Primary CLIP model
    "fallback_model_name": "google/siglip-base-patch16-224",  # Fallback model if primary fails
    
    # Rendering settings
    "output_image_size": (512, 512),  # Default size for rendered images (width, height)
    
    # Evaluation settings
    "generation_timeout": 300,  # Maximum time in seconds for generating a single SVG
    "save_images": True,  # Whether to save rendered images by default
    
    # Visualization settings
    "create_visualizations": True,  # Whether to create visualization plots
}