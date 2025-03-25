import os
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

CONFIG = {
    # Raw data paths (external location)
    "raw_images_dir": BASE_DIR / "biye/dataset/train",
    "raw_annotations": BASE_DIR / "biye/dataset/train.json",
    
    # Output directory (guided/dataset)
    "output_root": BASE_DIR / "biye/lguided/dataset",
    
    # Split parameters
    "val_ratio": 0.2,
    "random_seed": 42,
    
    # Proposal generationS
    "proposal_params": {
        "edge_model": BASE_DIR / "biye/model.yml",
        "max_proposals": 75,
        "alpha": 0.7,
        "beta": 0.7
    },
    
    # CLIP processing
    "clip_model": "ViT-B/32",
    "batch_size": 100,
    
    # Performance
    "num_workers": min(4, os.cpu_count()),  # Use up to 4 cores
    "image_cache_size": 1000
}