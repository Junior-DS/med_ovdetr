"""
Improved Proposal Generation Script
- Updated paths for your directory structure
- Handles splits from multiple locations
"""

import torch
import cv2
import numpy as np
import pickle
import json
import os
from pathlib import Path
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# After creating predictor
from detectron2.checkpoint import DetectionCheckpointer


from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid")

class ProposalGenerator:
    def __init__(self, custom_weights_path=None):
        self.cfg = get_cfg()
        
        # 1. Load base configuration
        self.cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        
        # 2. Force CPU-only mode
        self.cfg.MODEL.DEVICE = "cpu"
        
        # 3. REMOVED anchor generator modifications
        # Keep original anchor settings for compatibility
        
        # 4. Input resolution adjustments (keep these if needed)
        self.cfg.INPUT.MIN_SIZE_TEST = 512
        self.cfg.INPUT.MAX_SIZE_TEST = 512
        
        # 5. Load weights
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        
        # 6. Initialize predictor
        self.predictor = DefaultPredictor(self.cfg)
    
    def process_split(self, split_name, json_dir="splits"):
        """Process a dataset split to generate region proposals"""
        # Determine JSON path based on split type
        if "novel" in split_name:
            json_path = f"dataset/json_files/{split_name}.json"
        else:
            json_path = f"dataset/{json_dir}/{split_name}.json"
        
        output_path = f"dataset/proposals/{split_name}_proposals.pkl"
        
        # Create output directory if not exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path) as f:
            data = json.load(f)
        
        proposals = {}
        missing_files = []
        corrupted_files = []
        resolution_issues = []
        
        # Pre-filter valid images and paths
        valid_images = []
        for img in data['images']:
            img_path = Path("dataset/updated_train") / img['file_name']
            if img_path.exists():
                valid_images.append((img, img_path))
            else:
                missing_files.append(str(img_path))
        
        print(f"Processing {len(valid_images)}/{len(data['images'])} valid images")
        
        # Batch processing parameters
        batch_size = 4 if torch.cuda.is_available() else 1
        enable_optimizations = torch.cuda.is_available()
        
        for i in tqdm(range(0, len(valid_images), batch_size), 
                    desc=split_name, 
                    unit='batch'):
            batch = valid_images[i:i + batch_size]
            batch_proposals = {}
            
            for img, img_path in batch:
                try:
                    # Read with orientation handling
                    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    if image is None:
                        raise ValueError("Failed to read image")
                    
                    # Handle resolution scaling
                    original_h, original_w = image.shape[:2]
                    if enable_optimizations and (original_h > 1024 or original_w > 1024):
                        scale_factor = 1024 / max(original_h, original_w)
                        image = cv2.resize(image, (0,0), fx=scale_factor, fy=scale_factor)
                        resolution_issues.append(str(img_path))
                    
                    # Convert color space
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Generate proposals
                    with torch.no_grad():  # Reduce memory usage
                        outputs = self.predictor(image)
                    
                    # Store proposals with original coordinates
                    instances = outputs['instances']
                    batch_proposals[img['id']] = {
                        'boxes': instances.pred_boxes.tensor.cpu().numpy(),
                        'scores': instances.scores.cpu().numpy(),
                        'original_size': (original_w, original_h),
                        'processed_size': image.shape[:2][::-1]
                    }
                    
                except Exception as e:
                    corrupted_files.append(f"{img_path} - {str(e)}")
                    continue
            
            # Update proposals dictionary
            proposals.update(batch_proposals)
        
        # Save proposals
        with open(output_path, 'wb') as f:
            pickle.dump(proposals, f)
        
        # Print diagnostics
        print(f"\n{split_name} proposal generation complete")
        print(f"- Generated proposals for {len(proposals)} images")
        
        if missing_files:
            print(f"\n⚠️ Missing {len(missing_files)} files:")
            for f in missing_files[:5]:
                print(f"- {f}")
            if len(missing_files) > 5:
                print(f"... and {len(missing_files)-5} more")
                
        if corrupted_files:
            print(f"\n⚠️ Corrupted {len(corrupted_files)} files:")
            for f in corrupted_files[:5]:
                print(f"- {f}")
            if len(corrupted_files) > 5:
                print(f"... and {len(corrupted_files)-5} more")
        
        if resolution_issues:
            print(f"\n⚠️ Scaled {len(resolution_issues)} large images for GPU processing")
            print("Note: Box coordinates are relative to original size")

        return proposals

def main():
    # Add warning filter at start
    warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid")
    
    # Verify CUDA status
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    generator = ProposalGenerator()
    
    # Process all splits with correct paths
    splits = [
        ("base_train", "splits"),    # From splits/
        ("base_val", "splits"),       # From splits/
        ("novel_train", "json_files") # From json_files/
    ]
    
    for split_name, json_dir in splits:
        print(f"\n{'='*40}")
        print(f"Processing {split_name} split ({json_dir})")
        print(f"{'='*40}")
        generator.process_split(split_name, json_dir)

if __name__ == "__main__":
    main()