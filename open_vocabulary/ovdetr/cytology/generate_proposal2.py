"""
Medical Proposal Generation (COCO-Compatible)
- Works with pretrained COCO weights
- 512x512 medical image support
- CPU compatible
- Proper box scaling and filtering
"""

import torch
import cv2
import numpy as np
import pickle
import json
from tqdm.auto import tqdm
from pathlib import Path
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class ProposalGenerator:
    def __init__(self):
        self.cfg = get_cfg()
        
        # 1. Original COCO configuration
        self.cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        
        # 2. Medical adaptations
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.INPUT.MIN_SIZE_TEST = 512
        self.cfg.INPUT.MAX_SIZE_TEST = 512
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # From 0.3
        self.cfg.MODEL.RPN.SCORE_THRESH_TEST = 0.2  # Add this line
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000  # From 2000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1500  # From 1000
        
        # 3. Load COCO weights
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        
        # 4. Initialize predictor
        self.predictor = DefaultPredictor(self.cfg)

    def _filter_medical_boxes(self, boxes, scores, image_size):
        """Medical-adapted filtering with dynamic scaling"""
        img_w, img_h = image_size
        max_area = 0.95 * img_w * img_h
        keep = []
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1 = max(0, box[0])
            y1 = max(0, box[1])
            x2 = min(img_w, box[2])
            y2 = min(img_h, box[3])
            
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            # More lenient criteria
            if score > 0.1 and area > 100 and w < 350 and h < 350:  # Changed thresholds
                keep.append(i)
                
        return boxes[keep], scores[keep]

    def process_split(self, split_name, json_dir="splits"):
        """Process dataset split with proper error handling"""
        json_path = f"dataset/{json_dir}/{split_name}.json"
        output_path = f"dataset/proposals/{split_name}_proposals.pkl"
        
        with open(json_path) as f:
            data = json.load(f)
        
        proposals = {}
        missing_files = []
        corrupted_files = []
        
        for img in tqdm(data['images'], desc=f"Processing {split_name}"):
            img_path = Path("dataset/updated_train") / img['file_name']
            
            if not img_path.exists():
                missing_files.append(str(img_path))
                continue
                
            try:
                # Read image with orientation handling
                image = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if image is None:
                    raise ValueError("Failed to read image")
                
                # Convert color space
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = image.shape[:2]
                
                # Generate proposals
                with torch.no_grad():
                    outputs = self.predictor(image)
                
                # Get and filter boxes
                instances = outputs['instances']
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                scores = instances.scores.cpu().numpy()
                
                # Filter and store
                filtered_boxes, filtered_scores = self._filter_medical_boxes(
                    boxes, scores, (orig_w, orig_h)
                )
                
                proposals[img['id']] = {
                    'boxes': filtered_boxes,
                    'scores': filtered_scores,
                    'original_size': (orig_w, orig_h)
                }
                
            except Exception as e:
                corrupted_files.append(f"{img_path}: {str(e)}")
                continue
        
        # Save results with protocol 4 for compatibility
        with open(output_path, 'wb') as f:
            pickle.dump(proposals, f, protocol=4)
            
        # Print diagnostics
        print(f"\n{split_name} complete. Generated {sum(len(v['boxes']) for v in proposals.values())} proposals")
        if missing_files:
            print(f"Missing {len(missing_files)} files")
        if corrupted_files:
            print(f"Corrupted {len(corrupted_files)} files")
            
        return proposals

def main():
    generator = ProposalGenerator()
    
    splits = [
        ("base_train", "splits"),
        ("base_val", "splits"),
        ("novel_train", "json_files")
    ]
    
    for split_name, json_dir in splits:
        print(f"\n{'='*40}")
        print(f"Processing {split_name}")
        generator.process_split(split_name, json_dir)

if __name__ == "__main__":
    main()