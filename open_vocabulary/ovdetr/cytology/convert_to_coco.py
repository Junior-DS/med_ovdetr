import json
import os
import cv2
from tqdm import tqdm
from pathlib import Path

def convert_cytology_to_coco(metadata_dir, image_dir, output_path, split_ratio=0.2):
    
    """Convert cytology dataset to COCO format with train/val split"""
    all_meta = []
    for f in Path(metadata_dir).glob('*.json'):
        with open(f) as j:
            all_meta.extend(json.load(j))

    # Create category mapping
    categories = {cell['type']: idx+1 for idx, cell in 
                  enumerate(sorted({c['type'] for m in all_meta for c in m['cells']}))}
    
    # Split dataset
    image_ids = list({m['image_id'] for m in all_meta})
    val_size = int(len(image_ids) * split_ratio)
    val_ids = set(image_ids[:val_size])
    
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    for split in ['train', 'val']:
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": v, "name": k} for k, v in categories.items()]
        }
        
        ann_id = 0
        for meta in tqdm(all_meta, desc=f'Processing {split}'):
            img_id = meta['image_id']
            img_path = Path(image_dir) / meta['image_path']
            
            if (split == 'val') ^ (img_id not in val_ids):
                continue
                
            # Copy image to split directory
            img = cv2.imread(str(img_path))
            new_path = f'cytology/images/{split}/{img_path.name}'
            cv2.imwrite(new_path, img)
            
            # Add image info
            coco_data['images'].append({
                "id": img_id,
                "file_name": img_path.name,
                "width": img.shape[1],
                "height": img.shape[0]
            })
            
            # Add annotations
            for cell in meta['cells']:
                coco_data['annotations'].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": categories[cell['type']],
                    "bbox": cell['bbox'],
                    "area": cell['bbox'][2] * cell['bbox'][3],
                    "iscrowd": 0
                })
                ann_id += 1
        
        
        # Save COCO format annotations
        with open(f'lguided/cytology/annotations/instances_{split}.json', 'w') as f:
            json.dump(coco_data, f)
            

if __name__ == "__main__":
    convert_cytology_to_coco(
        metadata_dir="dataset/train.json",
        image_dir="dataset/train",
        output_path="lguided"
    )
    

