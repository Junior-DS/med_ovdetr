import json
import random
from collections import defaultdict

def split_classes(original_json, base_json, novel_json, split_ratio=0.8):
    # Load original metadata
    with open(original_json) as f:
        data = json.load(f)
    
    # Get all categories and shuffle
    categories = data['categories']
    random.shuffle(categories)
    
    # Split categories into base and novel
    split_idx = int(len(categories) * split_ratio)
    base_cats = {c['id'] for c in categories[:split_idx]}
    novel_cats = {c['id'] for c in categories[split_idx:]}
    
    print(f"Splitting {len(categories)} classes into:")
    print(f"- Base: {len(base_cats)} classes")
    print(f"- Novel: {len(novel_cats)} classes")
    
    # Create mappings
    image_ann_map = defaultdict(list)
    for ann in data['annotations']:
        image_ann_map[ann['image_id']].append(ann)
    
    # Split datasets
    base_data = {
        "type": data['type'],
        "categories": categories[:split_idx],
        "images": [],
        "annotations": []
    }
    
    novel_data = {
        "type": data['type'],
        "categories": categories[split_idx:],
        "images": [],
        "annotations": []
    }
    
    # Process images
    for img in data['images']:
        img_id = img['id']
        annotations = image_ann_map.get(img_id, [])
        
        # Split annotations
        base_anns = [a for a in annotations if a['category_id'] in base_cats]
        novel_anns = [a for a in annotations if a['category_id'] in novel_cats]
        
        # Add to base dataset if has base annotations
        if base_anns:
            base_data['images'].append(img)
            base_data['annotations'].extend(base_anns)
            
        # Add to novel dataset if has novel annotations
        if novel_anns:
            novel_data['images'].append(img)
            novel_data['annotations'].extend(novel_anns)
    
    # Save splits
    with open(base_json, 'w') as f:
        json.dump(base_data, f, indent=2)
    
    with open(novel_json, 'w') as f:
        json.dump(novel_data, f, indent=2)
    
    print(f"\nSplit statistics:")
    print(f"Base set: {len(base_data['images'])} images, {len(base_data['annotations'])} annotations")
    print(f"Novel set: {len(novel_data['images'])} images, {len(novel_data['annotations'])} annotations")

if __name__ == "__main__":
    original_json = "dataset/updated_train.json"
    base_json = "dataset/base_train.json"
    novel_json = "dataset/novel_train.json"
    
    split_classes(original_json, base_json, novel_json, split_ratio=0.8)