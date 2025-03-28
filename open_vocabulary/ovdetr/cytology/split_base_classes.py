"""
Robust Medical Data Splitter
- Handles class imbalance
- Maintains metadata-image alignment
- Preserves data integrity
"""

import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

def main():
    # Configuration - Modify these paths as needed
    INPUT_JSON = "dataset/json_files/base_train.json"
    OUTPUT_DIR = "dataset/splits"
    VAL_RATIO = 0.2  # 20% validation
    MIN_SAMPLES_PER_CLASS = 2  # Minimum annotations per class

    # Load base training data
    with open(INPUT_JSON) as f:
        base_data = json.load(f)

    # ---------------------------
    # 1. Class Validation & Cleaning
    # ---------------------------
    
    # Count annotations per class
    class_counts = defaultdict(int)
    for ann in base_data['annotations']:
        class_counts[ann['category_id']] += 1

    # Identify invalid classes with insufficient samples
    invalid_classes = {
        cid for cid, count in class_counts.items() 
        if count < MIN_SAMPLES_PER_CLASS
    }

    if invalid_classes:
        print(f"Removing {len(invalid_classes)} under-represented classes:")
        for cid in invalid_classes:
            class_name = next(
                c['name'] for c in base_data['categories'] 
                if c['id'] == cid
            )
            print(f"- {class_name} (ID: {cid})")

        # Filter out invalid classes
        base_data['annotations'] = [
            ann for ann in base_data['annotations']
            if ann['category_id'] not in invalid_classes
        ]
        base_data['categories'] = [
            cat for cat in base_data['categories']
            if cat['id'] not in invalid_classes
        ]

    # ---------------------------
    # 2. Image-Annotation Mapping
    # ---------------------------
    
    # Create mapping from image IDs to annotations
    image_ann_map = defaultdict(list)
    for ann in base_data['annotations']:
        image_ann_map[ann['image_id']].append(ann)

    # Create list of images with valid annotations
    valid_images = [
        img for img in base_data['images']
        if len(image_ann_map[img['id']]) > 0
    ]

    # ---------------------------
    # 3. Smart Stratification
    # ---------------------------
    
    # Calculate annotation counts per image
    annotation_counts = np.array([
        len(image_ann_map[img['id']]) 
        for img in valid_images
    ])

    # Create bins for stratification
    MAX_BINS = 5
    bins = np.linspace(
        start=0,
        stop=annotation_counts.max() + 1,
        num=MAX_BINS
    )
    stratify_bins = np.digitize(annotation_counts, bins)

    # ---------------------------
    # 4. Perform the Split
    # ---------------------------
    
    train_imgs, val_imgs = train_test_split(
        valid_images,
        test_size=VAL_RATIO,
        stratify=stratify_bins,
        random_state=42
    )

    # ---------------------------
    # 5. Create Output Data
    # ---------------------------
    
    def create_split_data(images):
        """Helper to create split dataset structure"""
        return {
            "type": base_data['type'],
            "categories": base_data['categories'],
            "images": images,
            "annotations": [
                ann for img in images
                for ann in image_ann_map[img['id']]]
        }

    train_data = create_split_data(train_imgs)
    val_data = create_split_data(val_imgs)

    # ---------------------------
    # 6. Save & Validate
    # ---------------------------
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save training split
    with open(os.path.join(OUTPUT_DIR, "base_train.json"), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    # Save validation split
    with open(os.path.join(OUTPUT_DIR, "base_val.json"), 'w') as f:
        json.dump(val_data, f, indent=2)

    # Final validation checks
    train_classes = {ann['category_id'] for ann in train_data['annotations']}
    val_classes = {ann['category_id'] for ann in val_data['annotations']}
    
    print("\nFinal Class Distribution:")
    print(f"Training Classes: {len(train_classes)}")
    print(f"Validation Classes: {len(val_classes)}")
    print(f"Common Classes: {len(train_classes & val_classes)}")
    print(f"Unique to Train: {len(train_classes - val_classes)}")
    print(f"Unique to Val: {len(val_classes - train_classes)}")

if __name__ == "__main__":
    main()