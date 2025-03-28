import torch
import copy
from clip import clip
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path
from collections import defaultdict

# Medical dataset configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()  # Disable gradient computation

# 1. Path Configuration for Cytology
base_path = Path("OV-DETR/ovdetr/cytology")
json_path = base_path / "splits/base_train.json"  # Your base training annotations
image_dir = base_path / "updated_train"  # Directory with 512x512 cytology images
save_path = base_path / "clip_features_cytology.pkl"  # Output path

# Load class splits
with open(base_path / "json_files/split_classes.json") as f:
    split_config = json.load(f)
    
base_classes = split_config["seen"]
novel_classes = split_config["unseen"]

# Create medical class mapping only for base classes
class_mapping = {cls: idx for idx, cls in enumerate(base_classes)}

# Load annotations and filter base classes only
with open(json_path) as f:
    data = json.load(f)

# Validate categories
all_cats = {cat["id"]: cat["name"] for cat in data["categories"]}
base_cat_ids = [cat_id for cat_id, name in all_cats.items() if name in base_classes]
# 3. Annotation Processing for Cytology
img2ann_gt = defaultdict(list)
for ann in data["annotations"]:
    if ann["category_id"] not in base_cat_ids:
        print(f"Skipping novel class annotation: {ann}")
        continue
    img2ann_gt[ann["image_id"]].append(ann)

# 4. Feature Extraction
feature_dict = defaultdict(list)

for image_id in tqdm(img2ann_gt.keys(), desc="Processing Cytology Images"):
    # Medical image path handling
    img_file = image_dir / f"{image_id}"  # Adjust extension if needed
    
    if not img_file.exists():
        # Try alternative extensions
        for ext in [".jpg", ".png", ".jpeg"]:
            alt_path = image_dir / f"{image_id}{ext}"
            if alt_path.exists():
                img_file = alt_path
                break
    
    if not img_file.exists():
        print(f"Missing image: {img_file}")
        continue
    try:
        image = Image.open(img_file).convert("RGB")
    except FileNotFoundError:
        print(f"Missing image: {img_file}")
        continue

    for ann in img2ann_gt[image_id]:
        # 5. Bounding Box Handling
        bbox = ann['bbox']  # Expected format [x_min, y_min, width, height]
        
        # Convert to [x0, y0, x1, y1]
        x0, y0 = bbox[0], bbox[1]
        x1 = x0 + bbox[2]
        y1 = y0 + bbox[3]
        
        # Skip small lesions under 16px
        if (x1 - x0) < 16 or (y1 - y0) < 16:
            continue
            
        try:
            # Crop and preprocess lesion ROI
            roi = image.crop((x0, y0, x1, y1))
            roi_tensor = preprocess(roi).unsqueeze(0).to(device)
            
            # Extract CLIP features
            with torch.no_grad():
                features = model.encode_image(roi_tensor)
            
            # Store by medical class ID
            feature_dict[ann['category_id']].append(features.cpu())
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
# Final validation
missing_base = set(base_classes) - set(class_mapping.keys())
if missing_base:
    raise ValueError(f"Missing base classes in annotations: {missing_base}")

novel_in_base = set(novel_classes) & set(class_mapping.keys())
if novel_in_base:
    raise ValueError(f"Novel classes in base annotations: {novel_in_base}")

# 6. Save Medical Features
final_features = {
    cls_id: torch.cat(feats) 
    for cls_id, feats in feature_dict.items()
}

torch.save(final_features, save_path)
print(f"Saved medical CLIP features to {save_path}")