import os
import json
from PIL import Image, ImageOps
from tqdm import tqdm

def process_images(input_dir, output_dir, json_path, new_json_path, target_size=512):
    # Load metadata
    with open(json_path) as f:
        data = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mapping for original image IDs to annotations
    image_ann_map = {img["id"]: [] for img in data["images"]}
    for ann in data["annotations"]:
        image_ann_map[ann["image_id"]].append(ann)
    
    # Create new data structure
    new_data = {
        "type": data["type"],
        "categories": data["categories"],
        "images": [],
        "annotations": []
    }
    
    # Process images
    for img in tqdm(data["images"], desc="Processing images"):
        # Load original image
        bmp_path = os.path.join(input_dir, img["file_name"])
        image = Image.open(bmp_path)
        
        # Get original dimensions
        orig_w, orig_h = image.size
        
        # Resize with padding while maintaining aspect ratio
        resized_img = ImageOps.pad(image, (target_size, target_size), color="black", centering=(0.5, 0.5))
        
        # Calculate scaling factor and padding
        ratio = min(target_size / orig_w, target_size / orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        
        # Calculate padding offsets
        pad_left = (target_size - new_w) // 2
        pad_top = (target_size - new_h) // 2
        
        # Generate new filename and ID
        base_name = os.path.splitext(img["file_name"])[0]
        new_filename = f"{base_name}.jpg"
        new_id = f"{base_name}.jpg"
        
        # Save as JPEG
        resized_img.save(os.path.join(output_dir, new_filename), "JPEG")
        
        # Update image metadata
        new_image = {
            "file_name": new_filename,
            "id": new_id,
            "width": target_size,
            "height": target_size,
            **{k: v for k, v in img.items() if k not in ["file_name", "id", "width", "height"]}
        }
        new_data["images"].append(new_image)
        
        # Update annotations with proper scaling and padding
        for ann in image_ann_map[img["id"]]:
            # Original bounding box coordinates
            x, y, w, h = ann["bbox"]
            
            # Scale coordinates
            scaled_x = x * ratio
            scaled_y = y * ratio
            scaled_w = w * ratio
            scaled_h = h * ratio
            
            # Apply padding offsets
            padded_x = scaled_x + pad_left
            padded_y = scaled_y + pad_top
            
            # Create new annotation
            new_ann = {
                "image_id": new_id,
                "bbox": [
                    round(padded_x, 2),
                    round(padded_y, 2),
                    round(scaled_w, 2),
                    round(scaled_h, 2)
                ],
                "area": round(scaled_w * scaled_h, 2),
                **{k: v for k, v in ann.items() if k not in ["image_id", "bbox", "area"]}
            }
            new_data["annotations"].append(new_ann)
    
    # Save updated metadata
    with open(new_json_path, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    input_dir = "dataset/train"
    output_dir = "dataset/updated_train/"
    original_json = "dataset/train.json"
    updated_json = "dataset/updated_train.json"
    
    process_images(input_dir, output_dir, original_json, updated_json, target_size=512)