import json
from pycocotools.coco import COCO

def validate_conversion(anno_path, image_dir):
    # Test loading with pycocotools
    coco = COCO(anno_path)
    
    print(f"Dataset contains:")
    print(f"- {len(coco.dataset['categories'])} categories")
    print(f"- {len(coco.dataset['images'])} images")
    print(f"- {len(coco.dataset['annotations'])} annotations")
    
    # Test first image
    img_id = coco.dataset['images'][0]['id']
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    print("\nFirst image validation:")
    print(f"Image ID: {img_id}")
    print(f"Annotations count: {len(anns)}")
    print(f"Example annotation: {anns[0] if anns else 'None'}")

if __name__ == "__main__":
    validate_conversion(
        anno_path="lguided/dataset/cytology_coco/annotations/instances_all.json",
        image_dir="dataset/train"
    )