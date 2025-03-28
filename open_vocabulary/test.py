# In a new training script:
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Your medical classes
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128]]  # Medical-optimized
# ... other medical-specific settings
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Start from COCO