from YOWOFormer.model.backbone2D import YOLOv8
from YOWOFormer.model.backbone2D.YOLOv11 import build_yolov11_backbone

def build_backbone2D(config):
    backbone_2D = config['backbone2D']

    if backbone_2D == 'yolov8':
        backbone2D = YOLOv8.build_yolov8(config)
    elif 'yolov11' in backbone_2D.lower():
        # Support yolov11_n, yolov11_s, yolov11_m, yolov11_l, yolov11_x
        backbone2D = build_yolov11_backbone(config)
    else:
        raise ValueError(f"Unknown backbone2D: {backbone_2D}")

    return backbone2D