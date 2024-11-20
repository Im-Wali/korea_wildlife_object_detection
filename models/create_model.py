# models/model_factory.py
from .yolov7 import YOLOv7
from .yolov11 import YOLOv11
from .rtdetr import RTDETR

def create_model(model_type, **kwargs):
    print(f"model_type : {model_type}")
    if model_type == "YOLOv7":
        return YOLOv7(**kwargs)
    elif model_type == "YOLOv11":
        return YOLOv11(**kwargs)
    elif model_type == "RTDETR":
        return RTDETR(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
