# models/yolo.py
from .base_model import BaseModel
import subprocess

class YOLOv7(BaseModel):
    def __init__(self, model_name, img_size, device, data, optimizer, epochs, hyp=None, cfg=None, wandb_token=None):
        super().__init__(model_name, img_size, device, data, optimizer, epochs)
        self.hyp = hyp
        self.cfg = cfg
        self.wandb_token = wandb_token

    def load_model(self):
        # YOLOv7 모델 초기화 또는 가중치 로드
        pass

    def train(self):
        pass

    def save(self, save_path):
        # 학습된 모델을 저장
        pass
