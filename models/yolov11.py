# models/yolo.py
from ultralytics import YOLO
from .base_model import BaseModel

class YOLOv11(BaseModel):
    def __init__(self, model_name, img_size, device, data, optimizer, epochs,hyp,cfg,wandb_token):
        super().__init__(model_name, img_size, device, data, optimizer, epochs)
        

    def load_model(self):
        # YOLOv11 모델 초기화 또는 가중치 로드
        self.model = YOLO(self.model_name)
        # self.model = YOLO("yolo11s.pt")
        # self.model = YOLO("yolo11m.pt")
        # self.model = YOLO("yolo11l.pt")
        # self.model = YOLO("yolo11x.pt")
        self.model.to(self.device)

    def train(self):
        self.model.train(data=self.data, epochs=self.epochs, imgsz=self.img_size)

    def save(self):
        #  학습 완료된 모델 저장
        fine_tuned_model_path = f"../model/yolov11/yolo_l_{self.epochs}.pt"
        self.model.save(fine_tuned_model_path)
