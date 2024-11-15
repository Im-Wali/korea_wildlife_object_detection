# models/yolo.py
from .base_model import BaseModel
from ultralytics import RTDETR

class RTDETR(BaseModel):
    def __init__(self, model_name, img_size, device, data, optimizer, epochs):
        super().__init__(model_name, img_size, device, data, optimizer, epochs)
        

    def load_model(self):
        # YOLOv11 모델 초기화 또는 가중치 로드
        self.model = RTDETR("rtdetr-l.pt")
        # self.model = RTDETR("rtdetr-x.pt")
        self.model.to(self.device)

    def train(self):
        self.model.train(data=self.data, epochs=self.epochs, imgsz=self.img_size)

    def save(self):
        #  학습 완료된 모델 저장
        fine_tuned_model_path = f"../model/rtdetr/rtdetr_l_{self.epochs}.pt"
        self.model.save(fine_tuned_model_path)
