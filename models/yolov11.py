# models/yolo.py
from ultralytics import YOLO
from .base_model import BaseModel
from datetime import datetime

class YOLOv11(BaseModel):
    def __init__(self, model_name, img_size, device, data, optimizer, epochs,batch_size,hyp,cfg,wandb_token):
        super().__init__(model_name, img_size, device, data, optimizer, epochs,batch_size)
        

    def load_model(self):
        # YOLOv11 모델 초기화 또는 가중치 로드
        self.model = YOLO(self.model_name)
        # self.model = YOLO("yolo11s.pt")
        # self.model = YOLO("yolo11m.pt")
        # self.model = YOLO("yolo11l.pt")
        # self.model = YOLO("yolo11x.pt")
        self.model.to(self.device)

    def train(self):
        # 현재 날짜와 시간 가져오기
        now = datetime.now()

        # 원하는 날짜 및 시간 형식으로 변환
        formatted_time = now.strftime("%Y%m%d_%H%M") 
        self.model.train(data=self.data, 
                        epochs=self.epochs, 
                        imgsz=self.img_size,
                        batch=self.batch_size,
                        project='runs/yolo11',   # 저장 경로의 상위 폴더 이름
                        name=f'yolo11_{self.epochs}_{formatted_time}'  ) # 하위 폴더 이름

    def save(self):
        #  학습 완료된 모델 저장
        fine_tuned_model_path = f"../model/yolov11/yolo_l_{self.epochs}.pt"
        self.model.save(fine_tuned_model_path)
