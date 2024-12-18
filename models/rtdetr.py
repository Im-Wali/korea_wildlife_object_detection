# models/rtedtr.py
from .base_model import BaseModel
from ultralytics import RTDETR
from datetime import datetime
import os

class RT_DETR(BaseModel):
    def __init__(self, model_name, img_size, device, data, optimizer, epochs,batch_size,hyp,cfg,wandb_token):
        super().__init__(model_name, img_size, device, data, optimizer,epochs,batch_size )
    

    def load_model(self):
        # YOLOv11 모델 초기화 또는 가중치 로드
        self.model = RTDETR(self.model_name)
        # self.model = RTDETR("rtdetr-x.pt")
        self.model.to(self.device)

    def train(self):
        # 현재 날짜와 시간 가져오기
        now = datetime.now()
        os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8)) 

        # 원하는 날짜 및 시간 형식으로 변환
        formatted_time = now.strftime("%Y%m%d_%H%M")
        self.model.train(data=self.data, 
                        epochs=self.epochs, 
                        imgsz=self.img_size,
                        batch=self.batch_size,
                        project='runs/rtdetr',   # 저장 경로의 상위 폴더 이름
                        name=f'rtdetr_{self.epochs}_{formatted_time}' ,# 하위 폴더 이름
                        workers=2,
                        lr0=0.005
                        ) 
    def val(self):
        # 현재 날짜와 시간 가져오기
        now = datetime.now()
        os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8)) 

        # 원하는 날짜 및 시간 형식으로 변환
        formatted_time = now.strftime("%Y%m%d_%H%M")
        self.model.val(data=self.data, 
                        imgsz=self.img_size,
                        batch=self.batch_size,
                        project='runs/rtdetr',   # 저장 경로의 상위 폴더 이름
                        name=f'rtdetr_{self.epochs}_val_{formatted_time}' ,# 하위 폴더 이름
                        workers=2
                        ) 
    
    def save(self):
        #  학습 완료된 모델 저장
        fine_tuned_model_path = f"pt/rtdetr/rtdetr_l_{self.epochs}.pt"
        self.model.save(fine_tuned_model_path)
