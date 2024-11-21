# models/yolo.py
from .base_model import BaseModel
import subprocess
import wandb
import os

class YOLOv7(BaseModel):
    def __init__(self, model_name, img_size, device, data, optimizer, epochs, batch_size, hyp=None, cfg=None, wandb_token=None):
        super().__init__(model_name, img_size, device, data, optimizer, epochs,batch_size)
        self.hyp = hyp
        self.cfg = cfg
        self.wandb_token = wandb_token

    def load_model(self):
        # YOLOv7 모델 초기화 또는 가중치 로드
        pass

    def train(self):
        # subprocess로 YOLOv7 학습 스크립트 실행
        command = [
            "python", "./yolov7/train.py",
            "--weights", self.model_name,
            "--img-size", str(self.img_size),
            "--device", self.device,
            "--data", self.data,
            "--batch-size", str(self.batch_size),
            # "--optimizer",
            '--' + self.optimizer,
            "--epochs", str(self.epochs)
        ]
        if self.hyp:
            command.extend(["--hyp", self.hyp])
        if self.cfg:
            command.extend(["--cfg", self.cfg])
        if self.wandb_token:
            wandb.login(key=self.wandb_token)
        os.environ["NUMEXPR_MAX_THREADS"] = "8"
        os.environ["OMP_NUM_THREADS"] = "1"

        subprocess.run(command)

    def save(self):
        # 학습된 모델을 저장
        pass
