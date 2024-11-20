# models/base_model.py
from abc import ABC, abstractmethod
import os

class BaseModel(ABC):
    def __init__(self, model_name, img_size, device, data, optimizer, epochs):
        self.model_name = model_name
        self.img_size = img_size
        self.device = device
        self.data = os.path.dirname(os.path.abspath(data)) + "/" + os.path.basename(data)
        # self.data = data
        self.optimizer = optimizer
        self.epochs = epochs
        self.model = None
        # print(self.data)

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save(self, save_path):
        pass

    def load_data(self):
                # YAML 파일 로드 및 경로 변환
        with open(self.data, "r") as f:
            config = self.data.safe_load(f)

        # YAML 파일에 있는 train/val 경로를 절대 경로로 변환
        base_path = os.path.dirname(os.path.abspath(self.data))
        config["train"] = os.path.join(base_path, config["train"])
        config["val"] = os.path.join(base_path, config["val"])

