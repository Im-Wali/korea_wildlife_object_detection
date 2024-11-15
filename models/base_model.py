# models/base_model.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name, img_size, device, data, optimizer, epochs):
        self.model_name = model_name
        self.img_size = img_size
        self.device = device
        self.data = data
        self.optimizer = optimizer
        self.epochs = epochs
        self.model = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save(self, save_path):
        pass
