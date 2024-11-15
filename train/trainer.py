# train/trainer.py
import os
from datetime import datetime

class Trainer:
    def __init__(self, model):
        self.model = model
        self.save_dir = f"./outputs/{model.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def train(self):
        self.model.load_model()
        self.model.train()

    def save_model(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, "best_model.pt")
        self.model.save(save_path)
