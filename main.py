# main.py
from models import create_model
import yaml

# 설정 파일 로드
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_params = config["model_params"]
training_params = config["training_params"]

# 모델 생성
model = create_model(**model_params)

# 학습 관리 클래스 생성 및 학습 시작
model.train()

model.save()