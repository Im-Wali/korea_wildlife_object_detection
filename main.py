# main.py
from models.create_model import create_model
import yaml
import os


# 설정 파일 로드
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_params = config["model_params"]
training_params = config["training_params"]


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP에서 사용할 쓰레드 수 제한

print(model_params)

# 모델 생성
model = create_model(**model_params)
model.load_model()
# 학습 관리 클래스 생성 및 학습 시작
model.train()

model.save()