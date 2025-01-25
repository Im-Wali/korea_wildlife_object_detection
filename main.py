import sys
import yaml
import os
from models.create_model import create_model

def load_config(config_path):
    """설정 파일 로드 함수"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(config_path="config.yaml", mode="train"):
    # 설정 파일 로드
    config = load_config(config_path)
    
    model_params = config["model_params"]
    training_params = config["training_params"]

    # 환경 변수 설정
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP에서 사용할 쓰레드 수 제한
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print("Model Parameters:", model_params)

    # 모델 생성
    model = create_model(**model_params)
    model.load_model()

    # 학습 관리 클래스 생성 및 학습 시작
    if(mode == "train"):
        model.train()
    else:
        model.val()

    # 모델 저장
    model.save()

if __name__ == "__main__":
    # 인자값 확인 및 설정 파일 경로 지정
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    mode = sys.argv[2] if len(sys.argv) > 2 else "train"
    
    # 메인 실행
    main(config_file,mode)
