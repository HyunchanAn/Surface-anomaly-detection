# 표면 이상 탐지 시스템 요구사항 (Surface Anomaly Detection Requirements)

## 시스템 요구사항 (System Requirements)
- 운영체제 (OS): Windows 10/11 또는 Linux
- GPU: NVIDIA RTX 4060 (노트북) 이상 권장 (PatchCore 학습용)
- CUDA: 11.8 또는 12.x 호환 드라이버

## Python 의존성 (Python Dependencies)
아래 내용을 `requirements.txt`로 저장하고 `pip install -r requirements.txt` 명령어로 설치하세요.

```txt
# Core Engine
anomalib[full]==1.0.0
torch>=2.0.0
torchvision>=0.15.0

# Data Handling & Processing
openvino-dev>=2023.0  # Optional for OpenVINO optimization
pandas
numpy
matplotlib
opencv-python

# UI
streamlit>=1.30.0
watchdog # for streamlit hot-reloading
```

## 설치 가이드 (Setup Instructions - 가정용 PC)
1. Python 3.10 설치 (권장).
2. 가상 환경 생성: `python -m venv venv`.
3. 가상 환경 활성화: `.\venv\Scripts\activate`.
4. 패키지 설치: `pip install -r requirements.txt`.
