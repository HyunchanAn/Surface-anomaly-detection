# Surface Anomaly Detection Requirements

## System Requirements
- OS: Windows 10/11 or Linux
- GPU: NVIDIA RTX 4060 (Laptop) or better recommended for PatchCore training
- CUDA: 11.8 or 12.x compatible driver

## Python Dependencies
Save this as `requirements.txt` and install via `pip install -r requirements.txt`

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

## Setup Instructions (for Home Machine)
1. Install Python 3.10 (Recommended).
2. Create virtual environment: `python -m venv venv`.
3. Activate: `.\venv\Scripts\activate`.
4. Install: `pip install -r requirements.txt`.
