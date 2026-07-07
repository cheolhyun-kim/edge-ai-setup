# day3_01_quick_predict.py
# 슬라이드 11 · 빠른 첫 성공: webcam direct predict
#
# 실행: python src/day3_01_quick_predict.py
#
# 목적: 복잡한 코드 없이 모델·카메라가 정상인지 먼저 확인
# 성공하면 → day3_02_realtime_yolo_basic.py 로 이동
#
# CLI 방식:
#   yolo predict model=../models/yolo11n.pt source=0 imgsz=320 conf=0.35 show=True

from pathlib import Path
from ultralytics import YOLO
from camera_utils import find_camera

ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "yolo11n.pt"
CAMERA_ID  = find_camera()
IMG_SIZE   = 320
CONF       = 0.35

model   = YOLO(str(MODEL_PATH))
results = model.predict(source=CAMERA_ID, imgsz=IMG_SIZE, conf=CONF, show=True)
