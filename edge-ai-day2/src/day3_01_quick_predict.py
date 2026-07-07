# day3_01_quick_predict.py
# 슬라이드 11 · 빠른 첫 성공: webcam direct predict
#
# 실행: python src/day3_01_quick_predict.py
#
# 목적: 복잡한 코드 없이 모델·카메라가 정상인지 먼저 확인
#   - 모델 다운로드 확인
#   - 카메라 입력 확인
#   - Ultralytics 실행 확인
#   - 속도 체감 확인
#
# 성공하면 → day3_02_realtime_yolo_basic.py 로 이동
#
# CLI 방식 (터미널에서 직접 실행 가능):
#   yolo predict model=../models/yolo11n.pt source=0 imgsz=320 conf=0.35 show=True

from pathlib import Path
from ultralytics import YOLO

# src/ 기준으로 한 단계 위가 프로젝트 루트 (edge-ai-day2/)
ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "yolo11n.pt"
CAMERA_ID  = 0      # 카메라 번호가 다르면 여기를 수정
IMG_SIZE   = 320
CONF       = 0.35

model = YOLO(str(MODEL_PATH))

# source 는 카메라 번호(int) 또는 영상 파일 경로(str) 모두 가능
results = model.predict(source=CAMERA_ID, imgsz=IMG_SIZE, conf=CONF, show=True)
