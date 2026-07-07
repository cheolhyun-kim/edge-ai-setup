# practice2.py
# 슬라이드 48 · 실습 2: 한 장 캡처해서 저장하기
#
# 실행: python src/practice2.py
# 확인: ls -lh ../outputs/capture.jpg
#
# 저장 위치: edge-ai-day1/outputs/capture.jpg

import cv2
from pathlib import Path
from camera_utils import find_camera

CAMERA_ID = find_camera()

ROOT    = Path(__file__).parent.parent
out_dir = ROOT / "outputs"
out_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError(f"Camera open failed (id={CAMERA_ID})")

ret, frame = cap.read()

if ret:
    save_path = out_dir / "capture.jpg"
    cv2.imwrite(str(save_path), frame)
    print(f"saved: {save_path}")
else:
    print("frame read failed")

cap.release()
