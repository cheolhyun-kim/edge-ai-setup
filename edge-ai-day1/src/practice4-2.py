# practice4-2.py
# 슬라이드 51 · 실습 4-2: 해상도 설정과 성능 trade-off
#
# 실행: python src/practice4-2.py

import cv2
from camera_utils import find_camera

CAMERA_ID = find_camera()

# 요청할 해상도 — 변경해서 실험해보세요
REQ_WIDTH  = 640
REQ_HEIGHT = 480
REQ_FPS    = 30

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError(f"Camera open failed (id={CAMERA_ID})")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  REQ_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_HEIGHT)
cap.set(cv2.CAP_PROP_FPS,          REQ_FPS)

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
f = cap.get(cv2.CAP_PROP_FPS)

print(f"요청: {REQ_WIDTH}x{REQ_HEIGHT} @ {REQ_FPS}fps")
print(f"적용: {int(w)}x{int(h)} @ {f:.1f}fps")

cap.release()
