# practice4-2.py
# 슬라이드 51 · 실습 4-2: 해상도 설정과 성능 trade-off
#
# 실행: python src/practice4-2.py
#
# 요청한 해상도와 실제 카메라가 적용한 해상도가 다를 수 있습니다.
# 출력값을 보고 카메라가 지원하는 해상도를 파악합니다.
#
# Pi 권장 시작값: 640×480
# 해상도를 올리면 → YOLO 전처리·추론 시간 증가 → FPS 하락

import cv2

CAMERA_ID = 0   # 카메라 번호가 다르면 여기를 수정

# 요청할 해상도 — 변경해서 실험해보세요
REQ_WIDTH  = 640
REQ_HEIGHT = 480
REQ_FPS    = 30

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError(f"Camera open failed (id={CAMERA_ID})")

# 해상도 요청
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  REQ_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_HEIGHT)
cap.set(cv2.CAP_PROP_FPS,          REQ_FPS)

# 실제 적용값 확인
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
f = cap.get(cv2.CAP_PROP_FPS)

print(f"요청: {REQ_WIDTH}x{REQ_HEIGHT} @ {REQ_FPS}fps")
print(f"적용: {int(w)}x{int(h)} @ {f:.1f}fps")

cap.release()
