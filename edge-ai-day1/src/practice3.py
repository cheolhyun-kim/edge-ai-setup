# practice3.py
# 슬라이드 49 · 실습 3: 실시간 미리보기 루프
#
# 실행: python src/practice3.py
# 종료: 창에서 q 키 입력
#
# 주의: GUI 창이 필요합니다 (모니터 직접 연결 또는 VNC 사용)

import cv2
from camera_utils import find_camera

CAMERA_ID = find_camera()

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError(f"Camera open failed (id={CAMERA_ID})")

print("실시간 미리보기 시작 — 종료하려면 q 를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("frame read failed")
        break

    cv2.imshow("Pi Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
