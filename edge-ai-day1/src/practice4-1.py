# practice4-1.py
# 슬라이드 50 · 실습 4-1: FPS 측정하기
#
# 실행: python src/practice4-1.py
# 종료: 창에서 q 키 입력
#
# 화면 왼쪽 위에 현재 FPS 가 실시간으로 표시됩니다.
# 오늘은 카메라 FPS, Day03 에서는 YOLO 추론 후 FPS 를 비교합니다.

import time
import cv2

CAMERA_ID = 0  # 카메라 번호가 다르면 여기를 수정

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError(f"Camera open failed (id={CAMERA_ID})")

print("FPS 측정 시작 — 종료하려면 q 를 누르세요.")

prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    fps = 1.0 / (now - prev)
    prev = now

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("FPS", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
