# practice1.py
# 슬라이드 47 · 실습 1: 카메라 열기 최소 코드
#
# 실행: python src/practice1.py
# 성공 기준:
#   - ret 가 True 로 출력
#   - frame.shape 가 출력 (예: (480, 640, 3))
#   - 에러 없이 종료
#
# 카메라 번호가 0 이 아니면 VideoCapture(1) 또는 VideoCapture(2) 로 변경

import cv2

CAMERA_ID = 0  # 카메라 번호가 다르면 여기를 수정

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError(f"Camera open failed (id={CAMERA_ID})")

ret, frame = cap.read()

print("ret   :", ret)
print("shape :", frame.shape if ret else None)

cap.release()
