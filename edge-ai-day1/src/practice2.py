# practice2.py
# 슬라이드 48 · 실습 2: 한 장 캡처해서 저장하기
#
# 실행: python src/practice2.py
# 확인:
#   ls -lh ../outputs/
#   file ../outputs/capture.jpg
#
# 저장 위치: edge-ai-day1/outputs/capture.jpg

import cv2
from pathlib import Path

CAMERA_ID = 0  # 카메라 번호가 다르면 여기를 수정

# src/ 기준으로 한 단계 위가 프로젝트 루트 (edge-ai-day1/)
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
