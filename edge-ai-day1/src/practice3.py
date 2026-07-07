# practice3.py
# 슬라이드 49 · 실습 3: 실시간 미리보기 루프
#
# 실행: python src/practice3.py
# 종료: 창에서 q 키 입력
#
# 주의: GUI 창이 필요합니다
#   - 모니터를 직접 연결하거나 VNC 사용
#   - SSH 단독 환경에서는 imshow 가 동작하지 않음
#     → headless 환경이면 practice2.py 의 저장 방식으로 대체

import cv2

CAMERA_ID = 0  # 카메라 번호가 다르면 여기를 수정

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
