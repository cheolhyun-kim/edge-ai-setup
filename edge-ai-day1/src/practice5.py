# practice5.py
# 슬라이드 52 · 실습 5: 데이터 수집용 캡처 프로그램
#
# 실행: python src/practice5.py
# 조작:
#   스페이스바 → 이미지 저장
#   q          → 종료
#
# 저장 위치: edge-ai-day1/outputs/captures/img_0000.jpg ...
# Day03 에서 커스텀 데이터셋 수집 도구로 그대로 이어집니다.
#
# ※ 슬라이드 원본의 'Import time' 오타를 'import time' 으로 수정

import time
import cv2
from pathlib import Path

CAMERA_ID = 0  # 카메라 번호가 다르면 여기를 수정

# src/ 기준으로 한 단계 위가 프로젝트 루트 (edge-ai-day1/)
ROOT    = Path(__file__).parent.parent
out_dir = ROOT / "outputs" / "captures"
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError(f"Camera open failed (id={CAMERA_ID})")

# 이미 저장된 파일 수부터 이어서 카운트
count = len(list(out_dir.glob("img_*.jpg")))

print(f"저장 폴더: {out_dir}")
print(f"기존 이미지: {count}장 (이어서 저장)")
print("스페이스바: 저장 | q: 종료")

last_save = 0.0  # 연속 저장 방지용 타임스탬프

while True:
    ret, frame = cap.read()
    if not ret:
        print("frame read failed")
        break

    # 저장 장수 화면 표시
    cv2.putText(frame, f"saved: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "SPACE: save  q: quit", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == 32:  # 스페이스바
        now = time.time()
        if now - last_save < 0.3:   # 0.3초 이내 중복 저장 방지
            continue
        last_save = now

        save_path = out_dir / f"img_{count:04d}.jpg"
        cv2.imwrite(str(save_path), frame)
        print(f"saved: {save_path}")
        count += 1

cap.release()
cv2.destroyAllWindows()
print(f"\n총 {count}장 저장 완료 → {out_dir}")
