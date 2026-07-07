# day3_06_capture_custom_dataset.py
# 슬라이드 63 · 커스텀 데이터셋 수집 — 클래스별 폴더에 스페이스바로 저장
#
# 실행:
#   python src/day3_06_capture_custom_dataset.py --class scissors --camera 0
#   python src/day3_06_capture_custom_dataset.py --class rock     --camera 0
#   python src/day3_06_capture_custom_dataset.py --class paper    --camera 0
#
# 저장 구조:
#   ../outputs/captures/scissors/scissors_0001.jpg
#   ../outputs/captures/rock/rock_0001.jpg
#   ../outputs/captures/paper/paper_0001.jpg
#
# 촬영 규칙:
#   - 스페이스바: 저장 / q: 종료
#   - 한 클래스 촬영이 끝나면 다음 클래스명으로 다시 실행
#   - 저장 후 흐릿한 사진은 바로 삭제
#   - 조명/배경/거리/각도를 다양하게 변경하며 촬영
#
# practice5.py 의 확장판 — Day05 통합 파이프라인과 연결됩니다

import argparse
import time
from pathlib import Path

import cv2
from camera_utils import find_camera

ROOT = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Custom dataset capture")
    parser.add_argument("--class",   dest="class_name", required=True,
                        help="클래스 이름 (영문 소문자, 예: scissors)")
    parser.add_argument("--camera",  type=int, default=find_camera(),
                        help="카메라 번호 (기본: 자동 감지)")
    parser.add_argument("--width",   type=int, default=640)
    parser.add_argument("--height",  type=int, default=480)
    return parser.parse_args()


def main():
    args       = parse_args()
    class_name = args.class_name.strip().lower()

    # 공백 → 언더스코어
    if " " in class_name:
        print("[경고] 클래스 이름 공백을 언더스코어로 변환합니다.")
        class_name = class_name.replace(" ", "_")

    # 저장 폴더: ../outputs/captures/{class_name}/
    save_dir = ROOT / "outputs" / "captures" / class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 기존 이미지 수부터 이어서 카운트
    count = len(list(save_dir.glob("*.jpg")))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed: {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"클래스   : {class_name}")
    print(f"저장 폴더: {save_dir}")
    print(f"기존 이미지: {count}장 (이어서 저장)")
    print("스페이스바: 저장  |  q: 종료")
    print("-" * 42)

    last_save = 0.0   # 연속 저장 방지

    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame read failed")
            break

        display = frame.copy()
        cv2.putText(display, f"class: {class_name}  saved: {count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "SPACE: save  |  q: quit",
                    (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(f"capture — {class_name}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == 32:                    # 스페이스바
            now = time.time()
            if now - last_save < 0.3:     # 0.3초 이내 중복 저장 방지
                continue
            last_save = now

            filename = save_dir / f"{class_name}_{count:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"saved: {filename}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n총 {count}장 저장 완료 → {save_dir}")
    print("다음 클래스 촬영 예시:")
    print(f"  python src/day3_06_capture_custom_dataset.py "
          f"--class <다음클래스> --camera {args.camera}")


if __name__ == "__main__":
    main()
