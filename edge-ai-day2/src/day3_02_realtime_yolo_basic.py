# day3_02_realtime_yolo_basic.py
# 슬라이드 13~14, 20 · 최소 실시간 YOLO — OpenCV 루프 + FPS 표시
#
# 실행:
#   python src/day3_02_realtime_yolo_basic.py
#   python src/day3_02_realtime_yolo_basic.py --source 1
#   python src/day3_02_realtime_yolo_basic.py --imgsz 480
#   python src/day3_02_realtime_yolo_basic.py --no-display     # headless
#   python src/day3_02_realtime_yolo_basic.py --model ../models/yolo11n.onnx
#
# 종료: q 키
#
# 주의: 모델 로드는 루프 밖에서 1번만 — 루프 안에서 로드하면 FPS 가 거의 나오지 않음

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# src/ 기준 프로젝트 루트 (edge-ai-day2/)
ROOT = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime YOLO basic")
    parser.add_argument("--model",      default=str(ROOT / "models" / "yolo11n.pt"))
    parser.add_argument("--source",     default="0",
                        help="카메라 번호(0,1,...) 또는 영상 파일 경로")
    parser.add_argument("--imgsz",      type=int,   default=320)
    parser.add_argument("--conf",       type=float, default=0.35)
    parser.add_argument("--width",      type=int,   default=640)
    parser.add_argument("--height",     type=int,   default=480)
    parser.add_argument("--no-display", action="store_true",
                        help="imshow 대신 ../outputs/demo_frames/ 에 30프레임마다 저장")
    return parser.parse_args()


def parse_source(value: str):
    """'0' → int(0),  'video.mp4' → str"""
    return int(value) if value.isdigit() else value


def open_camera(source, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_f = cap.get(cv2.CAP_PROP_FPS)
    print(f"camera: {real_w}x{real_h} @ {real_f:.1f}fps")
    return cap


def main():
    args   = parse_args()
    source = parse_source(args.source)

    # 모델 로드 — 루프 밖에서 1번만
    print(f"loading model: {args.model}")
    model = YOLO(args.model)

    cap = open_camera(source, args.width, args.height)

    save_dir = ROOT / "outputs" / "demo_frames"
    if args.no_display:
        save_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    prev      = time.perf_counter()

    print("실행 중 — 종료하려면 q 를 누르세요.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame read failed")
            break

        results   = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        annotated = results[0].plot()

        now = time.perf_counter()
        fps = 1.0 / (now - prev)
        prev = now

        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if args.no_display:
            if frame_idx % 30 == 0:
                path = save_dir / f"frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(path), annotated)
        else:
            cv2.imshow("YOLO Realtime", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
