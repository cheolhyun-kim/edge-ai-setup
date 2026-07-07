# day3_03_benchmark_yolo.py
# 슬라이드 17~18, 20 · FPS / latency 측정 → CSV 저장
#
# 실행:
#   python src/day3_03_benchmark_yolo.py
#   python src/day3_03_benchmark_yolo.py --model ../models/yolo11n.pt --imgsz 320 --frames 300
#   python src/day3_03_benchmark_yolo.py --model ../models/yolo11n.onnx --imgsz 320 --frames 300
#   python src/day3_03_benchmark_yolo.py --model ../models/yolo11n_ncnn_model --imgsz 320
#
# 출력: ../outputs/benchmarks/yolo_fps_{model}_{imgsz}.csv
# 제출할 값: 평균 FPS, 평균 latency(ms), imgsz, conf, 모델명

import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO benchmark — FPS/latency to CSV")
    parser.add_argument("--model",  default=str(ROOT / "models" / "yolo11n.pt"))
    parser.add_argument("--source", default="0")
    parser.add_argument("--imgsz",  type=int,   default=320)
    parser.add_argument("--conf",   type=float, default=0.35)
    parser.add_argument("--frames", type=int,   default=300,
                        help="측정할 프레임 수 (기본 300)")
    parser.add_argument("--width",  type=int,   default=640)
    parser.add_argument("--height", type=int,   default=480)
    return parser.parse_args()


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def main():
    args   = parse_args()
    source = parse_source(args.source)

    out_dir = ROOT / "outputs" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_tag = Path(args.model).stem
    csv_path  = out_dir / f"yolo_fps_{model_tag}_imgsz{args.imgsz}.csv"

    print(f"loading model: {args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"측정 시작 — {args.frames}프레임 / imgsz={args.imgsz} / model={Path(args.model).name}")
    print(f"저장: {csv_path}")

    frame_count   = 0
    total_latency = 0.0
    start_time    = time.perf_counter()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "latency_ms", "avg_fps", "detections"])

        while frame_count < args.frames:
            ok, frame = cap.read()
            if not ok:
                print("frame read failed")
                break

            t0         = time.perf_counter()
            results    = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            latency_ms = (time.perf_counter() - t0) * 1000

            frame_count   += 1
            total_latency += latency_ms
            elapsed        = time.perf_counter() - start_time
            avg_fps        = frame_count / elapsed
            detections     = len(results[0].boxes)

            writer.writerow([frame_count, f"{latency_ms:.2f}", f"{avg_fps:.2f}", detections])

            if frame_count % 50 == 0:
                print(f"  [{frame_count:>3}/{args.frames}]  "
                      f"latency={latency_ms:.1f}ms  avg_fps={avg_fps:.1f}  det={detections}")

    cap.release()

    avg_lat     = total_latency / frame_count if frame_count else 0
    overall_fps = frame_count / (time.perf_counter() - start_time)

    print(f"\n=== 결과 ===")
    print(f"  모델          : {Path(args.model).name}")
    print(f"  imgsz         : {args.imgsz}")
    print(f"  측정 프레임   : {frame_count}")
    print(f"  평균 latency  : {avg_lat:.1f} ms")
    print(f"  평균 FPS      : {overall_fps:.1f}")
    print(f"  CSV           : {csv_path}")


if __name__ == "__main__":
    main()
