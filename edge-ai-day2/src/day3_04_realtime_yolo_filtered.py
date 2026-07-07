# day3_04_realtime_yolo_filtered.py
# 슬라이드 31~35 · 클래스 필터링 + bbox 직접 그리기 + JSON 출력
#
# 실행:
#   python src/day3_04_realtime_yolo_filtered.py --target person
#   python src/day3_04_realtime_yolo_filtered.py --target cup --conf 0.25
#   python src/day3_04_realtime_yolo_filtered.py --target bottle --no-display
#
# 종료: q 키
# JSON 출력: 탐지 성공 시 터미널에 한 줄 출력

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime YOLO — class filtering")
    parser.add_argument("--model",      default=str(ROOT / "models" / "yolo11n.pt"))
    parser.add_argument("--source",     default="0")
    parser.add_argument("--target",     default="person",
                        help="탐지할 클래스 이름 (예: cup, bottle, person, scissors)")
    parser.add_argument("--imgsz",      type=int,   default=320)
    parser.add_argument("--conf",       type=float, default=0.35)
    parser.add_argument("--width",      type=int,   default=640)
    parser.add_argument("--height",     type=int,   default=480)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def screen_position(cx: float, cy: float, width: int, height: int) -> str:
    """bbox 중심점을 3×3 화면 위치 문자열로 변환 (슬라이드 34)"""
    col = ("left"   if cx < width  / 3
           else "right"  if cx > width  * 2 / 3
           else "center")
    row = ("top"    if cy < height / 3
           else "bottom" if cy > height * 2 / 3
           else "middle")
    if col == "center" and row == "middle":
        return "center"
    return f"{col}-{row}"


def draw_target_boxes(frame, result, model_names: dict, target_name: str) -> list:
    """target_name 과 일치하는 박스만 그리고 탐지 정보 리스트 반환 (슬라이드 32)"""
    h, w  = frame.shape[:2]
    found = []

    for box in result.boxes:
        class_id   = int(box.cls[0])
        class_name = model_names[class_id]
        conf       = float(box.conf[0])

        if class_name != target_name:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx  = (x1 + x2) / 2
        cy  = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)

        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # 중심점 표시
        cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)

        # 슬라이드 35 · Detection JSON 구조
        found.append({
            "found":           True,
            "target":          class_name,
            "confidence":      round(conf, 3),
            "bbox":            [x1, y1, x2, y2],
            "center":          [int(cx), int(cy)],
            "area":            area,
            "screen_position": screen_position(cx, cy, w, h),
            "timestamp":       round(time.time(), 3),
        })

    return found


def main():
    args   = parse_args()
    source = parse_source(args.source)

    print(f"loading model: {args.model}")
    model = YOLO(args.model)

    # 클래스 이름 존재 여부 확인
    name_to_id = {name: idx for idx, name in model.names.items()}
    if args.target not in name_to_id:
        print(f"[경고] '{args.target}' 클래스가 모델에 없습니다.")
        print(f"       사용 가능한 클래스(일부): {list(model.names.values())[:10]} ...")
    else:
        print(f"타겟 클래스: '{args.target}' (id={name_to_id[args.target]})")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    save_dir = ROOT / "outputs" / "demo_frames"
    if args.no_display:
        save_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    prev      = time.perf_counter()

    print(f"실행 중 (target='{args.target}') — 종료하려면 q 를 누르세요.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame read failed")
            break

        results    = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        detections = draw_target_boxes(frame, results[0], model.names, args.target)

        # FPS 표시
        now  = time.perf_counter()
        fps  = 1.0 / (now - prev)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}  target: {args.target}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # JSON 출력 — confidence 가장 높은 것만 출력
        if detections:
            best = max(detections, key=lambda d: d["confidence"])
            print(json.dumps(best, ensure_ascii=False))
        elif frame_idx % 30 == 0:
            print(json.dumps({"found": False, "target": args.target,
                              "timestamp": round(time.time(), 3)},
                             ensure_ascii=False))

        if args.no_display:
            if frame_idx % 30 == 0:
                cv2.imwrite(str(save_dir / f"frame_{frame_idx:05d}.jpg"), frame)
        else:
            cv2.imshow(f"YOLO filtered — {args.target}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
