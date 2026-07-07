# day5_04_final_demo.py
# 슬라이드 40~43 · 최종 통합 데모
# 자연어 명령 → SLM intent → YOLO 탐지 → 응답 문장
#
# 실행 위치: edge-ai-day2/src/
#
# 실행:
#   # mock mode — 카메라 없이 파이프라인 먼저 검증 (슬라이드 42)
#   python src/day5_04_final_demo.py --mock
#
#   # live mode — 웹캠 실시간 탐지
#   python src/day5_04_final_demo.py \
#     --model qwen2.5:0.5b \
#     --yolo ../models/best.pt \
#     --camera 0 \
#     --imgsz 320 \
#     --conf 0.35
#
# 종료: q 입력
#
# 전략 (슬라이드 41):
#   - YOLO 는 매 프레임 돌리고 결과를 cache
#   - SLM 은 명령 입력 시에만 1회 호출
#   - mock → live 순서로 검증

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent   # edge-ai-day2/

# 같은 src/ 폴더의 모듈 import
sys.path.insert(0, str(Path(__file__).parent))

from camera_utils                 import find_camera
from day5_01_intent_parser_ollama import parse_intent, CommandIntent
from day5_02_detection_json       import boxes_to_detections, make_mock_detections
from day5_03_find_object_tool     import (find_object, find_largest,
                                          list_visible_objects, build_response)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SLM × YOLO 최종 통합 데모")
    parser.add_argument("--mock",   action="store_true",
                        help="카메라 없이 mock detection 으로 실행")
    parser.add_argument("--model",  default="qwen2.5:0.5b")
    parser.add_argument("--yolo",   default=str(ROOT / "models" / "best.pt"),
                        help="YOLO 모델 경로 (없으면 yolo11n.pt 로 fallback)")
    parser.add_argument("--camera", type=int,   default=find_camera())
    parser.add_argument("--imgsz",  type=int,   default=320)
    parser.add_argument("--conf",   type=float, default=0.35)
    parser.add_argument("--log",    default=str(ROOT / "outputs" / "demo_log.jsonl"))
    return parser.parse_args()


# ── YOLO 로드 ────────────────────────────────────────────────────────────────

def load_yolo_model(yolo_path: str):
    from ultralytics import YOLO
    p = Path(yolo_path)
    if not p.exists():
        fallback = ROOT / "models" / "yolo11n.pt"
        print(f"[경고] {yolo_path} 없음 → {fallback} 로 대체")
        p = fallback
    print(f"YOLO 로드: {p}")
    return YOLO(str(p))


# ── YOLO 캐시 스레드 (live mode) ─────────────────────────────────────────────

class YoloCache:
    """별도 스레드에서 YOLO 를 지속 실행하고 최신 detection 을 보관"""
    def __init__(self, model, camera: int, imgsz: int, conf: float):
        self.model      = model
        self.camera     = camera
        self.imgsz      = imgsz
        self.conf       = conf
        self.detections: list[dict] = []
        self._lock      = threading.Lock()
        self._running   = True

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import cv2
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            print(f"[오류] Camera open failed: {self.camera}")
            self._running = False
            return
        while self._running:
            ok, frame = cap.read()
            if not ok:
                break
            results = self.model.predict(frame, imgsz=self.imgsz,
                                          conf=self.conf, verbose=False)
            dets = boxes_to_detections(results[0], self.model.names)
            with self._lock:
                self.detections = dets
        cap.release()

    def get(self) -> list[dict]:
        with self._lock:
            return list(self.detections)

    def stop(self):
        self._running = False


# ── 로그 저장 ────────────────────────────────────────────────────────────────

def write_log(log_path: str, record: dict):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── 파이프라인 실행 (슬라이드 40) ────────────────────────────────────────────

def run_pipeline(command: str, detections: list[dict],
                 model: str, log_path: str) -> str:
    # 1. SLM 호출 — 명령 입력 시 1회만
    t0 = time.perf_counter()
    intent, source = parse_intent(command, model)
    slm_ms = (time.perf_counter() - t0) * 1000

    print(f"  [SLM] {source}  {slm_ms:.0f}ms  → {intent.action} / {intent.target_en}")

    # 2. 도구 실행 (action 기반 if/else)
    if intent.action == "find_object" and intent.target_en:
        result = find_object(intent.target_en, detections, intent.min_confidence)
    elif intent.action == "list_objects":
        result = list_visible_objects(detections, intent.min_confidence)
    elif intent.action == "describe_largest":
        result = find_largest(detections, intent.min_confidence)
    else:
        result = {}

    # 3. 응답 생성 — 템플릿 기반
    answer = build_response(intent, result)

    write_log(log_path, {
        "command":   command,
        "intent":    intent.model_dump(),
        "source":    source,
        "slm_ms":    round(slm_ms, 1),
        "result":    result,
        "answer":    answer,
        "timestamp": round(time.time(), 3),
    })
    return answer


# ── mock mode ─────────────────────────────────────────────────────────────────

def run_mock(model: str, log_path: str):
    print("=" * 54)
    print("  MOCK MODE — 카메라 없이 파이프라인 검증")
    print("=" * 54)
    detections = make_mock_detections()
    print(f"mock detection {len(detections)}개: "
          f"{[d['class_name'] for d in detections]}")
    print("명령 입력 (q 로 종료)\n" + "-" * 54)

    while True:
        try:
            cmd = input("명령 입력> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd or cmd.lower() in {"q", "quit", "exit"}:
            break
        answer = run_pipeline(cmd, detections, model, log_path)
        print(f"  응답: {answer}\n")

    print(f"\n로그: {log_path}")


# ── live mode ─────────────────────────────────────────────────────────────────

def run_live(args):
    print("=" * 54)
    print("  LIVE MODE — 웹캠 실시간 탐지")
    print("=" * 54)
    yolo_model = load_yolo_model(args.yolo)
    cache      = YoloCache(yolo_model, args.camera, args.imgsz, args.conf)
    cache.start()

    print("YOLO 카메라 시작 중 ...")
    time.sleep(2)
    print("명령 입력 (q 로 종료)\n" + "-" * 54)

    try:
        while True:
            try:
                cmd = input("명령 입력> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not cmd or cmd.lower() in {"q", "quit", "exit"}:
                break
            detections = cache.get()
            print(f"  현재 탐지: {[d['class_name'] for d in detections]}")
            answer = run_pipeline(cmd, detections, args.model, args.log)
            print(f"  응답: {answer}\n")
    finally:
        cache.stop()

    print(f"\n로그: {args.log}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.mock:
        run_mock(args.model, args.log)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
