# day5_02_detection_json.py
# 슬라이드 36~38 · YOLO result.boxes → Detection JSON 변환
#
# 실행 위치: edge-ai-day2/src/
#
# 모듈로 import:
#   from day5_02_detection_json import boxes_to_detections, map_position, make_mock_detections

import time
from pathlib import Path

ROOT = Path(__file__).parent.parent


# ── 위치 매핑 (슬라이드 38) ───────────────────────────────────────────────────

def map_position(cx: float, cy: float, width: int, height: int) -> str:
    """
    bbox 중심점을 3×3 화면 위치 문자열로 변환
    반환값: 'left-top' | 'center' | 'right-bottom' 등
    """
    col = ("left"   if cx < width  / 3
           else "right"  if cx > width  * 2 / 3
           else "center")
    row = ("top"    if cy < height / 3
           else "bottom" if cy > height * 2 / 3
           else "middle")
    if col == "center" and row == "middle":
        return "center"
    return f"{col}-{row}"


# ── 위치 → 한국어 변환 ────────────────────────────────────────────────────────

POSITION_KO: dict[str, str] = {
    "left-top":      "왼쪽 위",
    "center-top":    "가운데 위",
    "right-top":     "오른쪽 위",
    "left-middle":   "왼쪽 가운데",
    "center":        "화면 중앙",
    "right-middle":  "오른쪽 가운데",
    "left-bottom":   "왼쪽 아래",
    "center-bottom": "가운데 아래",
    "right-bottom":  "오른쪽 아래",
}


def position_ko(pos: str) -> str:
    return POSITION_KO.get(pos, pos)


# ── YOLO boxes → Detection JSON 리스트 (슬라이드 37) ─────────────────────────

def boxes_to_detections(result, model_names: dict) -> list[dict]:
    """
    ultralytics result.boxes → Detection JSON 리스트
    result: model.predict() 반환값의 results[0]
    """
    h, w   = result.orig_shape[:2]
    output = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf            = round(float(box.conf[0]), 3)
        cls_id          = int(box.cls[0])
        class_name      = model_names[cls_id]

        cx   = (x1 + x2) / 2
        cy   = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        pos  = map_position(cx, cy, w, h)

        output.append({
            "class_name":  class_name,
            "confidence":  conf,
            "bbox_xyxy":   [x1, y1, x2, y2],
            "center":      [int(cx), int(cy)],
            "area":        area,
            "position":    pos,
            "position_ko": position_ko(pos),
            "timestamp":   round(time.time(), 3),
        })

    output.sort(key=lambda d: -d["confidence"])
    return output


# ── mock detection 생성 (슬라이드 42) ────────────────────────────────────────

def make_mock_detections() -> list[dict]:
    """카메라·YOLO 없이 파이프라인 검증용 더미 detection 목록"""
    now = round(time.time(), 3)
    return [
        {"class_name": "cup",      "confidence": 0.82,
         "bbox_xyxy": [120, 220, 260, 390], "center": [190, 305],
         "area": 23800, "position": "left-bottom",  "position_ko": "왼쪽 아래",   "timestamp": now},
        {"class_name": "person",   "confidence": 0.91,
         "bbox_xyxy": [300,  80, 520, 460], "center": [410, 270],
         "area": 96800, "position": "center",        "position_ko": "화면 중앙",   "timestamp": now},
        {"class_name": "scissors", "confidence": 0.74,
         "bbox_xyxy": [540, 300, 620, 420], "center": [580, 360],
         "area":  9600, "position": "right-bottom", "position_ko": "오른쪽 아래",  "timestamp": now},
    ]
