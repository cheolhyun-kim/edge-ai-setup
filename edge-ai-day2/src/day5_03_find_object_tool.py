# day5_03_find_object_tool.py
# 슬라이드 31, 39~40 · YOLO 탐지 결과 조회 도구 함수
#
# 실행 위치: edge-ai-day2/src/
#
# 모듈로 import:
#   from day5_03_find_object_tool import find_object, list_visible_objects, build_response

from day5_02_detection_json import position_ko


# ── find_object (슬라이드 31, 39) ────────────────────────────────────────────

def find_object(target: str,
                detections: list[dict],
                min_confidence: float = 0.35) -> dict:
    """
    Detection JSON 목록에서 target class 를 찾아 반환.
    여러 개 잡히면 confidence 가장 높은 것을 선택.

    Returns:
        {"found": True,  ...detection 필드...}   # 발견
        {"found": False, "target": target}        # 미발견
    """
    matches = [
        d for d in detections
        if d["class_name"] == target and d["confidence"] >= min_confidence
    ]
    if not matches:
        return {"found": False, "target": target}
    best = max(matches, key=lambda d: d["confidence"])
    return {"found": True, **best}


# ── list_visible_objects (슬라이드 40) ───────────────────────────────────────

def list_visible_objects(detections: list[dict],
                          min_confidence: float = 0.35) -> list[str]:
    """화면 속 물체 class 목록 반환 (중복 제거, confidence 기준 필터)"""
    seen    = set()
    classes = []
    for d in detections:
        if d["confidence"] >= min_confidence and d["class_name"] not in seen:
            seen.add(d["class_name"])
            classes.append(d["class_name"])
    return classes


# ── find_largest (슬라이드 39) ───────────────────────────────────────────────

def find_largest(detections: list[dict],
                 min_confidence: float = 0.35) -> dict:
    """면적이 가장 큰 물체 반환"""
    candidates = [d for d in detections if d["confidence"] >= min_confidence]
    if not candidates:
        return {"found": False}
    best = max(candidates, key=lambda d: d["area"])
    return {"found": True, **best}


# ── build_response (슬라이드 40, 45~46) ──────────────────────────────────────

def build_response(intent, result) -> str:
    """
    intent (CommandIntent) 와 도구 실행 결과로 자연어 응답 생성.
    템플릿 기반 — SLM 을 다시 호출하지 않음.
    """
    action = intent.action

    # find_object
    if action == "find_object":
        target_ko = intent.target_ko or intent.target_en or "물체"
        if isinstance(result, dict) and result.get("found"):
            pos_kr = result.get("position_ko",
                                position_ko(result.get("position", "")))
            conf   = result.get("confidence", 0)
            return (f"{target_ko}이(가) 화면 {pos_kr}에 있습니다. "
                    f"(신뢰도: {conf:.0%})")
        return f"화면에서 {target_ko}을(를) 찾지 못했습니다."

    # list_objects
    if action == "list_objects":
        if isinstance(result, list) and result:
            return f"화면에서 보이는 물체: {', '.join(result)}"
        return "화면에서 아무 물체도 탐지되지 않았습니다."

    # describe_largest
    if action == "describe_largest":
        if isinstance(result, dict) and result.get("found"):
            name   = result.get("class_name", "물체")
            pos_kr = result.get("position_ko", "")
            area   = result.get("area", 0)
            return f"가장 큰 물체는 '{name}' 입니다. 위치: {pos_kr}, 면적: {area}px²"
        return "화면에서 물체를 찾지 못했습니다."

    # unknown / fallback
    return "이 명령은 아직 처리할 수 없습니다."
