# day5_01_intent_parser_ollama.py
# 슬라이드 25~27 · SLM 의도 추출 모듈
# day4_08_ollama_intent_json.py 를 Day5 구조에 맞게 재사용
#
# 실행 위치: edge-ai-day2/src/
# 설정 파일: ../data/intent_schema.json
#            ../data/class_to_ko.json
# 프롬프트:  ../data/intent_system_prompt_ko.txt
#
# 모듈로 import:
#   from day5_01_intent_parser_ollama import parse_intent, CommandIntent
#
# 단독 실행 (대화형 테스트):
#   python src/day5_01_intent_parser_ollama.py
#   python src/day5_01_intent_parser_ollama.py --model qwen2.5:0.5b

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import requests
from pydantic import BaseModel, Field, ValidationError

# edge-ai-day2/ 루트
ROOT           = Path(__file__).parent.parent
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
TIMEOUT        = 120

# ── 설정 파일 경로 (edge-ai-day2/data/) ──────────────────────────────────────
SCHEMA_PATH = ROOT / "data" / "intent_schema.json"
ALIAS_PATH  = ROOT / "data" / "class_to_ko.json"
PROMPT_PATH = ROOT / "data" / "intent_system_prompt_ko.txt"


# ── Pydantic 모델 (슬라이드 21) ──────────────────────────────────────────────

class CommandIntent(BaseModel):
    action:         Literal["find_object", "list_objects", "describe_largest", "unknown"]
    target_ko:      str | None = None
    target_en:      str | None = None
    attributes:     list[str]  = Field(default_factory=list)
    min_confidence: float      = Field(default=0.35, ge=0.0, le=1.0)
    raw_command:    str | None = None


# ── 파일 로드 ────────────────────────────────────────────────────────────────

def _load_configs() -> tuple[dict, dict, str]:
    for p in [SCHEMA_PATH, ALIAS_PATH, PROMPT_PATH]:
        if not p.exists():
            print(f"[오류] 파일 없음: {p.resolve()}")
            sys.exit(1)
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    with open(ALIAS_PATH, encoding="utf-8") as f:
        class_to_ko: dict = json.load(f)
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    return schema, class_to_ko, prompt_template


def _allowed_classes(class_to_ko: dict) -> set[str]:
    return set(class_to_ko.keys())


# ── 프롬프트 빌드 ────────────────────────────────────────────────────────────

def _build_prompt(command: str, class_to_ko: dict, template: str) -> str:
    allowed = ", ".join(sorted(class_to_ko.keys()))
    return template.format(allowed_classes=allowed, user_command=command)


# ── Ollama 호출 ──────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, model: str, schema: dict) -> str:
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "format":  schema,
        "options": {"temperature": 0},
    }
    r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("response", "")


# ── fallback: rule-based ──────────────────────────────────────────────────────

def _rule_fallback(command: str, class_to_ko: dict) -> CommandIntent:
    LIST_KW  = ["목록", "뭐가", "무엇이", "다 말해", "전부", "모두"]
    FIND_KW  = ["찾아", "어디", "있어", "보여", "알려"]
    LARGE_KW = ["큰", "가장", "제일"]

    for kw in LIST_KW:
        if kw in command:
            return CommandIntent(action="list_objects", raw_command=command)
    for kw in LARGE_KW:
        if kw in command:
            return CommandIntent(action="describe_largest", raw_command=command)
    for en, ko_list in class_to_ko.items():
        for ko in ko_list:
            if ko in command:
                has_find = any(kw in command for kw in FIND_KW)
                if has_find or ko:
                    return CommandIntent(action="find_object",
                                         target_ko=ko, target_en=en,
                                         raw_command=command)
    return CommandIntent(action="unknown", raw_command=command)


# ── 핵심 공개 함수 ────────────────────────────────────────────────────────────

def parse_intent(command: str,
                 model: str = "qwen2.5:0.5b") -> tuple[CommandIntent, str]:
    """
    자연어 명령 → CommandIntent
    Returns (intent, source)
    source: 'ollama' | 'fallback_*'
    """
    schema, class_to_ko, template = _load_configs()
    allowed = _allowed_classes(class_to_ko)
    prompt  = _build_prompt(command, class_to_ko, template)

    try:
        raw = _call_ollama(prompt, model, schema)
    except Exception:
        return _rule_fallback(command, class_to_ko), "fallback_connection_error"

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return _rule_fallback(command, class_to_ko), "fallback_parse_error"

    try:
        intent = CommandIntent.model_validate(obj)
    except ValidationError:
        return _rule_fallback(command, class_to_ko), "fallback_validation_error"

    # alias guardrail (슬라이드 26~27)
    if intent.target_en and intent.target_en not in allowed:
        intent.action    = "unknown"
        intent.target_en = None
        intent.target_ko = None
        return intent, "fallback_guardrail"

    intent.raw_command = command
    return intent, "ollama"


# ── 단독 실행 ────────────────────────────────────────────────────────────────

def _interactive(model: str):
    print(f"intent parser 테스트 (model={model}) — q 로 종료")
    print("-" * 54)
    while True:
        try:
            cmd = input("명령> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd or cmd.lower() in {"q", "quit", "exit"}:
            break
        intent, source = parse_intent(cmd, model)
        print(json.dumps({"intent": intent.model_dump(), "source": source},
                         ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:0.5b")
    args = parser.parse_args()
    _interactive(args.model)
