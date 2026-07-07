# day4_08_ollama_intent_json.py
# 슬라이드 80~84 · 미니 프로젝트 핵심
# Ollama structured output + Pydantic 검증 + rule-based fallback
#
# 실행:
#   # 대화형 (단일 입력)
#   python src/day4_08_ollama_intent_json.py
#   python src/day4_08_ollama_intent_json.py --model qwen2.5:0.5b
#
#   # CSV 일괄 테스트 (20개 명령)
#   python src/day4_08_ollama_intent_json.py \
#     --csv data/commands_20.csv \
#     --out outputs/intent_test_results.jsonl
#
# 3중 안전장치 (슬라이드 80):
#   1. intent_schema.json → Ollama format 필드로 JSON 출력 강제
#   2. Pydantic 검증     → 타입·필수 필드 검증
#   3. alias guardrail   → target_en 이 ko_to_class.json 에 없으면 unknown
#
# Day5 에서 intent_parser_ollama.py 로 파일명 변경 후 재사용됩니다.

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Literal

import requests
from pydantic import BaseModel, Field, ValidationError

ROOT            = Path(__file__).parent.parent
OLLAMA_GEN_URL  = "http://localhost:11434/api/generate"
TIMEOUT         = 120


# ── Pydantic 모델 (슬라이드 80) ─────────────────────────────────────────────

class CommandIntent(BaseModel):
    action:         Literal["find_object", "list_objects", "unknown"]
    target_ko:      str | None = None
    target_en:      str | None = None
    attributes:     list[str]  = Field(default_factory=list)
    min_confidence: float      = Field(default=0.35, ge=0.0, le=1.0)


# ── 파일 로드 ────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | list:
    if not path.exists():
        print(f"[오류] 파일 없음: {path.resolve()}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_text(path: Path) -> str:
    if not path.exists():
        print(f"[오류] 파일 없음: {path.resolve()}")
        sys.exit(1)
    return path.read_text(encoding="utf-8")


# ── 프롬프트 빌드 ────────────────────────────────────────────────────────────

def build_prompt(command: str, ko_to_class: dict, prompt_template: str) -> str:
    allowed = sorted(set(ko_to_class.values()))
    return prompt_template.format(
        allowed_classes=", ".join(allowed),
        user_command=command,
    )


# ── Ollama 호출 ──────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str, schema: dict) -> str:
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "format":  schema,                       # JSON schema 강제
        "options": {"temperature": 0},
    }
    r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("response", "")


# ── fallback: rule-based parser ──────────────────────────────────────────────

def rule_based_fallback(command: str, ko_to_class: dict) -> CommandIntent:
    """SLM 실패 시 규칙 기반으로 대체 (day4_02 단순화 버전)"""
    LIST_KW = ["목록", "뭐가", "무엇이", "다 말해", "전부", "모두"]
    FIND_KW = ["찾아", "어디", "있어", "보여", "알려"]

    for kw in LIST_KW:
        if kw in command:
            return CommandIntent(action="list_objects")

    for ko, en in sorted(ko_to_class.items(), key=lambda x: -len(x[0])):
        if ko in command:
            has_find = any(kw in command for kw in FIND_KW)
            if has_find or ko:
                return CommandIntent(action="find_object",
                                     target_ko=ko, target_en=en)

    return CommandIntent(action="unknown")


# ── 핵심 파싱 함수 ───────────────────────────────────────────────────────────

def parse_intent(command: str, model: str,
                 ko_to_class: dict, schema: dict,
                 prompt_template: str) -> tuple[CommandIntent, str]:
    """
    Returns (intent, source)
    source: 'ollama' | 'fallback_parse_error' | 'fallback_validation_error'
             | 'fallback_guardrail' | 'fallback_connection_error'
    """
    allowed_en = set(ko_to_class.values())
    prompt     = build_prompt(command, ko_to_class, prompt_template)

    # 1단계: Ollama structured output 호출
    try:
        raw = call_ollama(prompt, model, schema)
    except requests.exceptions.ConnectionError:
        return rule_based_fallback(command, ko_to_class), "fallback_connection_error"
    except Exception:
        return rule_based_fallback(command, ko_to_class), "fallback_connection_error"

    # 2단계: JSON 파싱
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return rule_based_fallback(command, ko_to_class), "fallback_parse_error"

    # 3단계: Pydantic 검증
    try:
        intent = CommandIntent.model_validate(obj)
    except ValidationError:
        return rule_based_fallback(command, ko_to_class), "fallback_validation_error"

    # 4단계: alias guardrail — target_en 이 허용 목록에 없으면 unknown
    if intent.target_en and intent.target_en not in allowed_en:
        intent.action    = "unknown"
        intent.target_en = None
        intent.target_ko = None
        return intent, "fallback_guardrail"

    return intent, "ollama"


# ── 모드: 대화형 ─────────────────────────────────────────────────────────────

def interactive_mode(model: str, ko_to_class: dict,
                     schema: dict, prompt_template: str):
    print(f"intent parser 시작 (model={model})")
    print("종료: q / quit / exit")
    print("-" * 54)

    while True:
        try:
            command = input("명령 입력> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not command or command.lower() in {"q", "quit", "exit"}:
            break

        intent, source = parse_intent(command, model, ko_to_class,
                                       schema, prompt_template)
        result = {
            "text":   command,
            "intent": intent.model_dump(),
            "source": source,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()


# ── 모드: CSV 일괄 처리 ──────────────────────────────────────────────────────

def csv_batch_mode(csv_path: str, out_path: str, model: str,
                   ko_to_class: dict, schema: dict, prompt_template: str):
    in_file  = Path(csv_path)
    out_file = Path(out_path)

    if not in_file.exists():
        print(f"[오류] CSV 없음: {in_file.resolve()}")
        sys.exit(1)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    total   = success = ollama_cnt = fallback_cnt = 0

    with open(in_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            command         = row.get("text", "").strip()
            expected_action = row.get("expected_action", "").strip()
            if not command:
                continue

            intent, source = parse_intent(command, model, ko_to_class,
                                           schema, prompt_template)
            correct = (intent.action == expected_action) if expected_action else None

            record = {
                "text":            command,
                "intent":          intent.model_dump(),
                "source":          source,
                "expected_action": expected_action or None,
                "correct":         correct,
            }
            records.append(record)

            total += 1
            if source == "ollama":
                ollama_cnt += 1
            else:
                fallback_cnt += 1
            if correct:
                success += 1

            status = "✓" if correct else ("✗" if correct is False else "-")
            print(f"  {status} [{source:>30}] {command[:30]:<30} → {intent.action}")

    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*54}")
    print(f"총 명령       : {total}")
    print(f"Ollama 성공   : {ollama_cnt}")
    print(f"fallback 처리 : {fallback_cnt}")
    if any(r["correct"] is not None for r in records):
        evaluated = [r for r in records if r["correct"] is not None]
        acc = success / len(evaluated) * 100
        print(f"action 정확도 : {success}/{len(evaluated)} ({acc:.1f}%)")
    print(f"저장          : {out_file.resolve()}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Ollama intent JSON parser")
    parser.add_argument("--model",  default="qwen2.5:0.5b")
    parser.add_argument("--schema", default=str(ROOT / "data" / "intent_schema.json"),
                        help="intent_schema.json 경로")
    parser.add_argument("--prompt", default=str(ROOT / "data" / "intent_prompt_ko.txt"),
                        help="intent_prompt_ko.txt 경로")
    parser.add_argument("--alias",  default=str(ROOT / "data" / "ko_to_class.json"),
                        help="ko_to_class.json 경로")
    parser.add_argument("--csv",    default="",
                        help="일괄 처리할 CSV 파일 경로 (없으면 대화형)")
    parser.add_argument("--out",    default=str(ROOT / "outputs" / "intent_test_results.jsonl"),
                        help="CSV 일괄 처리 결과 JSONL 경로")
    return parser.parse_args()


def main():
    args = parse_args()

    ko_to_class     = load_json(Path(args.alias))
    schema          = load_json(Path(args.schema))
    prompt_template = load_text(Path(args.prompt))

    if args.csv:
        csv_batch_mode(args.csv, args.out, args.model,
                       ko_to_class, schema, prompt_template)
    else:
        interactive_mode(args.model, ko_to_class, schema, prompt_template)


if __name__ == "__main__":
    main()
