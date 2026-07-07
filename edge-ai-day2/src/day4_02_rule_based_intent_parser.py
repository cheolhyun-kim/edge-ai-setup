# day4_02_rule_based_intent_parser.py
# 슬라이드 28 Lab B · 34 · 규칙 기반 intent parser
#
# 실행:
#   # 단일 입력 (대화형)
#   python src/day4_02_rule_based_intent_parser.py
#
#   # CSV 일괄 처리
#   python src/day4_02_rule_based_intent_parser.py \
#     --csv data/commands_20.csv \
#     --out outputs/rule_parser_results.jsonl
#
# 목적:
#   - SLM 없이 alias 사전만으로 CommandIntent JSON 생성
#   - SLM 적용 전 baseline — 이후 SLM 결과와 같은 schema 로 비교
#   - find_object / list_objects / unknown 3가지 action 분류
#
# 한계 (슬라이드 32 데모):
#   "화면에 뭐가 보여?" → "보" 가 alias 에 걸려 paper 로 오탐 가능
#   → SLM + 검증 로직의 필요성

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ko_to_class.json 로드
ALIAS_PATH = ROOT / "data" / "ko_to_class.json"
if not ALIAS_PATH.exists():
    # src/ 옆 data/ 도 시도
    ALIAS_PATH = Path(__file__).parent.parent / "data" / "ko_to_class.json"

try:
    with open(ALIAS_PATH, encoding="utf-8") as f:
        KO_TO_CLASS: dict[str, str] = json.load(f)
except FileNotFoundError:
    print(f"[오류] ko_to_class.json 을 찾을 수 없습니다: {ALIAS_PATH}")
    sys.exit(1)

# list_objects 트리거 키워드
LIST_KEYWORDS = ["목록", "뭐가", "무엇이", "다 말해", "전부", "모두", "보여줘", "보이는"]

# find_object 트리거 키워드
FIND_KEYWORDS = ["찾아", "어디", "있어", "보여", "알려"]

ALLOWED_ACTIONS    = ["find_object", "list_objects", "unknown"]
DEFAULT_CONFIDENCE = 0.35


def build_intent(action: str, target_ko: str | None, target_en: str | None) -> dict:
    return {
        "action":         action,
        "target_ko":      target_ko,
        "target_en":      target_en,
        "attributes":     [],
        "min_confidence": DEFAULT_CONFIDENCE,
    }


def parse_intent(text: str) -> dict:
    """
    규칙 기반으로 CommandIntent JSON 생성.
    처리 순서:
      1. list_objects 키워드 먼저 확인 (find 보다 우선)
      2. alias 사전에서 target 탐색
      3. find 키워드 있으면 find_object
      4. 해당 없으면 unknown
    """
    # 1. list_objects 먼저
    for kw in LIST_KEYWORDS:
        if kw in text:
            return build_intent("list_objects", None, None)

    # 2. alias 사전에서 target 탐색 (긴 키워드 우선 매칭)
    matched_ko  = None
    matched_en  = None
    for ko, en in sorted(KO_TO_CLASS.items(), key=lambda x: -len(x[0])):
        if ko in text:
            matched_ko = ko
            matched_en = en
            break

    # 3. find 키워드 확인
    has_find = any(kw in text for kw in FIND_KEYWORDS)

    if matched_en and (has_find or matched_ko):
        return build_intent("find_object", matched_ko, matched_en)

    # 4. unknown
    return build_intent("unknown", None, None)


def interactive_mode():
    """단일 입력 대화형 모드"""
    print("규칙 기반 intent parser — q 로 종료")
    print(f"alias 사전: {len(KO_TO_CLASS)}개 항목")
    print("-" * 50)

    while True:
        try:
            text = input("명령 입력> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if text.lower() in {"q", "quit", "exit", ""}:
            break

        intent = parse_intent(text)
        print(json.dumps({"text": text, "intent": intent}, ensure_ascii=False, indent=2))
        print()


def csv_batch_mode(csv_path: str, out_path: str):
    """CSV 일괄 처리 → JSONL 저장 (슬라이드 34)"""
    csv_file = Path(csv_path)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        print(f"[오류] CSV 파일 없음: {csv_file.resolve()}")
        sys.exit(1)

    results    = []
    success    = 0
    total      = 0

    with open(csv_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text           = row.get("text", "").strip()
            expected_action = row.get("expected_action", "").strip()
            if not text:
                continue

            intent = parse_intent(text)
            record = {"text": text, "intent": intent}

            # expected_action 이 있으면 정답 비교
            if expected_action:
                record["expected_action"] = expected_action
                record["correct"]         = (intent["action"] == expected_action)
                if record["correct"]:
                    success += 1
            total += 1
            results.append(record)

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"처리 완료: {total}개 명령")
    if success > 0 or any("correct" in r for r in results):
        print(f"action 정확도: {success}/{total} ({success/total*100:.1f}%)")
    print(f"저장: {out_file.resolve()}")

    # 오탐 사례 출력
    errors = [r for r in results if "correct" in r and not r["correct"]]
    if errors:
        print(f"\n오탐 사례 ({len(errors)}개):")
        for e in errors:
            print(f"  입력: {e['text']}")
            print(f"  기대: {e['expected_action']}  실제: {e['intent']['action']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Rule-based intent parser")
    parser.add_argument("--csv", default="",
                        help="일괄 처리할 CSV 파일 경로 (없으면 대화형 모드)")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "rule_parser_results.jsonl"),
                        help="결과 JSONL 저장 경로")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.csv:
        csv_batch_mode(args.csv, args.out)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
