# day5_05_make_final_report.py
# 슬라이드 49 · demo_log.jsonl → final_report.md 자동 생성
#
# 실행 위치: edge-ai-day2/src/
#
# 실행:
#   python src/day5_05_make_final_report.py
#   python src/day5_05_make_final_report.py \
#     --log ../outputs/demo_log.jsonl \
#     --out ../outputs/final_report.md

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent   # edge-ai-day2/


def parse_args():
    parser = argparse.ArgumentParser(description="demo_log → final_report.md")
    parser.add_argument("--log",  default=str(ROOT / "outputs" / "demo_log.jsonl"))
    parser.add_argument("--out",  default=str(ROOT / "outputs" / "final_report.md"))
    parser.add_argument("--slm",  default="qwen2.5:0.5b")
    parser.add_argument("--yolo", default="best.pt")
    return parser.parse_args()


def load_log(log_path: str) -> list[dict]:
    p = Path(log_path)
    if not p.exists():
        print(f"[경고] 로그 파일 없음: {p.resolve()}")
        return []
    records = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def analyze(records: list[dict]) -> dict:
    total        = len(records)
    action_cnt   = Counter(r.get("intent", {}).get("action", "unknown") for r in records)
    source_cnt   = Counter(r.get("source", "unknown") for r in records)
    slm_cnt      = source_cnt.get("ollama", 0)
    fallback_cnt = total - slm_cnt
    slm_times    = [r["slm_ms"] for r in records if "slm_ms" in r]
    avg_slm      = sum(slm_times) / len(slm_times) if slm_times else 0

    success = 0
    for r in records:
        action = r.get("intent", {}).get("action", "")
        result = r.get("result", {})
        if action == "find_object" and isinstance(result, dict) and result.get("found"):
            success += 1
        elif action in ("list_objects", "describe_largest") and result:
            success += 1

    return {
        "total":        total,
        "action_cnt":   dict(action_cnt),
        "slm_cnt":      slm_cnt,
        "fallback_cnt": fallback_cnt,
        "avg_slm_ms":   round(avg_slm, 1),
        "success":      success,
        "success_rate": success / total * 100 if total else 0,
    }


def build_report(records: list[dict], stats: dict,
                 slm_model: str, yolo_model: str) -> str:
    lines = []

    lines += [
        "# Final Report",
        "",
        "## 1. 프로젝트 목표",
        "",
        "자연어 명령으로 화면 속 물체를 찾는 온디바이스 AI.",
        "Raspberry Pi 5 에서 SLM(Ollama) 과 YOLO11n 을 결합해",
        "'컵 찾아줘' → 탐지 → '컵이 화면 왼쪽 아래에 있습니다.' 흐름을 완성한다.",
        "",
        "## 2. 시스템 구조",
        "",
        "```",
        "사용자 한국어 명령",
        "  → SLM (Ollama)  : CommandIntent JSON 생성",
        "  → Router         : action 에 따라 도구 선택",
        "  → YOLO Tool      : 카메라 프레임에서 탐지",
        "  → Response       : 위치·신뢰도 자연어 응답",
        "```",
        "",
        "역할 분리: SLM = 명령 해석 / YOLO = 시각 인식 / Python = 제어·검증",
        "",
        "## 3. 사용 모델",
        "",
        "| 구분 | 모델 |",
        "|------|------|",
        f"| SLM  | `{slm_model}` |",
        f"| YOLO | `{yolo_model}` |",
        "",
        "## 4. 테스트 결과",
        "",
        "| 항목 | 값 |",
        "|------|---|",
        f"| 총 명령 수      | {stats['total']} |",
        f"| SLM 성공        | {stats['slm_cnt']} |",
        f"| fallback 처리   | {stats['fallback_cnt']} |",
        f"| 평균 SLM 응답   | {stats['avg_slm_ms']} ms |",
        f"| 응답 성공률     | {stats['success']}/{stats['total']} ({stats['success_rate']:.1f}%) |",
        "",
        "### action 분포",
        "",
        "| action | 횟수 |",
        "|--------|------|",
    ]
    for action, cnt in sorted(stats["action_cnt"].items()):
        lines.append(f"| {action} | {cnt} |")
    lines.append("")

    if records:
        lines += [
            "### 명령별 상세 (최대 20개)",
            "",
            "| # | 명령 | action | target | 응답 |",
            "|---|------|--------|--------|------|",
        ]
        for i, r in enumerate(records[:20], 1):
            cmd    = r.get("command", "")[:20]
            intent = r.get("intent", {})
            action = intent.get("action", "-")
            target = intent.get("target_en") or "-"
            answer = r.get("answer", "")[:30]
            lines.append(f"| {i} | {cmd} | {action} | {target} | {answer} |")
        lines.append("")

    lines += [
        "## 5. 개선 방향",
        "",
        "- 더 많은 커스텀 데이터 수집 → YOLO 커스텀 모델 정확도 향상",
        "- NCNN export 적용 → Pi5 FPS 개선",
        "- 프롬프트 few-shot 예시 보강 → SLM JSON 성공률 향상",
        "- UI 추가 (웹 스트리밍 또는 터미널 대시보드)",
        "",
        "---",
        "*생성: day5_05_make_final_report.py — outputs/final_report.md*",
    ]

    return "\n".join(lines)


def main():
    args    = parse_args()
    records = load_log(args.log)
    stats   = analyze(records)
    report  = build_report(records, stats, args.slm, args.yolo)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"리포트 생성: {out_path.resolve()}")
    print(f"  총 명령: {stats['total']}  성공률: {stats['success_rate']:.1f}%")
    print()
    for line in report.split("\n")[:25]:
        print(line)
    if report.count("\n") > 25:
        print("... (이하 생략)")


if __name__ == "__main__":
    main()
