# day4_07_make_model_report.py
# 슬라이드 74 · 벤치마크 CSV → 모델 비교 리포트 MD 생성
#
# 실행:
#   python src/day4_07_make_model_report.py
#   python src/day4_07_make_model_report.py \
#     --csv outputs/model_benchmark.csv \
#     --out outputs/model_compare_report.md
#
# 출력: outputs/model_compare_report.md
#   - 모델별 평균 latency / tokens/sec 비교표
#   - 슬라이드 68 벤치마크 결과표 템플릿 기준

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="벤치마크 CSV → 비교 리포트 생성")
    parser.add_argument("--csv", default=str(ROOT / "outputs" / "model_benchmark.csv"),
                        help="day4_05_benchmark_models.py 출력 CSV")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "model_compare_report.md"),
                        help="생성할 MD 파일 경로")
    return parser.parse_args()


def load_csv(csv_path: str) -> list[dict]:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV 파일 없음: {p.resolve()}\n"
                                f"  먼저 day4_05_benchmark_models.py 를 실행하세요.")
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict]) -> dict:
    """모델별 평균 latency / tokens_per_sec 집계"""
    data = defaultdict(lambda: {"latencies": [], "tps": [], "responses": []})
    for r in rows:
        model = r["model"]
        try:
            data[model]["latencies"].append(float(r["latency_s"]))
            data[model]["tps"].append(float(r["tokens_per_sec"]))
        except (ValueError, KeyError):
            pass
        data[model]["responses"].append(r.get("response_preview", ""))
    return data


def build_report(data: dict, csv_path: str) -> str:
    lines = []
    lines.append("# Day 4 · 모델 비교 리포트")
    lines.append("")
    lines.append(f"> 출처: `{csv_path}`")
    lines.append("")

    # 요약 비교표 (슬라이드 68)
    lines.append("## 모델별 평균 성능")
    lines.append("")
    lines.append("| 모델 | 평균 latency (s) | 평균 tokens/sec | 샘플 수 |")
    lines.append("|------|-----------------|----------------|--------|")

    ranked = []
    for model, vals in data.items():
        lats = vals["latencies"]
        tps  = vals["tps"]
        if not lats:
            continue
        avg_lat = sum(lats) / len(lats)
        avg_tps = sum(tps)  / len(tps) if tps else 0
        ranked.append((model, avg_lat, avg_tps, len(lats)))

    # tokens/sec 내림차순 정렬
    ranked.sort(key=lambda x: -x[2])

    for model, avg_lat, avg_tps, n in ranked:
        lines.append(f"| `{model}` | {avg_lat:.2f} | {avg_tps:.1f} | {n} |")

    lines.append("")

    # 모델별 응답 샘플
    lines.append("## 모델별 응답 샘플")
    lines.append("")
    for model, vals in data.items():
        lines.append(f"### {model}")
        unique = list(dict.fromkeys(vals["responses"]))[:5]
        for r in unique:
            if r:
                lines.append(f"- `{r}`")
        lines.append("")

    # 선택 가이드
    lines.append("## 모델 선택 기준 (슬라이드 53)")
    lines.append("")
    lines.append("| 기준 | 설명 |")
    lines.append("|------|------|")
    lines.append("| 속도 | tokens/sec 가 높을수록 Pi5 에서 응답이 빠름 |")
    lines.append("| 한국어 | 한국어 명령에서 target 을 정확히 뽑는지 확인 |")
    lines.append("| JSON 안정성 | 정해진 schema 로 응답하는지 확인 |")
    lines.append("| 장비 안정성 | 온도·RAM 기록 필요 |")
    lines.append("")
    lines.append("> **권장**: 속도가 가장 빠른 모델로 먼저 성공하고, "
                 "품질이 부족하면 다음 크기로 올린다.")
    lines.append("")
    lines.append("---")
    lines.append("*생성: day4_07_make_model_report.py*")

    return "\n".join(lines)


def main():
    args = parse_args()

    print(f"CSV 읽기: {args.csv}")
    try:
        rows = load_csv(args.csv)
    except FileNotFoundError as e:
        print(f"[오류] {e}")
        return

    data   = aggregate(rows)
    report = build_report(data, args.csv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"리포트 생성 완료: {out_path.resolve()}")
    print()
    # 콘솔 미리보기
    for line in report.split("\n")[:30]:
        print(line)
    if report.count("\n") > 30:
        print("... (이하 생략)")


if __name__ == "__main__":
    main()
