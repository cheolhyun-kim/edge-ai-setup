# day4_05_benchmark_models.py
# 슬라이드 65~68, 72 · Lab D: 모델 비교 벤치마크 → CSV 저장
#
# 실행:
#   python src/day4_05_benchmark_models.py
#   python src/day4_05_benchmark_models.py \
#     --models gemma3:270m qwen2.5:0.5b llama3.2:1b \
#     --prompts data/benchmark_prompts_ko.txt \
#     --out outputs/model_benchmark.csv \
#     --repeat 3
#
# 원칙 (슬라이드 66):
#   - 모델마다 같은 프롬프트 사용
#   - warm-up 1회 후 본 측정 (첫 실행은 모델 load 시간 포함)
#   - repeat 회 평균 기록
#   - 속도뿐 아니라 응답 내용도 함께 기록
#
# 출력: outputs/model_benchmark.csv
#   컬럼: model, prompt, run, latency_s, tokens_per_sec, response_preview

import argparse
import csv
import time
from pathlib import Path

import requests

ROOT            = Path(__file__).parent.parent
OLLAMA_GEN_URL  = "http://localhost:11434/api/generate"
TIMEOUT         = 180

DEFAULT_MODELS  = ["gemma3:270m", "qwen2.5:0.5b", "llama3.2:1b"]
DEFAULT_PROMPTS = [
    "화면에서 컵 찾아줘. 대상 물체만 영어 단어로 답해.",
    "가위 어디 있어? 대상 물체만 영어 단어로 답해.",
    "보이는 물체 목록을 알려줘.",
    "멋있는 거 찾아줘.",
    "사람이 보이면 알려줘. 대상 물체만 영어 단어로 답해.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Ollama 모델 벤치마크")
    parser.add_argument("--models",  nargs="+", default=DEFAULT_MODELS,
                        help="비교할 모델 목록")
    parser.add_argument("--prompts", default="",
                        help="프롬프트 파일 경로 (없으면 내장 기본값 사용)")
    parser.add_argument("--out",     default=str(ROOT / "outputs" / "model_benchmark.csv"),
                        help="결과 CSV 저장 경로")
    parser.add_argument("--repeat",  type=int, default=3,
                        help="모델·프롬프트 조합당 반복 횟수 (기본 3)")
    return parser.parse_args()


def load_prompts(prompts_path: str) -> list[str]:
    p = Path(prompts_path)
    if prompts_path and p.exists():
        lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
        print(f"프롬프트 파일 로드: {p} ({len(lines)}개)")
        return lines
    print("프롬프트 파일 없음 — 내장 기본값 사용")
    return DEFAULT_PROMPTS


def call_generate(model: str, prompt: str) -> dict:
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0}}
    r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def calc_tps(data: dict) -> float:
    ec  = data.get("eval_count",    0)
    ed  = data.get("eval_duration", 1)   # nanoseconds
    sec = ed / 1_000_000_000
    return ec / sec if sec > 0 else 0.0


def warmup(model: str, prompt: str):
    """슬라이드 65 — 첫 실행은 모델 load 시간 포함, warm-up 1회 후 측정"""
    print(f"  warm-up: {model} ...", end=" ", flush=True)
    try:
        call_generate(model, prompt)
        print("완료")
    except Exception as e:
        print(f"실패 ({e})")


def main():
    args    = parse_args()
    prompts = load_prompts(args.prompts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for model in args.models:
        print(f"\n{'='*54}")
        print(f"모델: {model}")
        print(f"{'='*54}")

        # warm-up — 첫 프롬프트로 1회
        try:
            warmup(model, prompts[0])
        except requests.exceptions.ConnectionError:
            print(f"  [오류] Ollama 서버 연결 실패 — {model} 건너뜀")
            continue

        for prompt in prompts:
            latencies = []
            tps_list  = []
            previews  = []

            for run in range(1, args.repeat + 1):
                try:
                    t0   = time.perf_counter()
                    data = call_generate(model, prompt)
                    lat  = time.perf_counter() - t0
                    tps  = calc_tps(data)
                    resp = data.get("response", "").strip().replace("\n", " ")[:60]

                    latencies.append(lat)
                    tps_list.append(tps)
                    previews.append(resp)

                    print(f"  [{run}/{args.repeat}] {lat:.2f}s  {tps:.1f} tok/s  "
                          f"→ {resp[:40]}")

                    rows.append({
                        "model":           model,
                        "prompt":          prompt[:50],
                        "run":             run,
                        "latency_s":       f"{lat:.3f}",
                        "tokens_per_sec":  f"{tps:.1f}",
                        "response_preview": resp,
                    })

                except Exception as e:
                    print(f"  [오류] run {run}: {e}")

            if latencies:
                avg_lat = sum(latencies) / len(latencies)
                avg_tps = sum(tps_list)  / len(tps_list)
                print(f"  평균: {avg_lat:.2f}s  {avg_tps:.1f} tok/s")

    # CSV 저장
    if rows:
        fieldnames = ["model", "prompt", "run", "latency_s",
                      "tokens_per_sec", "response_preview"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n저장 완료: {out_path.resolve()}")
        print(f"총 {len(rows)}행 기록")
    else:
        print("\n기록할 데이터가 없습니다.")

    print("\n다음 단계: day4_07_make_model_report.py 로 리포트를 생성하세요.")
    print(f"  python src/day4_07_make_model_report.py --csv {out_path}")


if __name__ == "__main__":
    main()
