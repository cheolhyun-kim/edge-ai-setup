# day4_03_generate_once.py
# 슬라이드 62 · Lab C: Python API 호출 구조 — /api/generate 단발 호출
#
# 실행:
#   python src/day4_03_generate_once.py
#   python src/day4_03_generate_once.py --model qwen2.5:0.5b
#   python src/day4_03_generate_once.py --prompt "가위 어디 있어?"
#
# 목적:
#   - requests 로 Ollama /api/generate 엔드포인트 호출
#   - stream=False 로 응답 전체를 한 번에 받아 파싱
#   - 모델 응답 내용·형식이 모델마다 크게 다름을 체감
#
# 사전 준비:
#   ollama serve  (또는 systemctl start ollama)
#   ollama pull qwen2.5:0.5b

import argparse
import json

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT    = 120   # seconds


def parse_args():
    parser = argparse.ArgumentParser(description="Ollama /api/generate 단발 호출")
    parser.add_argument("--model",  default="qwen2.5:0.5b")
    parser.add_argument("--prompt", default="화면에서 컵 찾아줘. target 만 영어로 답해.")
    return parser.parse_args()


def call_generate(model: str, prompt: str) -> dict:
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,        # 응답 전체를 한 번에 받음
    }

    print(f"model  : {model}")
    print(f"prompt : {prompt}")
    print("-" * 50)

    r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def main():
    args = parse_args()

    try:
        data = call_generate(args.model, args.prompt)
    except requests.exceptions.ConnectionError:
        print("[오류] Ollama 서버에 연결할 수 없습니다.")
        print("  → ollama serve 또는 systemctl start ollama 실행 후 재시도")
        return
    except requests.exceptions.HTTPError as e:
        print(f"[오류] HTTP {e.response.status_code}: {e.response.text}")
        return

    # 응답 출력
    response_text = data.get("response", "")
    print(f"응답:\n{response_text}")
    print()

    # 벤치마크 메트릭 출력 (슬라이드 64~65)
    eval_count    = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 1)          # nanoseconds
    total_dur     = data.get("total_duration", 0) / 1e9   # → seconds
    tps           = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0

    print("--- 성능 메트릭 ---")
    print(f"  eval_count    : {eval_count} tokens")
    print(f"  total_duration: {total_dur:.2f} s")
    print(f"  tokens/sec    : {tps:.1f}")
    print()
    print("※ 응답 내용·형식은 모델마다 크게 달라집니다.")
    print("  JSON 출력을 강제하려면 day4_08_ollama_intent_json.py 를 사용하세요.")


if __name__ == "__main__":
    main()
