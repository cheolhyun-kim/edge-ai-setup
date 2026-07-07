# day4_04_chat_api_once.py
# 슬라이드 63 · /api/chat 1회 호출 — generate 와 chat 구조 비교
#
# 실행:
#   python src/day4_04_chat_api_once.py
#   python src/day4_04_chat_api_once.py --model llama3.2:1b
#   python src/day4_04_chat_api_once.py --user "가위 어디 있어?"
#
# generate 와 chat 의 차이:
#   /api/generate → prompt 문자열 1개
#   /api/chat     → messages 배열 (system / user / assistant 역할 구분)
#
# chat 을 쓰는 이유:
#   - system prompt 로 역할 고정 가능
#   - 이전 대화 히스토리 유지 가능 → day4_06_console_chatbot.py 에서 활용
#
# 사전 준비:
#   ollama serve
#   ollama pull qwen2.5:0.5b

import argparse
import json

import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
TIMEOUT         = 120

SYSTEM_PROMPT = "너는 온디바이스 AI 명령 해석기야. 사용자 명령에서 대상 물체만 영어로 답해."


def parse_args():
    parser = argparse.ArgumentParser(description="Ollama /api/chat 1회 호출")
    parser.add_argument("--model",  default="qwen2.5:0.5b")
    parser.add_argument("--system", default=SYSTEM_PROMPT)
    parser.add_argument("--user",   default="화면에서 컵 찾아줘")
    return parser.parse_args()


def call_chat(model: str, system: str, user: str) -> dict:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
    }

    print(f"model   : {model}")
    print(f"system  : {system}")
    print(f"user    : {user}")
    print("-" * 50)

    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def main():
    args = parse_args()

    try:
        data = call_chat(args.model, args.system, args.user)
    except requests.exceptions.ConnectionError:
        print("[오류] Ollama 서버에 연결할 수 없습니다.")
        print("  → ollama serve 실행 후 재시도")
        return
    except requests.exceptions.HTTPError as e:
        print(f"[오류] HTTP {e.response.status_code}: {e.response.text}")
        return

    # chat 응답 구조: data["message"]["content"]
    message = data.get("message", {})
    print(f"응답:\n{message.get('content', '')}")
    print()

    # 메트릭
    eval_count    = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 1)
    total_dur     = data.get("total_duration", 0) / 1e9
    tps           = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0

    print("--- 성능 메트릭 ---")
    print(f"  eval_count    : {eval_count} tokens")
    print(f"  total_duration: {total_dur:.2f} s")
    print(f"  tokens/sec    : {tps:.1f}")
    print()
    print("비교 포인트:")
    print("  generate → data['response']")
    print("  chat     → data['message']['content']")
    print("  chat 은 messages 배열로 대화 히스토리를 그대로 이어갈 수 있음")
    print("  → day4_06_console_chatbot.py 에서 루프로 확장합니다.")


if __name__ == "__main__":
    main()
