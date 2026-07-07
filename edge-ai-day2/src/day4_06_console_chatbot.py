# day4_06_console_chatbot.py
# 슬라이드 73 · Lab E: 콘솔 챗봇 — messages 배열 유지 대화 루프
#
# 실행:
#   python src/day4_06_console_chatbot.py
#   python src/day4_06_console_chatbot.py --model llama3.2:1b
#   python src/day4_06_console_chatbot.py --model qwen2.5:0.5b
#
# 조작: YOU> 에 명령 입력 / q, quit, exit → 종료
#
# 목적:
#   - messages 배열로 대화 히스토리를 유지하는 구조 이해
#   - SLM 호출 루프 자체가 학습 목표 (대화형 UI 가 목표가 아님)
#   - 이어서 messages 에 JSON 출력 규칙을 넣으면 day4_08 로 발전
#
# 관찰 포인트 (슬라이드 73):
#   같은 명령도 "컵 → coffee", "가위 → glove" 처럼 빗나감
#   → JSON 강제(format 필드)와 alias 검증이 필요한 이유

import argparse

import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
TIMEOUT         = 120

SYSTEM_PROMPT = (
    "너는 온디바이스 AI 명령 해석기야. "
    "사용자 명령에서 대상 물체만 영어 단어로 답해. "
    "한 단어로만 답해."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Ollama 콘솔 챗봇")
    parser.add_argument("--model",  default="qwen2.5:0.5b")
    parser.add_argument("--system", default=SYSTEM_PROMPT)
    return parser.parse_args()


def call_chat(model: str, messages: list[dict]) -> str:
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": 0},
    }
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def main():
    args = parse_args()

    # system 메시지로 messages 초기화
    messages: list[dict] = [
        {"role": "system", "content": args.system},
    ]

    print(f"콘솔 챗봇 시작 (model={args.model})")
    print(f"system: {args.system}")
    print("종료: q / quit / exit")
    print("-" * 50)

    # 연결 확인
    try:
        call_chat(args.model, messages + [{"role": "user", "content": "ping"}])
    except requests.exceptions.ConnectionError:
        print("[오류] Ollama 서버에 연결할 수 없습니다.")
        print("  → ollama serve 실행 후 재시도")
        return
    except Exception:
        pass   # ping 실패해도 계속 진행

    while True:
        try:
            user_input = input("YOU> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"q", "quit", "exit"}:
            break

        # user 메시지 추가
        messages.append({"role": "user", "content": user_input})

        try:
            response = call_chat(args.model, messages)
        except requests.exceptions.ConnectionError:
            print("BOT> [오류] 서버 연결 끊김")
            break
        except Exception as e:
            print(f"BOT> [오류] {e}")
            continue

        # assistant 응답을 히스토리에 추가 (다음 턴에 컨텍스트로 전달)
        messages.append({"role": "assistant", "content": response})

        print(f"BOT> {response}")

    print("\n종료.")
    print(f"총 대화 턴: {(len(messages) - 1) // 2}")
    print()
    print("관찰:")
    print("  같은 명령도 매번 다른 단어가 나올 수 있습니다.")
    print("  → day4_08_ollama_intent_json.py 에서 JSON schema 로 출력을 강제합니다.")


if __name__ == "__main__":
    main()
