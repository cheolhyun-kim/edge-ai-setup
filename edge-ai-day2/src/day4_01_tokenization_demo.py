# day4_01_tokenization_demo.py
# 슬라이드 24 · Lab A: 토큰화 데모 실행
#
# 실행: python src/day4_01_tokenization_demo.py
#
# 목적:
#   - 한국어 문장이 문자 단위·띄어쓰기 단위로 어떻게 쪼개지는지 확인
#   - 의미 단위(형태소)와 단순 분리의 차이를 체감
#   - "왜 alias 와 SLM 이 필요한가" 를 이해하는 출발점
#
# 추가 패키지 불필요 — 파이썬 내장 함수만 사용

SAMPLE_SENTENCES = [
    "가위 찾아줘",
    "가위가 어디 있어?",
    "화면에서 가위를 찾아 봐",
    "가위 좀 보여줄래?",
    "잘라내는 도구를 찾아줘",
    "화면에 뭐가 보여?",
    "보이는 물체 목록 알려줘",
    "컵 어디 있어?",
    "멋있는 거 찾아줘",
]


def char_tokens(sentence: str) -> list[str]:
    """문자 단위 토큰화 — 공백 포함 한 글자씩"""
    return list(sentence)


def space_tokens(sentence: str) -> list[str]:
    """띄어쓰기 단위 토큰화"""
    return sentence.split()


def print_separator(char: str = "-", width: int = 60):
    print(char * width)


def main():
    print("=" * 60)
    print("  한국어 토큰화 데모 (슬라이드 24 · Lab A)")
    print("=" * 60)
    print()

    for sent in SAMPLE_SENTENCES:
        print(f"원문    : {sent}")

        c_tok = char_tokens(sent)
        print(f"문자 단위 ({len(c_tok)}개): {c_tok}")

        s_tok = space_tokens(sent)
        print(f"띄어쓰기 ({len(s_tok)}개): {s_tok}")

        print_separator()

    print()
    print("관찰 포인트")
    print("  1. 문자 단위 토큰은 의미 단위와 맞지 않음")
    print("  2. 띄어쓰기 기준도 조사/어미 때문에 한계가 있음")
    print("     예) '가위가' → '가위' + '가(주격 조사)' 를 분리 못함")
    print("  3. '가위 찾아줘' 와 '가위가 어디 있어?' 는 같은 의도지만")
    print("     단순 토큰 비교로는 같다고 판단하기 어려움")
    print()
    print("  → alias 사전과 SLM 이 이 간격을 메워줍니다.")
    print("  → ko_to_class.json: '가위' → 'scissors' 로 직접 매핑")
    print("  → SLM: 다양한 표현을 같은 intent 로 묶음")


if __name__ == "__main__":
    main()
