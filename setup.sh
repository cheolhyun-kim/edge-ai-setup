#!/usr/bin/env bash
# Edge AI Course — Raspberry Pi 5 자동 세팅 스크립트
# 사용법: git clone <repo_url> && cd edge-ai-setup && bash setup.sh
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DAY1_SRC="$REPO_DIR/edge-ai-day1"
DAY2_SRC="$REPO_DIR/edge-ai-day2"
DAY1_DEST="$HOME/edge-ai-day1"
DAY2_DEST="$HOME/edge-ai-day2"

echo "============================================"
echo " Edge AI Course Setup"
echo " repo : $REPO_DIR"
echo " day1 : $DAY1_DEST"
echo " day2 : $DAY2_DEST"
echo "============================================"

# ── Day1 ─────────────────────────────────────────
echo ""
echo "[1/3] Day1 파일 배치 중..."
mkdir -p "$DAY1_DEST/src" "$DAY1_DEST/data" "$DAY1_DEST/outputs" "$DAY1_DEST/logs"
cp -r "$DAY1_SRC/src/"* "$DAY1_DEST/src/"
echo "  ✓ edge-ai-day1/src/ 완료"

# ── Day2 코드 ────────────────────────────────────
echo ""
echo "[2/3] Day2 코드 파일 배치 중..."
mkdir -p "$DAY2_DEST/src" \
         "$DAY2_DEST/data" \
         "$DAY2_DEST/models" \
         "$DAY2_DEST/outputs/benchmarks" \
         "$DAY2_DEST/outputs/captures" \
         "$DAY2_DEST/outputs/demo_frames"
cp -r "$DAY2_SRC/src/"* "$DAY2_DEST/src/"
echo "  ✓ edge-ai-day2/src/ 완료"

# ── Day2 데이터 파일 ──────────────────────────────
echo ""
echo "[3/3] 데이터 파일 배치 중..."
cp -r "$DAY2_SRC/data/"* "$DAY2_DEST/data/"
echo "  ✓ edge-ai-day2/data/ 완료"

# ── 완료 ─────────────────────────────────────────
echo ""
echo "============================================"
echo " 세팅 완료!"
echo ""
echo " 최종 디렉토리 구조:"
echo "   ~/edge-ai-day1/"
echo "   ├── src/         ← Day2 practice 코드"
echo "   ├── data/"
echo "   ├── outputs/"
echo "   └── logs/"
echo ""
echo "   ~/edge-ai-day2/"
echo "   ├── src/         ← Day3~5 코드"
echo "   ├── data/        ← JSON, CSV, txt 설정 파일"
echo "   ├── models/      ← yolo11n.pt (별도 복사 필요)"
echo "   └── outputs/"
echo "       ├── benchmarks/"
echo "       ├── captures/"
echo "       └── demo_frames/"
echo ""
echo " 첫 실행 테스트:"
echo "   cd ~/edge-ai-day1 && python3 src/practice1.py"
echo "   cd ~/edge-ai-day2 && python3 src/day3_01_quick_predict.py"
echo "============================================"
