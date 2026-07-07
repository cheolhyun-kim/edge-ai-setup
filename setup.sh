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
echo "[1/4] Day1 파일 배치 중..."
mkdir -p "$DAY1_DEST/src" "$DAY1_DEST/data" "$DAY1_DEST/outputs" "$DAY1_DEST/logs"
cp -r "$DAY1_SRC/src/"* "$DAY1_DEST/src/"
echo "  ✓ edge-ai-day1/src/ 완료"

# ── Day2 코드 ────────────────────────────────────
echo ""
echo "[2/4] Day2 코드 파일 배치 중..."
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
echo "[3/4] 데이터 파일 배치 중..."
cp -r "$DAY2_SRC/data/"* "$DAY2_DEST/data/"
echo "  ✓ edge-ai-day2/data/ 완료"

# ── YOLO 모델 다운로드 ────────────────────────────
echo ""
echo "[4/4] YOLO11n 모델 확인 중..."
MODEL_PATH="$DAY2_DEST/models/yolo11n.pt"
if [ -f "$MODEL_PATH" ]; then
    echo "  ✓ yolo11n.pt 이미 존재 — 건너뜀"
else
    echo "  yolo11n.pt 다운로드 중 (ultralytics 사용)..."
    python3 - <<'PYEOF'
from ultralytics import YOLO
import shutil, os
model = YOLO("yolo11n.pt")   # ~/.cache/ultralytics/ 에 다운로드
src = os.path.expanduser("~/.cache/ultralytics/assets/yolo11n.pt")
# ultralytics가 현재 디렉토리에 저장하는 경우도 대비
if not os.path.exists(src):
    src = "yolo11n.pt"
dst = os.path.expanduser("~/edge-ai-day2/models/yolo11n.pt")
if os.path.exists(src) and src != dst:
    shutil.copy(src, dst)
    print(f"  복사 완료: {dst}")
elif os.path.exists(dst):
    print(f"  이미 존재: {dst}")
PYEOF
    if [ -f "$MODEL_PATH" ]; then
        echo "  ✓ yolo11n.pt 준비 완료"
    else
        echo "  ⚠ 자동 다운로드 실패. 수동으로 실행하세요:"
        echo "    cd ~/edge-ai-day2 && python3 -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\""
        echo "    mv yolo11n.pt ~/edge-ai-day2/models/"
    fi
fi

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
echo "   ├── models/      ← yolo11n.pt"
echo "   └── outputs/"
echo "       ├── benchmarks/"
echo "       ├── captures/"
echo "       └── demo_frames/"
echo ""
echo " 첫 실행 테스트:"
echo "   cd ~/edge-ai-day1 && python3 src/practice1.py"
echo "   cd ~/edge-ai-day2 && python3 src/day3_01_quick_predict.py"
echo "============================================"
