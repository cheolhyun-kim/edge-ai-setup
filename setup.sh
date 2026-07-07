#!/bin/bash
# setup.sh
# Raspberry Pi 5 Edge AI — 1회 실행으로 전체 환경 세팅
#
# 사용법:
#   git clone https://github.com/YOUR_USERNAME/edge-ai-setup.git
#   cd edge-ai-setup
#   bash setup.sh
#
# 소요 시간: 약 10~20분 (네트워크 속도에 따라 다름)

set -e  # 오류 발생 시 즉시 중단

# ── 색상 출력 ────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

info "=========================================="
info "  Raspberry Pi 5 Edge AI Setup"
info "  repo: $REPO_DIR"
info "=========================================="

# ── 1. 폴더 구조 생성 ────────────────────────────────────────────────────────
info "[1/6] 폴더 구조 생성"

mkdir -p ~/edge-ai-day1/{src,data,outputs,logs}
mkdir -p ~/edge-ai-day2/{src,data,models,outputs/{benchmarks,captures,demo_frames,}}

info "폴더 생성 완료"
echo ""
tree ~ -L 3 --dirsfirst -d 2>/dev/null || find ~ -maxdepth 3 -type d | sort

# ── 2. 소스 파일 복사 ────────────────────────────────────────────────────────
info ""
info "[2/6] 소스 파일 복사"

# Day1 (Day02 PPT 실습 코드)
cp "$REPO_DIR"/edge-ai-day1/src/*.py ~/edge-ai-day1/src/
info "edge-ai-day1/src/ 복사 완료"

# Day2 src (Day03~05 코드)
cp "$REPO_DIR"/edge-ai-day2/src/*.py ~/edge-ai-day2/src/
info "edge-ai-day2/src/ 복사 완료"

# Day2 data (json, txt, csv)
cp "$REPO_DIR"/edge-ai-day2/data/* ~/edge-ai-day2/data/
info "edge-ai-day2/data/ 복사 완료"

# ── 3. Python 패키지 설치 ────────────────────────────────────────────────────
info ""
info "[3/6] Python 패키지 확인"

# edge-ai-day1 venv
if [ -d ~/edge-ai-day1/.venv ]; then
    info "edge-ai-day1/.venv 이미 존재 — 건너뜀"
else
    info "edge-ai-day1/.venv 생성 중..."
    python3 -m venv ~/edge-ai-day1/.venv
    ~/edge-ai-day1/.venv/bin/pip install --quiet \
        opencv-python \
        numpy
    info "edge-ai-day1/.venv 설치 완료"
fi

# edge-ai-day2 venv
if [ -d ~/edge-ai-day2/.venv ]; then
    info "edge-ai-day2/.venv 이미 존재 — 건너뜀"
else
    info "edge-ai-day2/.venv 생성 중..."
    python3 -m venv ~/edge-ai-day2/.venv
    ~/edge-ai-day2/.venv/bin/pip install --quiet \
        opencv-python \
        "ultralytics[export]" \
        requests \
        pydantic \
        numpy
    info "edge-ai-day2/.venv 설치 완료"
fi

# ── 4. Ollama 설치 및 확인 ───────────────────────────────────────────────────
info ""
info "[4/6] Ollama 설치 확인"

if command -v ollama &> /dev/null; then
    info "Ollama 이미 설치됨: $(ollama --version)"
else
    info "Ollama 설치 중..."
    curl -fsSL https://ollama.com/install.sh | sh
    info "Ollama 설치 완료"
fi

# Ollama 서비스 시작
if systemctl is-active --quiet ollama 2>/dev/null; then
    info "Ollama 서비스 실행 중"
else
    info "Ollama 서비스 시작..."
    sudo systemctl enable ollama 2>/dev/null || true
    sudo systemctl start  ollama 2>/dev/null || ollama serve &
    sleep 3
fi

# 서비스 연결 확인
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    info "Ollama API 응답 확인 완료"
else
    warning "Ollama API 응답 없음 — 나중에 수동으로 'ollama serve' 실행하세요"
fi

# ── 5. SLM 모델 다운로드 ─────────────────────────────────────────────────────
info ""
info "[5/6] SLM 모델 다운로드"

SLM_MODELS=("qwen2.5:0.5b" "gemma3:270m")

for model in "${SLM_MODELS[@]}"; do
    if ollama list 2>/dev/null | grep -q "${model%%:*}"; then
        info "$model 이미 존재 — 건너뜀"
    else
        info "$model 다운로드 중..."
        ollama pull "$model" || warning "$model 다운로드 실패 — 나중에 수동으로 실행하세요"
    fi
done

# ── 6. YOLO 모델 다운로드 ────────────────────────────────────────────────────
info ""
info "[6/6] YOLO 모델 다운로드"

YOLO_MODEL=~/edge-ai-day2/models/yolo11n.pt

if [ -f "$YOLO_MODEL" ]; then
    info "yolo11n.pt 이미 존재 — 건너뜀"
else
    info "yolo11n.pt 다운로드 중..."
    python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" 2>/dev/null && \
        mv yolo11n.pt "$YOLO_MODEL" 2>/dev/null || \
        python3 -c "
from ultralytics import YOLO
import shutil, pathlib
m = YOLO('yolo11n.pt')
src = pathlib.Path('yolo11n.pt')
dst = pathlib.Path('$YOLO_MODEL')
dst.parent.mkdir(parents=True, exist_ok=True)
if src.exists(): shutil.move(str(src), str(dst))
print('yolo11n.pt 저장 완료:', dst)
"
fi

# ── 완료 메시지 ──────────────────────────────────────────────────────────────
echo ""
info "=========================================="
info "  설치 완료!"
info "=========================================="
echo ""
echo "  다음 명령으로 동작을 확인하세요:"
echo ""
echo "  # Day02 카메라 확인"
echo "  cd ~/edge-ai-day1"
echo "  python src/practice1.py"
echo ""
echo "  # Day03 YOLO smoke test"
echo "  cd ~/edge-ai-day2"
echo "  python src/day3_01_quick_predict.py"
echo ""
echo "  # Day04 Ollama API 확인"
echo "  cd ~/edge-ai-day2"
echo "  python src/day4_03_generate_once.py"
echo ""
echo "  # Day05 최종 데모 (mock mode)"
echo "  cd ~/edge-ai-day2"
echo "  python src/day5_04_final_demo.py --mock"
echo ""
info "=========================================="
