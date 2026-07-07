# day3_07_train_yolo11_custom.py
# 슬라이드 96~102 · YOLO11n 전이학습 — 커스텀 데이터셋으로 best.pt 생성
#
# 실행:
#   # 테스트 학습 (먼저 오류 여부 확인)
#   python src/day3_07_train_yolo11_custom.py --epochs 10 --batch 8
#
#   # 본 학습
#   python src/day3_07_train_yolo11_custom.py --epochs 50 --batch 16
#
#   # 메모리 부족 시
#   python src/day3_07_train_yolo11_custom.py --epochs 50 --imgsz 320 --batch 4
#
# 결과 파일:
#   runs/detect/{name}/weights/best.pt  ← Pi5 에서 실시간 탐지에 사용
#   runs/detect/{name}/weights/last.pt
#   runs/detect/{name}/results.png      ← 학습 그래프
#
# CLI 방식 (빠른 실행):
#   yolo detect train model=../models/yolo11n.pt \
#     data=../datasets/custom_objects/data.yaml \
#     epochs=10 imgsz=640 batch=8 name=test_run
#
# 주의: 학습은 Pi5 단독보다 PC/GPU 또는 Colab 권장
#       Pi5 단독 학습 시 발열·시간 주의

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11n transfer learning")
    parser.add_argument("--model",    default=str(ROOT / "models" / "yolo11n.pt"),
                        help="시작 모델 (기본: yolo11n.pt)")
    parser.add_argument("--data",     default=str(ROOT / "datasets" / "custom_objects" / "data.yaml"),
                        help="data.yaml 경로 (Roboflow export 후 경로 확인)")
    parser.add_argument("--epochs",   type=int,   default=50,
                        help="학습 반복 횟수 (테스트: 10 / 본학습: 30~50)")
    parser.add_argument("--imgsz",    type=int,   default=640,
                        help="입력 이미지 크기 (빠른 속도: 320 / 정확도: 640)")
    parser.add_argument("--batch",    type=int,   default=16,
                        help="배치 크기 (메모리 부족 시 8 또는 4 로 낮춤)")
    parser.add_argument("--patience", type=int,   default=20,
                        help="개선 없을 때 조기 종료 기준 epoch 수")
    parser.add_argument("--name",     default="team_yolo11n_custom",
                        help="결과 폴더 이름 — runs/detect/{name}/")
    parser.add_argument("--device",   default="",
                        help="학습 장치: '' (자동) / 'cpu' / 'cuda:0' / 'mps'")
    return parser.parse_args()


def check_data_yaml(data_path: str) -> bool:
    """학습 전 data.yaml 존재 여부 확인 (슬라이드 101)"""
    p = Path(data_path)
    if not p.exists():
        print(f"\n[오류] data.yaml 을 찾을 수 없습니다: {p.resolve()}")
        print("  Roboflow → Generate Version → YOLOv11 형식 export 후 경로를 확인하세요.")
        print(f"  확인 명령: ls -lh {p.resolve()}")
        return False
    print(f"[확인] data.yaml: {p.resolve()}")
    return True


def main():
    args = parse_args()

    print("=" * 54)
    print(f"  모델    : {args.model}")
    print(f"  데이터  : {args.data}")
    print(f"  epochs  : {args.epochs}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  batch   : {args.batch}")
    print(f"  patience: {args.patience}")
    print(f"  name    : {args.name}")
    print(f"  device  : {args.device if args.device else '자동'}")
    print("=" * 54)

    if not check_data_yaml(args.data):
        return

    print(f"\nloading model: {args.model}")
    model = YOLO(args.model)

    print("\n학습 시작 ...")
    print("(첫 번째 epoch 가 느릴 수 있습니다 — 정상입니다)")

    train_kwargs = dict(
        data     = args.data,
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        patience = args.patience,
        name     = args.name,
    )
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    # 결과 경로 안내
    best_path = Path("runs") / "detect" / args.name / "weights" / "best.pt"
    print("\n" + "=" * 54)
    print("학습 완료!")
    print(f"  best.pt  : {best_path.resolve()}")
    print(f"  결과 폴더: runs/detect/{args.name}/")
    print("\n다음 단계:")
    print(f"  1. Pi5 로 best.pt 복사")
    print(f"  2. python src/day3_02_realtime_yolo_basic.py "
          f"--model runs/detect/{args.name}/weights/best.pt")
    print(f"  3. python src/day3_03_benchmark_yolo.py "
          f"--model runs/detect/{args.name}/weights/best.pt --frames 300")


if __name__ == "__main__":
    main()
