# day3_05_export_models.py
# 슬라이드 40~41 · .pt → ONNX / NCNN export
#
# 실행:
#   python src/day3_05_export_models.py                            # ONNX + NCNN 모두
#   python src/day3_05_export_models.py --format onnx
#   python src/day3_05_export_models.py --format ncnn
#   python src/day3_05_export_models.py --model ../models/yolo11n.pt --imgsz 320
#
# 출력:
#   ../models/yolo11n.onnx
#   ../models/yolo11n_ncnn_model/
#
# 원칙: .pt 로 먼저 동작 확인 후 export → benchmark_yolo.py 로 FPS 비교

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO model export — ONNX / NCNN")
    parser.add_argument("--model",  default=str(MODELS_DIR / "yolo11n.pt"))
    parser.add_argument("--imgsz",  type=int, default=320)
    parser.add_argument("--format", default="all",
                        choices=["onnx", "ncnn", "all"],
                        help="변환 포맷 (기본: all — ONNX 와 NCNN 모두)")
    return parser.parse_args()


def export_onnx(model_path: str, imgsz: int) -> str:
    print(f"\n[ONNX export] {Path(model_path).name} → imgsz={imgsz}")
    model    = YOLO(model_path)
    exported = model.export(format="onnx", imgsz=imgsz)
    print(f"  저장: {exported}")

    # 동작 확인 (1프레임 predict)
    print("  [동작 확인] ONNX 모델 predict 1회 ...")
    onnx_model = YOLO(str(exported))
    results    = onnx_model.predict(source=0, imgsz=imgsz, conf=0.35,
                                    show=False, verbose=False, save=False)
    print(f"  탐지 수: {len(results[0].boxes)}")
    print(f"  완료: {exported}")
    return str(exported)


def export_ncnn(model_path: str, imgsz: int) -> str:
    print(f"\n[NCNN export] {Path(model_path).name} → imgsz={imgsz}")
    model    = YOLO(model_path)
    exported = model.export(format="ncnn", imgsz=imgsz)
    # NCNN 결과는 폴더로 생성됨
    print(f"  저장(폴더): {exported}")
    print("  주의: NCNN 은 파일이 아니라 폴더입니다 — 경로를 폴더로 지정해야 합니다.")

    # 동작 확인
    print("  [동작 확인] NCNN 모델 predict 1회 ...")
    ncnn_model = YOLO(str(exported))
    results    = ncnn_model.predict(source=0, imgsz=imgsz, conf=0.35,
                                    show=False, verbose=False, save=False)
    print(f"  탐지 수: {len(results[0].boxes)}")
    print(f"  완료: {exported}/")
    return str(exported)


def main():
    args = parse_args()

    print("=" * 52)
    print(f"  모델    : {args.model}")
    print(f"  저장    : {MODELS_DIR}/")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  포맷    : {args.format}")
    print("=" * 52)
    print("원칙: .pt 로 먼저 동작 확인 후 export → FPS 비교")

    if args.format in ("onnx", "all"):
        try:
            export_onnx(args.model, args.imgsz)
        except Exception as e:
            print(f"  [오류] ONNX export 실패: {e}")

    if args.format in ("ncnn", "all"):
        try:
            export_ncnn(args.model, args.imgsz)
        except Exception as e:
            print(f"  [오류] NCNN export 실패: {e}")

    print("\nexport 완료 — benchmark_yolo.py 로 FPS 를 비교하세요.")
    print(f"  예) python src/day3_03_benchmark_yolo.py "
          f"--model ../models/yolo11n.onnx --imgsz {args.imgsz} --frames 300")
    print(f"  예) python src/day3_03_benchmark_yolo.py "
          f"--model ../models/yolo11n_ncnn_model --imgsz {args.imgsz} --frames 300")


if __name__ == "__main__":
    main()
