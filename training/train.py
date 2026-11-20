#!/usr/bin/env python3
"""
Fine-tune YOLOv8-nano on the merged VisionAI dataset (local GPU/CPU run).
On Colab, use train_colab.ipynb instead (it calls the same Ultralytics API).

Examples:
  python train.py --data datasets/visionai/data.yaml
  python train.py --data datasets/visionai/data.yaml --model yolo26n.pt --name visionai_yolo26
  python train.py --data datasets/visionai/data.yaml --name visionai_v1 --resume
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml made by merge_datasets.py")
    ap.add_argument("--model", default="yolov8n.pt", help="starting weights (your yolov8n.pt, or yolo26n.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20, help="stop early if val does not improve for N epochs")
    ap.add_argument("--name", default="visionai_v1")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--device", default=None, help="e.g. 0 for first GPU, cpu")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run with the same --name")
    ap.add_argument("--pi-imgsz", type=int, default=320, help="NCNN export size; must match imgsz in rpi_test.py")
    ap.add_argument("--export-ncnn", action="store_true", help="also export NCNN for the Raspberry Pi")
    args = ap.parse_args()

    run_dir = Path(args.project) / args.name
    if args.resume:
        model = YOLO(str(run_dir / "weights" / "last.pt"))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            patience=args.patience, project=args.project, name=args.name,
            device=args.device, exist_ok=True,
        )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best}")

    best_model = YOLO(str(best))
    metrics = best_model.val(data=args.data, imgsz=args.imgsz)
    print(f"\nOverall  mAP50: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")
    for i, c in enumerate(metrics.box.ap_class_index):
        print(f"  {best_model.names[int(c)]:<22} mAP50 {metrics.box.ap50[i]:.3f}   "
              f"mAP50-95 {metrics.box.maps[int(c)]:.3f}")

    if args.export_ncnn:
        out = best_model.export(format="ncnn", imgsz=args.pi_imgsz)
        print(f"NCNN model: {out}")


if __name__ == "__main__":
    main()
