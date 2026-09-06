"""Fine-tune YOLO11m on the fixed 40-image training split."""
import argparse
from pathlib import Path

from common import ROOT, dataset_yaml, require_model
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Use 0 for the first CUDA GPU")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/training")
    args = parser.parse_args()
    model = YOLO(str(require_model("yolo11m.pt")))
    model.train(
        data=str(dataset_yaml()), epochs=args.epochs, patience=10,
        imgsz=960, batch=4, freeze=10, optimizer="AdamW", lr0=0.0005,
        seed=42, device=args.device, workers=0, amp=False,
        project=str(args.output.parent), name=args.output.name, plots=True,
        exist_ok=False,  # A rerun gets a new directory; the submitted best model stays unchanged.
    )


if __name__ == "__main__":
    main()
