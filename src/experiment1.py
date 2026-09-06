"""Detect objects in the two apartment photos with YOLO11."""
import argparse
import csv
import json

from common import ROOT, PHOTOS, require_model
import cv2
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="Use the original nano model and settings")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    from pathlib import Path
    stage = "baseline" if args.baseline else "improved"
    output = Path(args.output) if args.output else ROOT / "runs/experiment1" / stage
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    name = "yolo11n.pt" if args.baseline else "yolo11m.pt"
    model = YOLO(str(require_model(name)))
    options = {"conf": 0.5, "imgsz": 640, "device": args.device} if args.baseline else {
        "conf": 0.25, "imgsz": 1920, "iou": 0.45, "agnostic_nms": True, "device": args.device}
    counts = {}
    with (output / "detections.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2"])
        for photo in PHOTOS:
            image = cv2.imread(str(photo))
            if image is None:
                raise ValueError(f"Cannot read {photo}")
            result = model.predict(image, **options)[0]
            drawn = result.plot() if args.baseline else result.plot(line_width=6)
            if not cv2.imwrite(str(output / f"{photo.stem}_annotated.jpg"), drawn):
                raise OSError("Could not save annotated photo")
            for box in result.boxes:
                writer.writerow([photo.name, model.names[int(box.cls.item())],
                    float(box.conf.item()), *box.xyxy[0].tolist()])
            counts[photo.name] = len(result.boxes)
    (output / "settings.json").write_text(json.dumps({"model": name, "predict_options": options,
        "detection_counts": counts, "note": "Counts are predictions, not accuracy."}, indent=2))
    print(counts)
    print("Saved results to", output)


if __name__ == "__main__":
    main()
