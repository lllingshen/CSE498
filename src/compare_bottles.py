"""Compare bottle detection before and after fine-tuning, using the same images."""
import argparse
import csv
import json
from pathlib import Path

from common import ROOT, PHOTOS, dataset_yaml

import cv2
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix

DATA = None
OUTPUT = ROOT / "runs/experiment2"
MODELS = {
    "before": ROOT / "models/yolo11m.pt",
    "after": ROOT / "models/best_bottles.pt",
}
COMMON = dict(iou=0.45, agnostic_nms=True, device="cpu", max_det=300, augment=False)


def bottle_id(names):
    """The COCO model and our one-class model use different class numbers."""
    matches = [index for index, name in names.items() if name.lower() == "bottle"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one bottle class, found {matches}")
    return matches[0]


class BottleValidator(DetectionValidator):
    """Evaluate only bottle scores; map this class to dataset class 0 before NMS."""

    def init_metrics(self, model):
        if self.data["names"] != {0: "bottle"}:
            raise ValueError("This comparison requires a single class: 0 = bottle")
        if len(self.dataloader.dataset) != 10:
            raise ValueError("Expected the fixed 10-image validation split")
        self.source_id = bottle_id(model.names)
        super().init_metrics(model)
        self.names = self.metrics.names = {0: "bottle"}
        self.nc = 1
        self.class_map = [1]
        self.confusion_matrix = ConfusionMatrix(names=self.names)

    def postprocess(self, preds):
        # YOLO11 outputs [x, y, w, h, class scores...] for each candidate box.
        raw = preds[0] if isinstance(preds, (tuple, list)) else preds
        score = raw[:, 4 + self.source_id:5 + self.source_id, :]
        bottle_only = torch.cat((raw[:, :4, :], score), dim=1)
        return super().postprocess(bottle_only)

    def preprocess(self, batch):
        if torch.any(batch["cls"] != 0):
            raise ValueError("Validation labels must all use bottle class 0")
        return super().preprocess(batch)


def evaluate(model, stage):
    # Low confidence is needed to measure the full precision-recall curve for AP.
    # Do not set classes=[COCO bottle ID]: it would also filter the dataset labels.
    validator = BottleValidator(save_dir=OUTPUT / f"val_{stage}", args={
        **COMMON, "data": str(DATA), "split": "val", "imgsz": 960,
        "conf": 0.001, "batch": 1, "workers": 0, "plots": False,
        "save_json": False, "save_txt": False, "classes": None,
        "single_cls": False, "verbose": False,
    })
    validator(model=model.model)
    return {
        "AP50": float(validator.metrics.box.map50),
        "AP50_95": float(validator.metrics.box.map),
        "validation_images": int(validator.seen),
        "ground_truth_bottles": int(validator.metrics.nt_per_class.sum()),
        "source_class_id": validator.source_id,
        "evaluated_class_id": 0,
    }


def main():
    global DATA, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/experiment2")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/best_bottles.pt",
                        help="Optional checkpoint from a new training run")
    args = parser.parse_args()
    DATA = dataset_yaml()
    OUTPUT = args.output
    COMMON["device"] = args.device
    MODELS["after"] = args.weights
    for path in [DATA, *MODELS.values(), *PHOTOS]:
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(8)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "target": "bottle",
        "validation_settings": {**COMMON, "imgsz": 960, "conf": 0.001, "batch": 1},
        "apartment_settings": {**COMMON, "imgsz": 1920, "conf": 0.25},
        "models": {},
        "limitations": [
            "The 10 validation images also select best.pt; they are not an independent test set.",
            "Apartment photos have no ground-truth boxes: detection counts are not accuracy.",
            "This evaluates bottle detection only, not retention of all 80 COCO classes.",
        ],
    }
    with (OUTPUT / "apartment_detections.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["stage", "image", "class", "model_class_id", "confidence", "x1", "y1", "x2", "y2"])
        for stage, weights in MODELS.items():
            model = YOLO(str(weights))
            class_id = bottle_id(model.names)
            scores = evaluate(model, stage)
            scores["weights"] = str(weights)
            scores["apartment_detection_counts"] = {}
            for photo in PHOTOS:
                # Each model gets its own bottle ID; filters cannot carry over between models.
                result = model.predict(str(photo), **COMMON, conf=0.25, imgsz=1920,
                                       classes=[class_id], verbose=False)[0]
                image_path = OUTPUT / f"{stage}_{photo.stem}.jpg"
                if not cv2.imwrite(str(image_path), result.plot(line_width=6)):
                    raise OSError(f"Cannot save {image_path}")
                for box in result.boxes:
                    writer.writerow([stage, photo.name, "bottle", int(box.cls.item()),
                                     float(box.conf.item()), *box.xyxy[0].tolist()])
                scores["apartment_detection_counts"][photo.name] = len(result.boxes)
            report["models"][stage] = scores
            print(stage, scores)
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Comparison saved to {OUTPUT}")


if __name__ == "__main__":
    main()
