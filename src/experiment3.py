"""Run YOLOE with the 33 text embeddings saved in the original experiment."""
import argparse
import csv
import json
from pathlib import Path

from common import ROOT, PHOTOS, require_model
import torch
from PIL import Image
from ultralytics import YOLOE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/experiment3")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    names = json.loads((ROOT / "data/prompts/prompts.json").read_text())
    torch_device = f"cuda:{args.device}" if args.device.isdecimal() else args.device
    model = YOLOE(require_model("yoloe-11m-seg.pt")).to(torch_device)
    # The official API restores the recorded vectors; no MobileCLIP download is needed.
    model.load_prompt_embeddings(ROOT / "data/prompts/prompt_embeddings.npz")
    assert list(model.names.values()) == names
    rows, summary = [], []
    for photo in PHOTOS:
        result = model.predict(photo, device=args.device, imgsz=1920, conf=0.25,
            iou=0.7, max_det=300, half=False, verbose=True)[0]
        for box in result.boxes:
            rows.append([photo.name, names[int(box.cls.item())], float(box.conf.item()),
                         *box.xyxy[0].tolist()])
        image = Image.fromarray(result.plot(masks=False, line_width=6)[:, :, ::-1])
        image.save(args.output / f"{photo.stem}_annotated.jpg", quality=95)
        image.thumbnail((1920, 1920))
        image.save(args.output / f"{photo.stem}_preview.jpg", quality=93)
        summary.append({"image": photo.name, "detections": len(result.boxes),
            "speed_ms": result.speed, "original_shape": result.orig_shape})
    with (args.output / "detections.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "label", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved results to", args.output)


if __name__ == "__main__":
    main()
