"""YOLOE text-prompt demo: local photos in, boxes and CSV out; CPU only."""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "step4"
CACHE = Path(os.getenv("CSE498_YOLOE_CACHE", ROOT / ".cache"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE / "ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import torch
from PIL import Image
from ultralytics import YOLOE


def main():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    names = json.loads((OUT / "prompts.json").read_text())
    # The same vocabulary was fixed before viewing either model output.
    model = YOLOE(ROOT / "models/yoloe-11m-seg.pt").to("cpu")
    os.chdir(ROOT / "models")  # MobileCLIP finds its local TorchScript file.
    model.set_classes(names, model.get_text_pe(names))
    model.save_prompt_embeddings(OUT / "prompt_embeddings.npz")
    rows, summary = [], []
    for filename in ("1P9A1048.JPG", "1P9A1049.JPG"):
        source = ROOT.parent / "step2/inputs" / filename
        result = model.predict(source, device="cpu", imgsz=1920, conf=0.25,
                               iou=0.7, max_det=300, half=False, verbose=True)[0]
        # Keep every returned box, including mistakes. CSV uses original pixels.
        for box in result.boxes:
            rows.append([filename, names[int(box.cls.item())], float(box.conf.item()),
                         *box.xyxy[0].tolist()])
        annotated = result.plot(masks=False, line_width=6)
        image = Image.fromarray(annotated[:, :, ::-1])  # BGR to RGB
        stem = source.stem
        image.save(OUT / f"{stem}_annotated.jpg", quality=95)
        image.thumbnail((1920, 1920))
        image.save(OUT / f"{stem}_preview.jpg", quality=93)
        summary.append({"image": filename, "detections": len(result.boxes),
                        "speed_ms": result.speed, "original_shape": result.orig_shape})
    with (OUT / "detections.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
