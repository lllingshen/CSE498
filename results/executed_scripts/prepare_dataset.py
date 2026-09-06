"""Make a small, one-class Bottle dataset from Open Images."""
import csv
import hashlib
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image

root = Path(__file__).resolve().parent.parent
output = root / "step3" / "dataset"
cache = root.parent.parent / "work" / "cse498-step3" / "dataset-source"
cache.mkdir(parents=True, exist_ok=True)
output.mkdir(parents=True, exist_ok=True)
sources = {
    "validation-annotations-bbox.csv": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "oidv7-class-descriptions-boxable.csv": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
    "validation-images-with-rotation.csv": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
}


def download(url, path):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        path.write_bytes(data)


def read_csv(name):
    with (cache / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


# Download only annotations now; the photos are downloaded one at a time below.
for name, url in sources.items():
    download(url, cache / name)
classes = read_csv("oidv7-class-descriptions-boxable.csv")
bottle_id = next(row["LabelName"] for row in classes if row["DisplayName"] == "Bottle")
metadata = {row["ImageID"]: row for row in read_csv("validation-images-with-rotation.csv")}
boxes = defaultdict(list)
for row in read_csv("validation-annotations-bbox.csv"):
    if row["LabelName"] == bottle_id:
        boxes[row["ImageID"]].append(row)

# Group boxes and drawings are not individual real bottles. Exclude the whole image.
eligible = sorted(image_id for image_id, rows in boxes.items()
                  if all(row["IsGroupOf"] == "0" and row["IsDepiction"] == "0" for row in rows))
random.Random(42).shuffle(eligible)
selected, skipped, hashes = [], [], set()
for image_id in eligible:
    path = cache / f"{image_id}.jpg"
    url = f"https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
    try:
        download(url, path)
        with Image.open(path) as photo:
            photo.load()
            width, height = photo.size
            pixel_hash = hashlib.sha256(photo.convert("RGB").tobytes()).hexdigest()
        if pixel_hash in hashes:
            raise ValueError("duplicate image pixels")
        for row in boxes[image_id]:
            x1, x2, y1, y2 = (float(row[key]) for key in ("XMin", "XMax", "YMin", "YMax"))
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("invalid bounding box")
    except Exception as error:
        skipped.append({"image_id": image_id, "reason": str(error)})
        continue
    hashes.add(pixel_hash)
    selected.append((image_id, width, height, url))
    if len(selected) == 50:
        break
if len(selected) != 50:
    raise RuntimeError("Could not obtain 50 valid, distinct photos")

# The first 40 randomly selected images train the model; the last 10 validate it.
manifest, original_boxes = [], []
for index, (image_id, width, height, url) in enumerate(selected):
    split = "train" if index < 40 else "val"
    images_dir, labels_dir = output / "images" / split, output / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    photo_bytes = (cache / f"{image_id}.jpg").read_bytes()
    (images_dir / f"{image_id}.jpg").write_bytes(photo_bytes)
    labels = []
    for row in boxes[image_id]:
        x1, x2, y1, y2 = (float(row[key]) for key in ("XMin", "XMax", "YMin", "YMax"))
        labels.append(f"0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}")
        original_boxes.append(row)
    (labels_dir / f"{image_id}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    info = metadata[image_id]
    manifest.append({"image_id": image_id, "split": split, "source_split": "validation",
                     "width": width, "height": height, "bottle_count": len(labels), "source_url": url,
                     "original_url": info["OriginalURL"], "landing_url": info["OriginalLandingURL"],
                     "license": info["License"], "author": info["Author"], "title": info["Title"],
                     "sha256": hashlib.sha256(photo_bytes).hexdigest()})

for name, rows in (("manifest.csv", manifest), ("source_bottle_boxes.csv", original_boxes)):
    with (output / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
(output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
(output / "selection.json").write_text(json.dumps({"seed": 42, "eligible_images": len(eligible),
    "candidate_order": eligible, "skipped": skipped, "metadata_sources": sources}, indent=2), encoding="utf-8")
(output / "data.yaml").write_text(f"path: {output}\ntrain: images/train\nval: images/val\nnames:\n  0: bottle\n", encoding="utf-8")
print(f"Saved {len(manifest)} photos and {len(original_boxes)} bottle labels to {output}")
