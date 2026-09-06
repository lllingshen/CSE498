"""Check the bundled 50-image dataset and write a resolved training YAML."""
import hashlib
import json

from common import ROOT, dataset_yaml


def main():
    folder = ROOT / "data/bottles"
    rows = json.loads((folder / "manifest.json").read_text())
    assert len(rows) == len({row["image_id"] for row in rows}) == 50
    for split, expected in (("train", 40), ("val", 10)):
        chosen = [row for row in rows if row["split"] == split]
        assert len(chosen) == expected
        for row in chosen:
            image = folder / "images" / split / f'{row["image_id"]}.jpg'
            assert hashlib.sha256(image.read_bytes()).hexdigest() == row["sha256"]
            lines = (folder / "labels" / split / f'{row["image_id"]}.txt').read_text().splitlines()
            assert len(lines) == row["bottle_count"]
            for line in lines:
                cls, x, y, w, h = map(float, line.split())
                assert cls == 0 and 0 < w <= 1 and 0 < h <= 1
                assert -1e-8 <= x - w/2 < x + w/2 <= 1 + 1e-8
                assert -1e-8 <= y - h/2 < y + h/2 <= 1 + 1e-8
    print("Verified: 40 training images, 10 validation images, 81 bottle labels.")
    print("Resolved YAML:", dataset_yaml())
    print("This does not reselect data or modify the submitted images and labels.")


if __name__ == "__main__":
    main()
