"""Download the pretrained models used in this project."""
from pathlib import Path
from urllib.request import urlretrieve

models = Path(__file__).resolve().parent / "models"
models.mkdir(exist_ok=True)
url = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

for name in ["yolo11n.pt", "yolo11m.pt", "yoloe-11m-seg.pt"]:
    path = models / name
    if path.exists():
        print(f"{name} is already downloaded")
        continue
    print(f"Downloading {name}...")
    temporary = path.with_suffix(".part")
    urlretrieve(f"{url}/{name}", temporary)
    temporary.replace(path)
    print(f"Saved {name}")
