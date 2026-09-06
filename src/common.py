"""Shared paths and local configuration for the three experiments."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = [ROOT / "data/apartment" / f"1P9A104{i}.JPG" for i in (8, 9)]
CACHE = ROOT / "runs/cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE))
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def require_model(name):
    path = ROOT / "models" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}. Run: python setup_models.py")
    return path


def dataset_yaml():
    """Resolve the bundled data independently of the shell's working directory."""
    import yaml
    folder = ROOT / "data/bottles"
    config = yaml.safe_load((folder / "data.yaml").read_text())
    config["path"] = str(folder)
    runtime = ROOT / "runs/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    path = runtime / "bottles.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path
