"""Download the official pretrained checkpoints and verify their SHA-256 hashes."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = json.loads((ROOT / "models/sources.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=[item["name"] for item in sources],
                        help="Download only these filenames; the default downloads all three")
    args = parser.parse_args()
    for item in sources:
        if args.models and item["name"] not in args.models:
            continue
        path = ROOT / "models" / item["name"]
        if not path.exists():
            temporary = path.with_suffix(".part")
            print("Downloading", item["name"])
            with urllib.request.urlopen(item["url"], timeout=60) as response, temporary.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
            if temporary.stat().st_size != item["bytes"] or digest(temporary) != item["sha256"]:
                raise ValueError(f"Download verification failed for {item['name']}")
            temporary.replace(path)
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Existing model does not match the recorded checksum: {path.name}")
        print("Verified", path.name)


if __name__ == "__main__":
    main()
