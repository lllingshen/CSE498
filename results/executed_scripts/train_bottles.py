"""Fine-tune YOLO11m on our small bottle dataset."""
from pathlib import Path

from ultralytics import YOLO

project = Path(__file__).resolve().parent
step3 = project.parent / "step3"
model = YOLO(str(project / "models" / "yolo11m.pt"))

if __name__ == "__main__":
    model.train(
        data=str(step3 / "dataset" / "data.yaml"),
        epochs=30,
        patience=10,
        imgsz=960,
        batch=4,
        freeze=10,  # Keep the first 10 layers' parameters fixed.
        optimizer="AdamW",
        lr0=0.0005,
        seed=42,
        device=0,  # Use the GPU; change to "cpu" if needed.
        workers=0,
        amp=False,
        project=str(step3),
        name="training",
        plots=True,
    )
