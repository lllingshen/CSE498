"""Run YOLO11m on the two apartment photos."""
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

# Find the photos and choose where to save the results.
project = Path(__file__).resolve().parent
input_dir = project.parent / "step2" / "inputs"
output_dir = project.parent / "step2" / "improved_results"
output_dir.mkdir(parents=True, exist_ok=True)
image_names = ["1P9A1048.JPG", "1P9A1049.JPG"]

# Use a larger YOLO11 model to find more objects.
model = YOLO(str(project / "models" / "yolo11m.pt"))

with (output_dir / "detections.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2"])

    for image_name in image_names:
        image_path = input_dir / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Keep more detail and remove overlapping predictions of the same object.
        result = model.predict(image, conf=0.25, imgsz=1920, iou=0.45,
                               agnostic_nms=True, device="cpu")[0]

        # Draw the boxes and save an annotated photo.
        output_path = output_dir / f"{image_path.stem}_annotated.jpg"
        if not cv2.imwrite(str(output_path), result.plot(line_width=6)):
            raise OSError(f"Cannot save image: {output_path}")

        for box in result.boxes:
            class_name = model.names[int(box.cls.item())]
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            writer.writerow([image_name, class_name, confidence, x1, y1, x2, y2])
            print(f"{class_name}: {confidence:.2f}")

        print(f"{image_name}: {len(result.boxes)} detections")

print(f"Results saved to {output_dir}")
