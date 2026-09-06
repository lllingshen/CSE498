import cv2
from ultralytics import YOLO

coco_model = YOLO("./models/yolo11n.pt")  # Load a pretrained YOLOv11 model


cap = cv2.VideoCapture(0)  # Open a video file

# read frames
frame_number = -1
ret = True

while ret:
    frame_number += 1
    ret, frame = cap.read()  # Read a frame from the video
    if ret:
        detections = coco_model.predict(
            frame, conf=0.5
        )  # Perform detection on the frame
        for detection in detections:
            boxes = detection.boxes.xyxy
            scores = detection.boxes.conf
            classes = detection.boxes.cls
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = coco_model.names[int(cls)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
        cv2.imshow("YOLOv11 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
