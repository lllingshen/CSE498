# Webcam Object Recognition with YOLOv11

This project provides a simple implementation of object detection from a webcam using the YOLO (You Only Look Once) model, specifically YOLOv11.

## Overview

The application captures video from your webcam and performs real-time object detection using a pre-trained YOLOv11 model. Detected objects are highlighted with bounding boxes and labeled with their class name and confidence score.

## Features

- Real-time object detection from webcam feed
- Display of bounding boxes around detected objects
- Display of object class names and confidence scores
- Simple interface with keyboard control to exit the application

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLO
- Webcam access

## How It Works

The main script (`main.py`) performs the following operations:

1. Initializes the YOLO model by loading the pre-trained weights
2. Opens a connection to the webcam
   - For each frame captured:
   - Performs object detection using the YOLO model
   - Draws bounding boxes around detected objects
   - Displays class names and confidence scores
   - Shows the processed frame in a window
3. The application can be closed by pressing 'q'

## Usage

To run the application:

`python main.py`

Press 'q' to exit the application while the webcam feed is active.

## Model Information

The project uses YOLOv11n, a lightweight version of the YOLOv11 object detection model trained on the COCO dataset, capable of detecting a wide range of common objects.

## License

This project uses the AGPL-3.0 License, as specified by the Ultralytics YOLO package.
