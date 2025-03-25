from flask import Blueprint, jsonify, request, Response
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

detect_endpoint = Blueprint('detect_endpoint', __name__)

model = YOLO('models/yolov11s.pt')

def convert_rgba_to_rgb(image_file):
    """Converts an RGBA image to RGB."""
    img = Image.open(io.BytesIO(image_file.read()))
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    return np.array(img)

def perform_yolov5_detection(image_np):
    """Performs YOLOv5 object detection on an image."""
    results = model(image_np)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'label': class_name,
            })
    return detections

@detect_endpoint.route("/detect_image", methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    try:
        image_np = convert_rgba_to_rgb(image_file)
        detections = perform_yolov5_detection(image_np)
        return jsonify(detections)

    except Exception as e:
        return jsonify({'error': str(e)}), 500