import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, 'videos')
OUT_DIR = os.path.join(PROJECT_DIR, 'output')

os.makedirs(OUT_DIR, exist_ok=True)

USE_WEBCAM = True
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    video_path = os.path.join(VIDEO_DIR, 'name')
    cap = cv2.VideoCapture(video_path)

model = YOLO('yolov8s.pt')

CONF_THRESHOLD = 0.5

RESIZE_WIDTH = 960

prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_w = RESIZE_WIDTH
        new_h = int(h * scale)

        frame = cv2.resize(frame, (new_w, new_h))

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    people_count = 0
    pseudo_id = 0

    PERSON_CLASS_ID = 0

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == PERSON_CLASS_ID:
                people_count += 1
                pseudo_id += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f'id: {pseudo_id} conf: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'People count: {people_count}', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 255), 2)
    cv2.imshow('YOLO', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break