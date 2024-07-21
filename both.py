import os
import torch
import cv2
import sys
import numpy as np
from ultralytics import YOLO
import pandas as pd

# Load YOLO model for helmet detection
helmet_model = YOLO('best.pt')

# Load YOLOv5 model for two-wheeler detection
two_wheeler_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cv2.namedWindow('ROI')

video_file = sys.argv[1]  # Video file name passed as command-line argument
cap = cv2.VideoCapture(video_file)
my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

    # Skip frames, 
    if count % 5 != 0:
        continue
    frame = cv2.resize(frame, (1020, 600))

    # Two-wheeler detection
    results_two_wheeler = two_wheeler_model(frame)
    
    for index, row in results_two_wheeler.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])   #`row = Series({'xmin': 100, 'ymin': 150, 'xmax': 200, 'ymax': 300, 'confidence': 0.85, 'class': 3, 'name': 'motorcycle'})
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d = row['name']

        if 'motorcycle' in d:
            y1 = max(0, y1 - int(0.5 * (y2 - y1)))
            y2 = min(frame.shape[0], y2 + 5)
            x1 = max(0, x1 - 5)
            x2 = min(frame.shape[1], x2 + 5)

            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.putText(frame, str("Two-Wheeler Vehicle"), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

            # Cropping the two-wheeler region
            cropped_vehicle = frame[y1:y2, x1:x2]

            # Helmet detection on the cropped two-wheeler region
            results_helmet = helmet_model.predict(cropped_vehicle)
            a = results_helmet[0].boxes.data
            px = pd.DataFrame(a.cpu().numpy()).astype("float")

            for _, row in px.iterrows():
                x1_helmet = int(row[0])
                y1_helmet = int(row[1])
                x2_helmet = int(row[2])
                y2_helmet = int(row[3])
                d_helmet = int(row[5])
                c_helmet = class_list[d_helmet]

                # Conditionally set colors based on class
                if c_helmet == "With Helmet":
                    # Green bounding box, white text color with bold font
                    cv2.rectangle(frame, (x1 + x1_helmet, y1 + y1_helmet), (x1 + x2_helmet, y1 + y2_helmet), (0, 255, 0), 2)
                    cv2.putText(frame, f'{c_helmet}', (x1 + x1_helmet, y1 + y1_helmet),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                elif c_helmet == "Without Helmet":
                    # Red bounding box, white text color with bold font
                    cv2.rectangle(frame, (x1 + x1_helmet, y1 + y1_helmet), (x1 + x2_helmet, y1 + y2_helmet),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, f'{c_helmet}', (x1 + x1_helmet, y1 + y1_helmet),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("ROI", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc key
        break

cap.release()
cv2.destroyAllWindows()
