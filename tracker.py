import os
import torch
import cv2
import numpy as np
import math

# Define the POINTS function


def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for obj_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 10:  # Increase the distance threshold
                    self.center_points[obj_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, obj_id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, obj_id = obj_bb_id
            center = self.center_points[obj_id]
            new_center_points[obj_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids


# Define the size threshold
threshold_size = 1000  # Adjust this value based on your scenario

cv2.namedWindow('ROI')
cv2.setMouseCallback('ROI', POINTS)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
count = 0
cap = cv2.VideoCapture('tvid.mp4')
tracker = Tracker()

output_dir = 'rider_images'
os.makedirs(output_dir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))

    results = model(frame)
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d = (row['name'])

        if 'motorcycle' in d:
            y1 = max(0, y1 - 20)
            y2 = min(frame.shape[0], y2 + 20)
            x1 = max(0, x1 - 20)
            x2 = min(frame.shape[1], x2 + 20)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, str(d), (x1, y1),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

            # Check the size of the bounding box
            if (x2 - x1) * (y2 - y1) > threshold_size:
                y2 += 50
                x2 += 50

                y2 = min(frame.shape[0], y2)
                x2 = min(frame.shape[1], x2)

                cropped_vehicle = frame[y1:y2, x1:x2]

                resized_image = cv2.resize(cropped_vehicle, (640, 480))

                image_filename = os.path.join(
                    output_dir, f'cropped_vehicle_{count}.jpg')
                cv2.imwrite(image_filename, resized_image)

    cv2.imshow("ROI", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
