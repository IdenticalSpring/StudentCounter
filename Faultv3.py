import os
import glob
import torch
import cv2
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import time
import math

app = Flask(__name__)

model = YOLO('models/weights/last.pt')  #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

cap = None
current_frame = 0
playback_speed = 1.0
paused = False

walk_in_count = 0
walk_out_count = 0

os.makedirs('data/vipham', exist_ok=True)

violation_tracker = {}

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
            for id, data in self.center_points.items():
                dist = math.hypot(cx - data['center'][0], cy - data['center'][1])

                if dist < 35:
                    self.center_points[id]['center'] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = {'center': (cx, cy), 'last_position': None}
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]['center']
            new_center_points[object_id] = {'center': center, 'last_position': self.center_points[object_id]['last_position']}

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

tracker = Tracker()

def process_frame(frame):
    global walk_in_count, walk_out_count
    target_width = 1280
    target_height = 720
    frame = cv2.resize(frame, (target_width, target_height))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame_rgb)

    people_rects = []
    uniform_rects = []
    badge_rects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            print(f"Detected class: {class_name}")
            if class_name in ['people']:
                people_rects.append((x1, y1, x2, y2))
            elif class_name in ['dong phuc']:
                uniform_rects.append((x1, y1, x2, y2))
            elif class_name in ['the ten']:
                badge_rects.append((x1, y1, x2, y2))

    tracked_objects = tracker.update([(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in people_rects])

    line1_y = target_height // 2 - 20
    line2_y = target_height // 2 + 20
    cv2.line(frame, (0, line1_y), (target_width, line1_y), (0, 255, 255), 2)
    cv2.line(frame, (0, line2_y), (target_width, line2_y), (0, 255, 255), 2)

    for obj in tracked_objects:
        x, y, w, h, obj_id = obj
        label = f"ID {obj_id}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        center_y = y + h // 2
        if tracker.center_points[obj_id]['last_position'] is not None:
            last_position = tracker.center_points[obj_id]['last_position']
            if last_position <= line1_y and center_y > line1_y:
                walk_in_count += 1
                print(f"Walk In Count: {walk_in_count}")
            elif last_position >= line2_y and center_y < line2_y:
                walk_out_count += 1
                print(f"Walk Out Count: {walk_out_count}")

        tracker.center_points[obj_id]['last_position'] = center_y

        person_bbox = (x, y, x + w, y + h)
        violating_uniform = True

        has_uniform = any(is_overlapping(person_bbox, (ux1, uy1, ux2, uy2)) for ux1, uy1, ux2, uy2 in uniform_rects)
        has_badge = any(is_overlapping(person_bbox, (bx1, by1, bx2, by2)) for bx1, by1, bx2, by2 in badge_rects)

        if has_uniform and has_badge:
            if obj_id in violation_tracker:
                print(f"ID {obj_id}: no longer violating")
                delete_violation_images(obj_id)
                del violation_tracker[obj_id]
            print(f"ID {obj_id}: has both đồng phục and thẻ tên")
            continue
        else:
            violating_uniform = True

        if violating_uniform:
            if obj_id not in violation_tracker:
                violation_tracker[obj_id] = 0

            if violation_tracker[obj_id] < 10:
                print(f"ID {obj_id}: vi phạm đồng phục")
                save_cropped_image(frame, person_bbox, obj_id)
                violation_tracker[obj_id] += 1

    for x1, y1, x2, y2 in uniform_rects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, 'dong phuc', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for x1, y1, x2, y2 in badge_rects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, 'the ten', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

def is_overlapping(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    ux1, uy1, ux2, uy2 = bbox2
    return not (x2 < ux1 or x1 > ux2 or y2 < uy1 or y1 > uy2)

def save_cropped_image(frame, bbox, obj_id):
    x1, y1, x2, y2 = bbox
    cropped_img = frame[y1:y2, x1:x2]
    cv2.imwrite(f'data/vipham/violation_{obj_id}_{time.time()}.jpg', cropped_img)

def delete_violation_images(obj_id):
    files = glob.glob(f'data/vipham/violation_{obj_id}_*.jpg')
    for file in files:
        os.remove(file)


def generate_frames(source):
    global cap, current_frame, playback_speed, paused
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Unable to open video source {source}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        if not paused:
            if current_frame >= total_frames:
                current_frame = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Unable to read frame at position {current_frame}")
                break

            frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print(f"Error: Unable to encode frame at position {current_frame}")
                break

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            current_frame += int(playback_speed)
            time.sleep(0.033 / playback_speed)
        else:
            time.sleep(0.1)

    cap.release()
    print("Video capture released.")

@app.route('/')
def index():
    videos = ['video/test1.mp4', 'video/test2.mp4', 'video/test3.mp4']
    return render_template('index.html', videos=videos)

@app.route('/video_feed')
def video_feed():
    source = request.args.get('source', default='video/test1.mp4')
    if source == 'webcam':
        source = 0
    return Response(generate_frames(source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():
    global paused, playback_speed, current_frame
    data = request.json
    action = data.get('action')
    if action == 'pause':
        paused = True
    elif action == 'play':
        paused = False
    elif action == 'slow':
        playback_speed = max(playback_speed - 0.5, 0.5)
    elif action == 'fast':
        playback_speed = min(playback_speed + 0.5, 4.0)
    elif action == 'backward':
        current_frame = max(current_frame - 300, 0)
    elif action == 'forward':
        current_frame += 300
    return jsonify(success=True)

@app.route('/count_info')
def count_info():
    global walk_in_count, walk_out_count
    return jsonify(walk_in_count=walk_in_count, walk_out_count=walk_out_count)

if __name__ == '__main__':
    app.run(debug=True)
