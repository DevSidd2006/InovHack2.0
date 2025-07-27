# ==============================================================================
# --- Imports and Initializations ---
# ==============================================================================

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file
import threading
import time
import queue
from collections import defaultdict, deque
import os
from dotenv import load_dotenv
import logging
import json
import requests
from ultralytics import YOLO
import torch
from datetime import datetime
import gc
import psutil

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# --- Configuration ---
# ==============================================================================

VIDEO_SOURCE = 0  # Default to live camera (OpenCV index 0)
DETECTION_INTERVAL = 1 # Process every frame for live video
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_QUEUE_SIZE = 10

# --- Feature Config ---
TRACKING_ENABLED = True
POSE_ESTIMATION = False
SEGMENTATION = False

# --- Analytics & Alerting Config ---
ALERT_THRESHOLDS = {'person': 10, 'car': 5} # Alert if count exceeds this
DWELL_TIME_THRESHOLD = 15 # seconds
ALERT_COOLDOWN = 20 # seconds

# --- OpenRouter LLM Config ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'
LLM_MODEL = "google/gemini-flash-1.5" # Using a fast and capable model

# ==============================================================================
# --- Global State & Data Structures ---
# ==============================================================================
analytics_lock = threading.Lock()

# --- Thread-safe Queues ---
frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
processed_frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)


# --- Data Storage ---
current_detections = defaultdict(int)
object_trajectories = defaultdict(lambda: deque(maxlen=50))
alert_history = deque(maxlen=100)
time_series_data = defaultdict(lambda: deque(maxlen=300))
zone_object_tracker = defaultdict(lambda: {'entry_time': None, 'last_seen': None, 'id': None})
last_alert_times = defaultdict(lambda: 0)

# --- Detection History (per second, for all features) ---
detection_history = deque(maxlen=3600)  # Store up to 1 hour of history (1 per second)


# --- LLM State ---
latest_llm_response = {'text': 'AI analysis is initializing...'}
latest_llm_error = {'error': ''}
# Store LLM insights history (list of dicts: {text, timestamp})
llm_insights_history = deque(maxlen=50)


# ==============================================================================
# --- YOLO Model Loading ---
# ==============================================================================
try:
    # Switch to YOLOv10 for faster inference. Download yolov10n.pt from https://github.com/ultralytics/ultralytics/releases/tag/v8.2.0 or the official YOLOv10 release page.
    model_name = 'yolov10n.pt'
    if POSE_ESTIMATION: model_name = 'yolov8n-pose.pt'  # YOLOv10 pose not yet standard; fallback to v8 pose if needed
    model = YOLO(model_name)
    model.fuse()
    model.to(DEVICE)
    logging.info(f"✅ YOLO Model '{model_name}' loaded successfully on device: {DEVICE}")
except Exception as e:
    logging.error(f"❌ Error loading YOLO model: {e}")
    exit(1)

# ==============================================================================
# --- LLM Communication ---
# ==============================================================================
def query_openrouter_llm(prompt, model=LLM_MODEL):
    """Sends a prompt to the OpenRouter LLM and returns the response text."""
    global latest_llm_error
    if not OPENROUTER_API_KEY or 'sk-or-v1-...' in OPENROUTER_API_KEY:
        error_msg = "OpenRouter API Key is not set or is a placeholder."
        logging.warning(error_msg)
        latest_llm_error['error'] = error_msg
        return None

    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json',
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an AI Agent that monitors the video screen and gives real-time insights from the video feed and you update the dashboard accordingly."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        result = response.json()
        latest_llm_error['error'] = ''
        return result['choices'][0]['message']['content']
    except requests.exceptions.HTTPError as http_err:
        error_msg = f"OpenRouter HTTP error {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        error_msg = f"OpenRouter connection exception: {e}"
    except (KeyError, IndexError) as e:
        error_msg = f"OpenRouter response format error: {e}"

    logging.error(error_msg)
    latest_llm_error['error'] = error_msg
    return None

# ==============================================================================
# --- Core Logic Functions ---
# ==============================================================================
def point_in_polygon(point, polygon):
    """Checks if a point is inside a given polygon using the ray-casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def detect_objects(frame):
    """Performs object detection on a frame using the loaded YOLO model."""
    try:
        if TRACKING_ENABLED:
            results = model.track(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, persist=True, tracker="bytetrack.yaml", verbose=False)
        else:
            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        return results[0]
    except Exception as e:
        logging.error(f"YOLO detection error: {e}")
        return None

def process_detections(results):
    """Extracts and formats detection data from YOLO results."""
    detected_objects = []
    if results is None or results.boxes is None:
        logging.debug("[process_detections] No results or boxes found.")
        return detected_objects

    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    track_ids = results.boxes.id.cpu().numpy() if TRACKING_ENABLED and results.boxes.id is not None else [None] * len(boxes)

    # Map YOLO class indices to names (update as per your model)
    CLASS_NAMES = model.names if hasattr(model, 'names') else {0: 'person', 1: 'car', 2: 'bus', 3: 'truck', 4: 'bicycle', 5: 'motorcycle', 6: 'knife', 7: 'gun'}

    for box, conf, cls, track_id in zip(boxes, confidences, classes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        label = CLASS_NAMES.get(int(cls), str(cls))
        det = {
            'bbox': (x1, y1, x2 - x1, y2 - y1),
            'center': (int(center_x), int(center_y)),
            'label': label,
            'confidence': float(conf),
            'track_id': int(track_id) if track_id is not None else None,
            'zones': [],  # For future zone logic
        }
        detected_objects.append(det)
        logging.debug(f"[process_detections] Detected: {det}")

    # --- Modular analytics stubs ---
    # Crowd detection: count people
    crowd_count = sum(1 for d in detected_objects if d['label'] == 'person')
    # Queue monitoring: TODO (cluster analysis)
    # Stampede: TODO (density/speed analysis)
    # Weapons: flag if 'gun' or 'knife' in detected labels
    weapons = [d for d in detected_objects if d['label'] in ['gun', 'knife']]
    # Traffic: count vehicles
    vehicle_count = sum(1 for d in detected_objects if d['label'] in ['car', 'bus', 'truck', 'motorcycle', 'bicycle'])
    # Parking: TODO (detect empty spots, optimize positions)

    # Attach analytics to detections if needed (for future frontend)
    # ...
    return detected_objects

def generate_alerts(detections):
    """Generates alerts based on detection counts, zone intrusions, and dwell times."""
    alerts = []
    current_time = time.time()

    def can_send_alert(alert_key):
        if current_time - last_alert_times.get(alert_key, 0) > ALERT_COOLDOWN:
            last_alert_times[alert_key] = current_time
            return True
        return False

    detection_counts = defaultdict(int)
    for det in detections:
        detection_counts[det['label']] += 1

    # --- Crowd detection alert ---
    crowd_count = detection_counts.get('person', 0)
    if crowd_count >= 30:  # Example threshold
        if can_send_alert('crowd_alert'):
            alerts.append({'type': 'Crowd', 'message': f'Crowd detected: {crowd_count} people.', 'severity': 'warning'})

    # --- Queue monitoring alert (stub) ---
    # TODO: Implement queue detection logic

    # --- Stampede alert (stub) ---
    # TODO: Implement stampede logic (density/speed)

    # --- Weapons detection alert ---
    for det in detections:
        if det['label'] in ['gun', 'knife']:
            if can_send_alert(f"weapon_{det['label']}"):
                alerts.append({'type': 'Weapon', 'message': f"Weapon detected: {det['label']}!", 'severity': 'high'})

    # --- Traffic management alert (stub) ---
    vehicle_count = sum(detection_counts[l] for l in ['car', 'bus', 'truck', 'motorcycle', 'bicycle'])
    if vehicle_count >= 20:  # Example threshold
        if can_send_alert('traffic_alert'):
            alerts.append({'type': 'Traffic', 'message': f'Heavy traffic: {vehicle_count} vehicles.', 'severity': 'warning'})

    # --- Emergency lane detection (stub) ---
    # TODO: Implement emergency lane logic

    # --- Smart parking alert (stub) ---
    # TODO: Implement parking logic

    # --- Existing zone/dwell/intrusion logic ---
    objects_in_zones_now = set()
    for det in detections:
        if not det['track_id']: continue
        for zone_name in det['zones']:
            object_key = f"{zone_name}_{det['track_id']}"
            objects_in_zones_now.add(object_key)
            tracker = zone_object_tracker[object_key]
            if tracker['entry_time'] is None:
                tracker['entry_time'] = current_time
                if can_send_alert(f"intrusion_{object_key}"):
                    alerts.append({'type': 'Intrusion', 'message': f"{det['label']} (ID:{det['track_id']}) entered {zone_name}.", 'severity': 'high'})
            dwell_time = current_time - tracker['entry_time']
            if dwell_time > DWELL_TIME_THRESHOLD:
                if can_send_alert(f"dwell_{object_key}"):
                    alerts.append({'type': 'Dwell Time', 'message': f"{det['label']} (ID:{det['track_id']}) dwelling in {zone_name}.", 'severity': 'warning'})

    for key in list(zone_object_tracker.keys()):
        if key not in objects_in_zones_now:
            del zone_object_tracker[key]

    for alert in alerts:
        alert['timestamp'] = datetime.now().isoformat()
    return alerts

def draw_visualizations(frame, detections):
    """Draws bounding boxes, labels, zones, and trajectories on the frame."""
    # Removed drawing of detection zones (no polygons or zone names)

    for det in detections:
        x, y, w, h = det['bbox']
        label, conf, track_id = det['label'], det['confidence'], det['track_id']
        color = (0, 0, 255) if any('Restricted' in z for z in det['zones']) else (0, 255, 0)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label_text = f"{label} {conf:.2f}" + (f" ID:{track_id}" if track_id is not None else "")
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if track_id and track_id in object_trajectories:
            points = np.array(list(object_trajectories[track_id]), dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)
    return frame

# ==============================================================================
# --- Background Threads ---
# ==============================================================================

def llm_monitoring_thread():
    """Periodically sends a snapshot of analytics to the LLM for insights."""
    global latest_llm_response
    while True:
        time.sleep(20)
        with analytics_lock:
            # Gather all relevant stats for LLM
            stats = dict(current_detections)
            person_count = stats.get('person', 0)
            car_count = stats.get('car', 0)
            bus_count = stats.get('bus', 0)
            truck_count = stats.get('truck', 0)
            bicycle_count = stats.get('bicycle', 0)
            motorcycle_count = stats.get('motorcycle', 0)
            knife_count = stats.get('knife', 0)
            gun_count = stats.get('gun', 0)
            total_objects = sum(stats.values())
            recent_critical_alerts = [a['message'] for a in list(alert_history)[:3] if a['severity'] in ['high', 'warning']]
            alert_summary = ". ".join(recent_critical_alerts) if recent_critical_alerts else "No critical alerts."
            video_source = VIDEO_SOURCE

        # Compose a detailed prompt for the LLM
        prompt = (
            f"Current time: {datetime.now().strftime('%I:%M %p')}. "
            f"Video source: {video_source}. "
            f"Detected: {person_count} people, {car_count} cars, {bus_count} buses, {truck_count} trucks, "
            f"{bicycle_count} bicycles, {motorcycle_count} motorcycles, {knife_count} knives, {gun_count} guns. "
            f"Total objects: {total_objects}. "
            f"Recent alerts: {alert_summary} "
            f"Please provide a concise, actionable summary for the dashboard user, highlighting any anomalies or safety issues."
        )
        llm_response = query_openrouter_llm(prompt)

        with analytics_lock:
            if llm_response:
                latest_llm_response['text'] = llm_response
                llm_insights_history.appendleft({
                    'text': llm_response,
                    'timestamp': datetime.now().isoformat()
                })
                logging.info(f"[LLM Insight] {llm_response}")
# --- LLM Q&A API: User can ask custom questions about analytics ---
@app.route('/llm_ask', methods=['POST'])
def llm_ask():
    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided.'}), 400
    # Compose analytics snapshot for context
    with analytics_lock:
        analytics_snapshot = {
            'current_detections': dict(current_detections),
            'total_objects': sum(current_detections.values()),
            'active_tracks': len(object_trajectories),
            'recent_alerts': list(alert_history)[:5],
        }
    prompt = f"System analytics: {json.dumps(analytics_snapshot)}\nUser question: {question}"
    llm_response = query_openrouter_llm(prompt)
    if llm_response:
        # Optionally add to insights history
        with analytics_lock:
            llm_insights_history.appendleft({
                'text': f"Q: {question}\nA: {llm_response}",
                'timestamp': datetime.now().isoformat()
            })
        return jsonify({'answer': llm_response})
    else:
        return jsonify({'error': latest_llm_error['error']}), 500

def video_capture_thread():
    """Captures video frames and puts them in a queue."""
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    try:
        if not cap.isOpened():
            logging.error(f"Could not open video source: {VIDEO_SOURCE}")
            return
        frame_count = 0
        while True:
            if isinstance(VIDEO_SOURCE, str) and cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_count += 1
            if frame_count % DETECTION_INTERVAL != 0:
                frame = None
                continue

            # Resize to 1280x720 for optimal YOLO performance
            resized_frame = cv2.resize(frame, (1280, 720))
            if not frame_queue.full():
                frame_queue.put(resized_frame)
            else:
                frame_queue.get()
                frame_queue.put(resized_frame)
            frame = None
            resized_frame = None

            # Periodic garbage collection and memory logging
            if frame_count % 100 == 0:
                gc.collect()
                mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                logging.info(f"[VideoCapture] Memory usage: {mem:.2f} MB")
            time.sleep(0.01)
    finally:
        cap.release()

def analytics_thread():
    """Processes frames, runs detection, and updates analytics."""
    global current_detections, object_trajectories, alert_history, time_series_data
    iter_count = 0
    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        logging.info("[analytics_thread] Processing new frame from queue.")
        results = detect_objects(frame)
        detections = process_detections(results)
        logging.info(f"[analytics_thread] Number of detections: {len(detections)}")
        if detections:
            logging.info(f"[analytics_thread] Detections: {detections}")
        else:
            logging.info("[analytics_thread] No detections found in this frame.")
        new_alerts = generate_alerts(detections)

        with analytics_lock:
            current_detections.clear()
            for det in detections:
                current_detections[det['label']] += 1
                if det['track_id']:
                    object_trajectories[det['track_id']].append(det['center'])
            for alert in new_alerts:
                alert_history.appendleft(alert)

            # --- Detection History (per second, for all features) ---
            # Save a snapshot of all current detections with timestamp
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'detections': dict(current_detections)
            }
            detection_history.append(snapshot)

        annotated_frame = draw_visualizations(frame.copy(), detections)
        if not processed_frame_queue.full():
            processed_frame_queue.put(annotated_frame)

        frame = None
        annotated_frame = None
        results = None
        detections = None

        iter_count += 1
        if iter_count % 100 == 0:
            gc.collect()
            mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            logging.info(f"[Analytics] Memory usage: {mem:.2f} MB")

        frame_queue.task_done()

# ==============================================================================
# --- Flask API Endpoints ---
# ==============================================================================
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                frame = processed_frame_queue.get(timeout=1)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except queue.Empty:
                continue
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/local_video')
def local_video():
    # Update the path if you want to serve a different file
    video_path = r"C:\Users\siddh\Cars Traffic - Stock Video! [CIs-q7eIpvE].webm"
    return send_file(video_path, mimetype='video/webm')


# --- LLM Insights API: latest and history ---
@app.route('/llm_insights')
def get_llm_insights():
    with analytics_lock:
        return jsonify({
            "text": latest_llm_response['text'],
            "error": latest_llm_error['error'],
            "history": list(llm_insights_history)
        })

@app.route('/analytics')
def get_analytics():
    with analytics_lock:
        return jsonify({'current_detections': dict(current_detections)})

@app.route('/alerts')
def get_alerts():
    with analytics_lock:
        return jsonify(list(alert_history))

# --- Set Video Source API ---
# --- Set Video Source API ---
@app.route('/set_video_source', methods=['POST'])
def set_video_source():
    global VIDEO_SOURCE
    data = request.get_json()
    url = data.get('video_url', '').strip()
    if not url:
        return jsonify({'success': False, 'error': 'No URL provided.'}), 400
    VIDEO_SOURCE = url
    logging.info(f"🔄 Video source updated to: {VIDEO_SOURCE}")
    return jsonify({'success': True, 'video_source': VIDEO_SOURCE})


# --- Detection History API ---
@app.route('/detection_history')
def get_detection_history():
    # Returns the detection history (per second) for all features
    with analytics_lock:
        return jsonify(list(detection_history))

# ==============================================================================
# --- Main Application Execution ---
# ==============================================================================
if __name__ == "__main__":
    logging.info("🚀 Starting Real-Time AI Analytics Dashboard...")
    
    threading.Thread(target=video_capture_thread, daemon=True, name="VideoCapture").start()
    threading.Thread(target=analytics_thread, daemon=True, name="Analytics").start()
    threading.Thread(target=llm_monitoring_thread, daemon=True, name="LLMMonitor").start()
    
    logging.info(f"✅ Dashboard running at: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
