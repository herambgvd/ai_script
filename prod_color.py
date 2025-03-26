import cv2
import base64
import json
import threading
import time
import requests
import base
import numpy as np

# ✅ Define Target Classes
TARGET_CLASSES = {"person", "handbag", "backpack", "suitcase"}

# ✅ Store Processed Tracking IDs
processed_bag_ids = set()  
person_data = {}  

# ✅ Load Clothing Model Asynchronously
clothing_model = None
clothing_model_ready = False

def load_clothing_model():
    """Preloads the clothing model asynchronously at startup."""
    global clothing_model, clothing_model_ready
    if clothing_model is None:
        clothing_model = base.load_clothing_model()
        clothing_model_ready = True

# Load clothing model asynchronously at startup
threading.Thread(target=load_clothing_model, daemon=True).start()

# ✅ Function to Draw Corner Bounding Box
def draw_corner_box(image, x1, y1, x2, y2, color=(255, 0, 0), thickness=2, corner_length=15):
    """Draws a corner-style bounding box on the given image."""
    # Top-left corner
    cv2.line(image, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_length), color, thickness)

    # Top-right corner
    cv2.line(image, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x1, y2), (x1 + corner_length, y2), color, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - corner_length), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x2, y2), (x2 - corner_length, y2), color, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - corner_length), color, thickness)

# ✅ Function to Encode Frame to Base64
def encode_frame_to_base64(frame, resize_shape=None):
    """Encodes frame to Base64 with optional resizing."""
    if resize_shape:
        frame = cv2.resize(frame, resize_shape)
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

# ✅ Function to Extract Dominant Color using Mode Cut
def get_dominant_color(image):
    """Extracts the dominant color using Mode Cut technique."""
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        return None, None  # Return None if image is empty

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    pixels = image.reshape(-1, 3)

    # Apply K-Means to find the dominant color
    k = 3
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    dominant_color = centers[np.argmax(np.bincount(labels.flatten()))].astype(int)
    
    r, g, b = int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2])
    hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)

    return (r, g, b), hex_color

# ✅ Function to Send Alerts to API & Kafka
def send_alert(alert_msg, producer, topic, api_url):
    try:
        # ✅ Convert `details` to a JSON string
        alert_msg["details"] = json.dumps(alert_msg["details"])  

        # ✅ Send to API
        response = requests.post(api_url, json=alert_msg)
        if response.status_code != 200:
            print(f"❌ Failed to send alert to API: {response.status_code} - {response.text}")

        # ✅ Send to Kafka
        if producer:
            producer.produce(topic, json.dumps(alert_msg).encode('utf-8'))
            producer.flush()

    except Exception as e:
        print(f"❌ Failed to send alert: {e}")


# ✅ Function to Process RTSP Stream and Send Alerts
def process_stream(cap, out, ai_model, cam_id, cam_name, roi, producer):
    api_url = "http://0.0.0.0:8080/api/v1/color/event"
    roi_x1, roi_y1, roi_x2, roi_y2 = roi[:4]  

    while not clothing_model_ready:
        time.sleep(2)  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame = cv2.resize(frame, (640, 480))
            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            results = ai_model.track(roi_frame, persist=True, verbose=False)

            # ✅ Draw ROI Rectangle on the inference frame
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

            if len(results) > 0 and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy()
                detections = results[0].boxes.data.cpu().numpy()

                for i, row in enumerate(detections):
                    x1, y1, x2, y2 = map(int, row[:4])
                    conf = float(row[5])
                    cls = int(row[6])
                    track_id = int(track_ids[i])
                    class_name = ai_model.names[cls]

                    if class_name not in TARGET_CLASSES:
                        continue

                    obj_frame = frame[y1:y2, x1:x2]

                    # ✅ Draw Bounding Box on the full frame
                    draw_corner_box(frame, x1, y1, x2, y2, color=(255, 0, 0), thickness=1)
                    cv2.putText(frame, f"{class_name}", (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # ✅ Encode Frames
                    alert_frame = encode_frame_to_base64(obj_frame)  # Alert frame (No bounding boxes)
                    full_frame = encode_frame_to_base64(frame, resize_shape=(320, 240))  # Full frame (With bounding boxes)

                    # ✅ Process Person Detection
                    if class_name == "person" and conf > 0.80:
                        if track_id in person_data:
                            continue  

                        clothing_results = clothing_model(obj_frame, conf=0.4, iou=0.4, verbose=False)

                        upper_wear, lower_wear = None, None

                        if clothing_results and clothing_results[0].boxes is not None:
                            clothing_boxes = clothing_results[0].boxes.xyxy.cpu().numpy()
                            clothing_classes = clothing_results[0].boxes.cls.cpu().numpy()

                            for j, clothing_box in enumerate(clothing_boxes):
                                cx1, cy1, cx2, cy2 = map(int, clothing_box)
                                clothing_label = clothing_model.names[int(clothing_classes[j])]

                                if clothing_label in ["long sleeve top", "short sleeve top", "jacket"]:
                                    upper_wear = obj_frame[cy1:cy2, cx1:cx2]
                                elif clothing_label in ["trousers", "shorts", "skirt"]:
                                    lower_wear = obj_frame[cy1:cy2, cx1:cx2]

                        # ✅ Get Dominant Colors
                        upper_rgb, upper_hex = get_dominant_color(upper_wear) if upper_wear is not None else (None, None)
                        lower_rgb, lower_hex = get_dominant_color(lower_wear) if lower_wear is not None else (None, None)

                        # ✅ Skip alert if no valid clothing detection
                        if upper_rgb is None or lower_rgb is None:
                            continue

                        alert_msg = {
                            "cam_name": cam_name,
                            "cam_id": cam_id,
                            "alert_frame": alert_frame,
                            "full_frame": full_frame,
                            "details": {
                                "category": class_name,
                                "upper_wear": {"rgb": upper_rgb, "hex": upper_hex},
                                "lower_wear": {"rgb": lower_rgb, "hex": lower_hex}
                            }
                        }
                        send_alert(alert_msg, producer, "suspect", api_url)
                        person_data[track_id] = alert_msg 
                    elif class_name in ["handbag", "backpack", "suitcase"]:
                        if track_id in processed_bag_ids:
                            continue  

                        obj_rgb, obj_hex = get_dominant_color(obj_frame)

                        if obj_rgb is None:
                            continue

                        alert_msg = {
                            "cam_name": cam_name,
                            "cam_id": cam_id,
                            "alert_frame": alert_frame,
                            "full_frame": full_frame,
                            "details": {
                                "category": class_name,
                                "rgb": obj_rgb,
                                "hex": obj_hex
                            }
                        }
                        send_alert(alert_msg, producer, "suspect", api_url)
                        processed_bag_ids.add(track_id) 

            out.write(frame)
            # cv2.imshow("Inference", frame)
            # if cv2.waitKey(1) & 0xFF == 27:
                # break

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue

    cap.release()
    out.release()
    # cv2.destroyAllWindows()


# ✅ Main Process
def main(rtsp, rtmp, cam_name, cam_id, roi):
    producer = None
    try:
        # Setup Kafka Producer
        producer = base.get_kafka_producer()

        # Load The Model
        ai_model = base.load_model()

        # Setup Live Capture
        cap = base.initialize_capture(rtsp_url=rtsp)
        if cap is None:
            base.logging.error(f"❌ Exiting: Unable to open RTSP stream for camera {cam_name} (ID: {cam_id})")
            return

        # Setup Live for RTMP streaming
        out = base.setup_live_broadcasting(rtmp_url=rtmp)

        # Process Frames and Stream to RTMP
        process_stream(cap, out, ai_model, cam_id, cam_name, roi, producer)
        base.logging.info("Processing and streaming completed.")
    
    except Exception as e:
        base.logging.error(f"Error in main system: {e}")
        error_msg = f"Main system error in camera {cam_name}: {e}"
        if producer:
            try:
                producer.produce("notifications", error_msg.encode('utf-8'))
                producer.flush()
            except Exception as kafka_ex:
                base.logging.error(f"❌ Failed to produce error message to Kafka: {kafka_ex}")


if __name__ == "__main__":
    parser = base.argparse.ArgumentParser(
        description="Real-time Clothes/Bag color recogn. with ROI (using YOLOv8, RTSP, and RTMP)"
    )
    parser.add_argument("--rtmp", type=str, required=True, help="RTMP output URL for broadcasting")
    parser.add_argument("--rtsp", type=str, required=True, help="RTSP URL of the camera")
    parser.add_argument("--cam_name", type=str, required=True, help="Camera Name")
    parser.add_argument("--cam_id", type=str, required=True, help="Camera ID")
    parser.add_argument("--roi", type=int, nargs=4, required=True, help="ROI coordinates in format x1 y1 x2 y2")
    args = parser.parse_args()

    main(
        rtsp=args.rtsp,
        rtmp=args.rtmp,
        cam_name=args.cam_name,
        cam_id=args.cam_id,
        roi=args.roi
    )
