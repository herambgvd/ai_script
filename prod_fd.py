import base
import os
import numpy as np
import cv2
import json
import requests
import base64
from face_utils.detection import SCRFD

# ✅ Load SCRFD Model with GPU Support
def load_scrfd_model(model_path):
    scrfd = SCRFD(model_file=model_path)
    scrfd.prepare(ctx_id=0, input_size=(640, 640))  # Adjust input size as needed
    print("✅ SCRFD Model Loaded with Tracking Enabled")
    return scrfd

# ✅ Function to Convert Cropped Face to Base64
def encode_face_to_base64(frame, x1, y1, x2, y2, padding=20, target_size=(128, 128)):
    """Crops the detected face region with padding, resizes to a fixed size, and encodes it to base64."""
    try:
        h, w, _ = frame.shape
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(w, x2 + padding), min(h, y2 + padding)

        cropped_face = frame[y1:y2, x1:x2]

        if cropped_face.shape[0] < 64 or cropped_face.shape[1] < 64:
            return None  # Skip small faces

        resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_LINEAR)
        _, buffer = cv2.imencode(".jpg", resized_face)
        return base64.b64encode(buffer).decode("utf-8")  
    
    except Exception as e:
        print(f"❌ Error encoding face image: {e}")
        return None

# ✅ Process Stream with Tracking and ROI-based Detection
def process_stream(cap, out, ai_model, cam_id, cam_name, roi, producer):
    """
    Process video frames using SCRFD, but detect faces **only within the ROI**.
    """
    x1_roi, y1_roi, x2_roi, y2_roi = roi  # Extract ROI coordinates
    COLORS = np.random.randint(0, 255, size=(100, 3), dtype="uint8")

    API_ENDPOINT = "http://0.0.0.0:8080/api/v1/fd/event"
    KAFKA_TOPIC = "face_detection"
    CONFIDENCE_THRESHOLD = 0.60  

    alerted_track_ids = set()
    TRACKING_BUFFER = {}  # Store recent face positions for tracking stability

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            base.logging.error("❌ Stream capture failed")
            break

        frame = cv2.resize(frame, (640, 480))
        copied_frame = frame.copy()

        # ✅ Draw ROI Bounding Box
        cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 0), 2)  

        # ✅ Extract ROI from the frame
        roi_frame = frame[y1_roi:y2_roi, x1_roi:x2_roi].copy()

        # ✅ Run SCRFD Face Detection on **ROI only**
        detections, img_info, bboxes, _ = ai_model.detect_tracking(image=roi_frame, thresh=0.5)

        for i in range(len(bboxes)):
            try:
                # ✅ Extract bounding box coordinates and track ID (adjusting to full frame coordinates)
                x1, y1, x2, y2, track_id = map(int, bboxes[i])
                x1, x2 = x1 + x1_roi, x2 + x1_roi
                y1, y2 = y1 + y1_roi, y2 + y1_roi

                # ✅ Extract confidence score
                confidence_score = detections[i][-1].item()
                conf = int(float(confidence_score) * 100)  # Convert to percentage

                # ✅ Stabilize tracking using a buffer
                if track_id in TRACKING_BUFFER:
                    prev_x1, prev_y1, prev_x2, prev_y2 = TRACKING_BUFFER[track_id]
                    if abs(prev_x1 - x1) < 15 and abs(prev_y1 - y1) < 15:  # Small movement allowed
                        track_id = track_id  # Keep the same ID
                    else:
                        track_id = int(track_id)  # Reset tracking if large displacement detected
                TRACKING_BUFFER[track_id] = (x1, y1, x2, y2)  # Store latest position

                # ✅ Check if Face is Fully Inside ROI & Confidence > Threshold
                if conf > CONFIDENCE_THRESHOLD and track_id not in alerted_track_ids:
                    # ✅ Properly Crop the Face with Padding
                    face_base64 = encode_face_to_base64(copied_frame, x1, y1, x2, y2, padding=30)

                    if face_base64:
                        payload = {
                            "cam_name": cam_name,
                            "cam_id": cam_id,
                            "alert_frame": face_base64,  
                        }

                        # ✅ Send Data to API Endpoint
                        try:
                            response = requests.post(API_ENDPOINT, json=payload)
                            if response.status_code != 200:
                                base.logging.error(f"❌ API Error: {response.status_code} - {response.text}")
                        except Exception as e:
                            base.logging.error(f"❌ Failed to send data to API: {e}")

                        # ✅ Publish to Kafka Topic
                        try:
                            producer.produce(KAFKA_TOPIC, json.dumps(payload).encode("utf-8"))
                            producer.flush()
                        except Exception as kafka_ex:
                            base.logging.error(f"❌ Kafka Error: {kafka_ex}")

                        # ✅ Mark as Alerted
                        alerted_track_ids.add(track_id)

                # ✅ Draw Bounding Box with Track ID
                color = [int(c) for c in COLORS[track_id % len(COLORS)]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID {track_id} ({conf}%)"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            except Exception as e:
                base.logging.error(f"❌ Error processing face detection: {e}")
                continue

        # ✅ Stream processed frame to RTMP
        if out:
            out.write(frame)

        # ✅ Show the inference window
        cv2.imshow("Face Detection (SCRFD)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()



# ✅ Main Process
def main(rtsp, rtmp, cam_name, cam_id, roi, model_path):
    producer = None
    try:
        # Setup Kafka Producer
        producer = base.get_kafka_producer()
        if model_path == "fd":
            MODEL_PATH = os.getenv('FD_PATH')

        # Load The Model with GPU Acceleration
        ai_model = load_scrfd_model(MODEL_PATH)

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

if __name__ == "__main__":
    parser = base.argparse.ArgumentParser(description="Real-time Face Detection with ROI and Tracking (SCRFD, RTSP, RTMP)")
    parser.add_argument("--rtmp", type=str, required=True, help="RTMP output URL for broadcasting")
    parser.add_argument("--rtsp", type=str, required=True, help="RTSP URL of the camera")
    parser.add_argument("--cam_name", type=str, required=True, help="Camera Name")
    parser.add_argument("--cam_id", type=str, required=True, help="Camera ID")
    parser.add_argument("--roi", type=int, nargs=4, required=True, help="ROI coordinates in format x1 y1 x2 y2")
    parser.add_argument("--model_path", type=str, required=True, help="Custom Model Path")
    args = parser.parse_args()

    main(args.rtsp, args.rtmp, args.cam_name, args.cam_id, args.roi, args.model_path)
