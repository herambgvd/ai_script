import base
import requests
import cv2
import base64
import json
import numpy as np
from datetime import datetime

# ✅ Function to Convert Cropped Face to Base64
def encode_face_to_base64(frame, x1, y1, x2, y2, padding=30, target_size=(512,512)):
    """Crops the detected face region with padding, resizes to a fixed size, and encodes it to base64."""
    try:
        # Get frame dimensions
        h, w, _ = frame.shape

        # Apply padding while ensuring it stays within bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        # Crop the expanded face region
        cropped_face = frame[y1:y2, x1:x2]

        # Ensure the face has a minimum width & height
        min_size = 64  # Minimum size for width/height to avoid very small crops
        if cropped_face.shape[0] < min_size or cropped_face.shape[1] < min_size:
            return None  # Skip encoding if the face is too small

        # Resize to a fixed size (optional, can be adjusted)
        resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_LINEAR)

        # Convert to JPEG and encode as base64
        _, buffer = cv2.imencode(".jpg", resized_face)
        return base64.b64encode(buffer).decode("utf-8")  # Encode to string
    
    except Exception as e:
        print(f"❌ Error encoding face image: {e}")
        return None  # Return None if encoding fails


# ✅ Process Stream and Broadcast Inference (Face Detection)
def process_stream(cap, out, ai_model, cam_id, cam_name, roi, producer):
    """
    Process video frames to detect and track faces within the defined ROI, stream to RTMP, 
    and generate payloads with tracking data (ensuring only one alert per tracking ID).
    """
    x1_roi, y1_roi, x2_roi, y2_roi = roi  # Extract ROI coordinates
    COLORS = np.random.randint(0, 255, size=(80, 3), dtype="uint8")  # Define bounding box colors

    API_ENDPOINT = "http://0.0.0.0:8080/api/v1/fd/event"  # Replace with actual API endpoint
    KAFKA_TOPIC = "face_detection"
    CONFIDENCE_THRESHOLD = 0.80  # ✅ Send alert only if confidence > 70%

    # ✅ Track alerted IDs to avoid duplicate alerts
    alerted_track_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            base.logging.error("❌ Stream capture failed at frame")
            break

        frame = cv2.resize(frame, (640, 480))
        copied_frame = frame.copy()

        # ✅ Draw ROI Bounding Box on Full Frame
        cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 0), 2)  # Blue box for ROI

        # ✅ Extract ROI region for processing
        roi_frame = frame[y1_roi:y2_roi, x1_roi:x2_roi].copy()

        # ✅ Run Face Detection with Tracking
        results = ai_model.track(roi_frame, persist=True, verbose=False)

        # ✅ Track current IDs in the frame
        current_track_ids = set()

        # ✅ Process detected faces
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])  # Get bounding box coordinates
                    conf = float(box.conf[0].item())  # Confidence score
                    track_id = int(box.id[0].item()) if box.id is not None else -1  # Tracking ID fix

                    # ✅ Convert bounding box coordinates from ROI to full frame
                    x1 += x1_roi
                    y1 += y1_roi
                    x2 += x1_roi
                    y2 += y1_roi

                    # ✅ Track IDs currently detected
                    current_track_ids.add(track_id)

                    # ✅ Assign a color to each tracking ID
                    color = [int(c) for c in COLORS[track_id % len(COLORS)]]

                    # ✅ Draw bounding box and tracking info
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # label = f"ID {track_id} ({conf:.2f})"
                    label = f"{track_id}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # ✅ Check if face is fully inside ROI and confidence > 70%
                    if (
                        conf > CONFIDENCE_THRESHOLD and
                        track_id not in alerted_track_ids
                    ):
                        # ✅ Convert Detected Face to Base64
                        face_base64 = encode_face_to_base64(copied_frame, x1, y1, x2, y2, padding=30, target_size=(512, 512))

                        if face_base64:  # Ensure face image was properly encoded
                            # ✅ Prepare Payload
                            payload = {
                                "cam_name": cam_name,
                                "cam_id": cam_id,
                                "face_frame": face_base64,  # ✅ Sending only the face image
                            }

                            # # ✅ Send Data to API Endpoint
                            # try:
                            #     response = requests.post(API_ENDPOINT, json=payload)
                            #     if response.status_code != 200:
                            #         base.logging.error(f"❌ API Error: {response.status_code} - {response.text}")
                            # except Exception as e:
                            #     base.logging.error(f"❌ Failed to send data to API: {e}")

                            # ✅ Publish to Kafka Topic
                            try:
                                producer.produce(KAFKA_TOPIC, json.dumps(payload).encode("utf-8"))
                                producer.flush()
                            except Exception as kafka_ex:
                                base.logging.error(f"❌ Kafka Error: {kafka_ex}")

                            # ✅ Add this tracking ID to the set (Avoid duplicate alerts)
                            alerted_track_ids.add(track_id)

                except Exception as e:
                    base.logging.error(f"❌ Error processing face detection: {e}")
                    continue

        # ✅ Reset tracking IDs if they disappear from the frame
        alerted_track_ids.intersection_update(current_track_ids)

        # ✅ Stream processed frame to RTMP
        if out:
            out.write(frame)

        # ✅ Show the inference window
        # cv2.imshow("Face Detection", frame)
        # if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        #     break

    cap.release()
    out.release()
    # cv2.destroyAllWindows()


# ✅ Main Process
def main(rtsp, rtmp, cam_name, cam_id, roi, model_path):
    producer = None
    try:
        # Setup Kafka Producer
        producer = base.get_kafka_producer()

        # Load The Model
        ai_model = base.load_custom_model(model_path)

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
    parser = base.argparse.ArgumentParser(description="Real-time Face Detection with ROI (YOLOv8, RTSP, RTMP)")
    parser.add_argument("--rtmp", type=str, required=True, help="RTMP output URL for broadcasting")
    parser.add_argument("--rtsp", type=str, required=True, help="RTSP URL of the camera")
    parser.add_argument("--cam_name", type=str, required=True, help="Camera Name")
    parser.add_argument("--cam_id", type=str, required=True, help="Camera ID")
    parser.add_argument("--roi", type=int, nargs=4, required=True, help="ROI coordinates in format x1 y1 x2 y2")
    parser.add_argument("--model_path", type=str, required=True, help="Custom Model Path")
    args = parser.parse_args()

    main(args.rtsp, args.rtmp, args.cam_name, args.cam_id, args.roi, args.model_path)
