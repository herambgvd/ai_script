import base
import requests
import cv2
import base64
import json
import numpy as np

# ✅ Function to Convert Full Frame to Base64
def encode_frame_to_base64(frame):
    """Encodes an OpenCV frame (numpy array) to a base64 string."""
    try:
        _, buffer = cv2.imencode(".jpg", frame)  # Convert frame to JPEG
        return base64.b64encode(buffer).decode("utf-8")  # Encode and decode to string
    except Exception as e:
        base.logging.error(f"❌ Error encoding frame: {e}")
        return None

# ✅ Function to Draw Only Corner Rectangles
def draw_corner_rect(img, x1, y1, x2, y2, color=(0, 255, 0), thickness=2, corner_length=15):
    """Draws only the corners of a bounding box."""
    # Top-left corner
    cv2.line(img, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_length), color, thickness)

    # Top-right corner
    cv2.line(img, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_length), color, thickness)

    # Bottom-left corner
    cv2.line(img, (x1, y2), (x1 + corner_length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_length), color, thickness)

    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2 - corner_length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_length), color, thickness)

# ✅ Process Stream and Broadcast Inference
def process_stream(cap, out, ai_model, cam_id, cam_name, roi, producer):
    """
    Process video frames to detect "Door-Open" and "Door-Close" states within the ROI,
    stream to RTMP, and generate an alert only when the door opens.
    """
    x1_roi, y1_roi, x2_roi, y2_roi = roi  # Extract ROI coordinates
    API_ENDPOINT = "http://0.0.0.0:8080/api/v1/door/event"  # Replace with actual API endpoint
    KAFKA_TOPIC = "door"
    CONFIDENCE_THRESHOLD = 0.70  # ✅ Only send an alert if confidence > 70%

    # ✅ Class mappings for Door Open/Close
    CLASS_LABELS = {0: "Door-Open", 1: "Door-Close"}  # Adjust index as per your model's output

    # ✅ Flag to track the door state (Initially assumed closed)
    door_open = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            base.logging.error("❌ Stream capture failed at frame")
            break

        # ✅ Resize frame to 640x480 for processing
        frame = cv2.resize(frame, (640, 480))

        # ✅ Draw ROI Bounding Box on Full Frame
        cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 0), 2)  # Blue box for ROI

        # ✅ Extract ROI region for processing
        roi_frame = frame[y1_roi:y2_roi, x1_roi:x2_roi].copy()

        # ✅ Run YOLO detection
        results = ai_model(roi_frame, verbose=False)  # Run inference
        detected_label = None  # Track the detected class

        # ✅ Ensure results[0] exists and has boxes
        if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                conf = float(box.conf[0])  # Get confidence score
                cls = int(box.cls[0])  # Get class index

                if cls in CLASS_LABELS and conf > CONFIDENCE_THRESHOLD:
                    detected_label = CLASS_LABELS[cls]

                    # ✅ Draw bounding box on the main frame (not just ROI)
                    cv2.rectangle(
                        frame, 
                        (x1_roi + x1, y1_roi + y1), 
                        (x1_roi + x2, y1_roi + y2), 
                        (0, 255, 0), 2
                    )

                    # ✅ Display class label
                    cv2.putText(
                        frame,
                        f"{detected_label} ({conf*100:.1f}%)",
                        (x1_roi + x1, y1_roi + y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,  # Font scale
                        (0, 255, 0) if detected_label == "Door-Open" else (0, 0, 255),  # Green for open, red for close
                        2,  # Thickness
                        cv2.LINE_AA
                    )
                    break  # Only process the first valid detection

        # ✅ Add "Door Status" text at the top-left of the frame
        cv2.putText(
            frame,
            f"Door Status: {detected_label or 'Close'}",
            (20, 40),  # Position at the top-left
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,  # Font scale
            (0, 255, 0) if detected_label == "Door-Open" else (0, 0, 255),  # Green for open, red for close
            2,  # Thickness
            cv2.LINE_AA
        )

        # ✅ If the detected class is "Door-Open" and was previously closed, send an alert
        if detected_label == "Door-Open" and not door_open:
            base.logging.info("🚪 Door Open Detected! Sending alert...")
            
            # Prepare payload
            payload = {
                "camera_id": cam_id,
                "camera_name": cam_name,
                "status": "open",
                "alert_frame": encode_frame_to_base64(frame)
            }

            # ✅ Send to Kafka
            producer.send(KAFKA_TOPIC, json.dumps(payload).encode("utf-8"))

            # ✅ Send to API
            try:
                requests.post(API_ENDPOINT, json=payload, timeout=5)
            except requests.exceptions.RequestException as e:
                base.logging.error(f"❌ API request failed: {e}")

            door_open = True  # Update state to avoid duplicate alerts

        # ✅ If "Door-Close" is detected, reset flag
        if detected_label == "Door-Close" and door_open:
            base.logging.info("🚪 Door Closed Detected.")
            door_open = False  # Reset state

        # ✅ Stream processed frame to RTMP
        if out:
            out.write(frame)

        # ✅ Show the inference window
        # cv2.imshow("ROI Tracking", frame)
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
    parser = base.argparse.ArgumentParser(description="Real-time Box Detection with ROI (YOLOv8, RTSP, RTMP)")
    parser.add_argument("--rtmp", type=str, required=True, help="RTMP output URL for broadcasting")
    parser.add_argument("--rtsp", type=str, required=True, help="RTSP URL of the camera")
    parser.add_argument("--cam_name", type=str, required=True, help="Camera Name")
    parser.add_argument("--cam_id", type=str, required=True, help="Camera ID")
    parser.add_argument("--roi", type=int, nargs=4, required=True, help="ROI coordinates in format x1 y1 x2 y2")
    parser.add_argument("--model_path", type=str, required=True, help="Custom Model Path")
    args = parser.parse_args()

    main(args.rtsp, args.rtmp, args.cam_name, args.cam_id, args.roi, args.model_path)
