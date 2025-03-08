import base
import requests
import cv2
import base64
import json
import numpy as np

# ✅ Function to Convert Frame to Base64
def encode_frame_to_base64(frame):
    """
    Encodes an OpenCV frame (numpy array) to a base64 string.
    """
    _, buffer = cv2.imencode(".jpg", frame)  # Convert frame to JPEG
    return base64.b64encode(buffer).decode("utf-8")  # Encode and decode to string

# ✅ Function to Draw Only Corner Rectangles
def draw_corner_rect(img, x1, y1, x2, y2, color=(0, 255, 0), thickness=2, corner_length=15):
    """
    Draws only the corners of a bounding box.
    """
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
    Process video frames to detect and track objects within the defined ROI, stream to RTMP, 
    and generate payloads with tracking data.
    """
    x1_roi, y1_roi, x2_roi, y2_roi = roi  # Extract ROI coordinates
    COLORS = np.random.randint(0, 255, size=(80, 3), dtype="uint8")  # Define bounding box colors

    API_ENDPOINT = "http://0.0.0.0:8080/box/event"  # Replace with actual API endpoint
    KAFKA_TOPIC = "box"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.resize(frame, (640, 480))

        # Extract the ROI region
        roi_frame = frame[y1_roi:y2_roi, x1_roi:x2_roi].copy()

        # Run YOLO detection with tracking enabled
        results = ai_model.track(roi_frame, persist=True)

        # Process detected objects
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class index
                track_id = int(box.id[0].item()) if box.id is not None else None  # Tracking ID

                # Adjust coordinates relative to the full frame
                x1 += x1_roi
                y1 += y1_roi
                x2 += x1_roi
                y2 += y1_roi

                # Assign a color to each class
                color = [int(c) for c in COLORS[cls % len(COLORS)]]

                # Draw only corner rectangles
                draw_corner_rect(frame, x1, y1, x2, y2, color)

                # Draw tracking ID and confidence score
                label = f"ID {track_id} | {ai_model.names[cls]}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Extract the bounding box image
                box_crop = frame[y1:y2, x1:x2].copy()
                box_base64 = encode_frame_to_base64(box_crop)

                # ✅ Prepare Payload
                payload = {
                    "cam_name": cam_name,
                    "cam_id": cam_id,
                    "tracking_id": track_id,
                    "box_base64": box_base64
                }

                # ✅ Send Data to API Endpoint
                try:
                    response = requests.post(API_ENDPOINT, json=payload)
                    if response.status_code != 200:
                        print(f"❌ API Error: {response.status_code} - {response.text}")
                except Exception as e:
                    print(f"❌ Failed to send data to API: {e}")

                # ✅ Publish to Kafka Topic
                try:
                    producer.produce(KAFKA_TOPIC, json.dumps(payload).encode("utf-8"))
                    producer.flush()
                except Exception as kafka_ex:
                    print(f"❌ Kafka Error: {kafka_ex}")

        # Stream processed frame to RTMP
        if out:
            out.write(frame)

        # Show the inference window
        cv2.imshow("ROI Tracking", frame)
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
        error_msg = f"Main system error in camera {cam_name}: {e}"
        if producer:
            try:
                producer.produce("notifications", error_msg.encode('utf-8'))
                producer.flush()
            except Exception as kafka_ex:
                base.logging.error(f"❌ Failed to produce error message to Kafka: {kafka_ex}")

if __name__ == "__main__":
    parser = base.argparse.ArgumentParser(
        description="Real-time Box Detection with ROI (using YOLOv8, RTSP, and RTMP)"
    )
    parser.add_argument("--rtmp", type=str, required=True, help="RTMP output URL for broadcasting")
    parser.add_argument("--rtsp", type=str, required=True, help="RTSP URL of the camera")
    parser.add_argument("--cam_name", type=str, required=True, help="Camera Name")
    parser.add_argument("--cam_id", type=str, required=True, help="Camera ID")
    parser.add_argument("--roi", type=int, nargs=4, required=True, help="ROI coordinates in format x1 y1 x2 y2")
    parser.add_argument("--model_path", type=str, required=True, help="Custom Model Path")
    args = parser.parse_args()

    main(
        rtsp=args.rtsp,
        rtmp=args.rtmp,
        cam_name=args.cam_name,
        cam_id=args.cam_id,
        roi=args.roi,
        model_path=args.model_path
    )
