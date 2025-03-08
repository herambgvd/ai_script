import base
import requests
import cv2
import base64
import json

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
def process_stream(cap, out, ai_model, cam_id, cam_name, roi, producer, threshold):
    """
       Process video frames to detect and track objects, stream to RTMP, and generate payloads for crowd counting.
    """
    base.logging.info(f"📌 Starting inference for Camera: {cam_name} (ID: {cam_id})")

    try:
        threshold_value = int(threshold)
    except ValueError:
        base.logging.error("❌ Threshold must be an integer. Exiting...")
        return

    api_endpoint = "http://0.0.0.0:8080/api/v1/crowd/event"
    last_alert_count = 0  # Store last alerted count to avoid duplicate alerts

    while True:
        ret, frame = cap.read()
        if not ret:
            base.logging.error("❌ Stream capture failed at frame")
            break
        
        try:
            frame = base.cv2.resize(frame, (640, 480))
            roi_x1, roi_y1, roi_x2, roi_y2 = roi[:4]
            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            base.cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 1)

            results = ai_model.track(roi_frame, persist=True, verbose=False)
            person_count = 0

            if len(results) > 0 and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy()
                detections = results[0].boxes.data.cpu().numpy()

                for i, row in enumerate(detections):
                    x1, y1, x2, y2 = map(int, row[:4])
                    conf = float(row[5])
                    cls = int(row[6])
                    track_id = int(track_ids[i])

                    if cls == 0:
                        person_count += 1
                        x1_full = roi_x1 + x1
                        y1_full = roi_y1 + y1
                        x2_full = roi_x1 + x2
                        y2_full = roi_y1 + y2
                        label = f'ID {track_id}'

                        draw_corner_rect(frame, x1_full, y1_full, x2_full, y2_full, (0, 0, 255), 2)
                        base.cv2.putText(frame, label, (x1_full, y1_full - 5), base.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            count_text = f"Person Count: {person_count}"
            base.cv2.putText(frame, count_text, (10, 30), base.cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            alert_frame_b64 = encode_frame_to_base64(frame)

            if person_count > threshold_value and person_count != last_alert_count:
                last_alert_count = person_count  # Update last alert count to prevent duplicate alerts
                alert_msg = {
                    "cam_name": cam_name,
                    "cam_id": cam_id,
                    "count": person_count,
                    "threshold": threshold_value,
                    "alert_frame": alert_frame_b64,
                    "message": "Crowd threshold exceeded!"
                }
                # base.logging.info(f"🚨 {alert_msg}")
                if producer:
                    producer.produce("crowd", json.dumps(alert_msg).encode('utf-8'))
                    producer.flush()
                try:
                    response = requests.post(api_endpoint, json=alert_msg, timeout=2)
                    if response.status_code == 200:
                        base.logging.info(f"✅ Sent count={person_count} to API.")
                    else:
                        base.logging.warning(f"⚠️ API responded with status code {response.status_code}, Response: {response.text}")
                except Exception as api_ex:
                    base.logging.error(f"❌ Error sending data to API: {api_ex}")

            out.write(frame)
            base.cv2.imshow("Inference", frame)
            if base.cv2.waitKey(1) & 0xFF == 27:
                break

        except Exception as e:
            base.logging.error(f"❌ Unexpected error: {e}")
            continue

    cap.release()
    out.release()
    base.cv2.destroyAllWindows()

# ✅ Main Process
def main(rtsp, rtmp, cam_name, cam_id, roi, threshold):
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
        process_stream(cap, out, ai_model, cam_id, cam_name, roi, producer, threshold)
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
        description="Real-time Crowd Counting & Threshold Alerts with ROI (using YOLOv8, RTSP, and RTMP)"
    )
    parser.add_argument("--rtmp", type=str, required=True, help="RTMP output URL for broadcasting")
    parser.add_argument("--rtsp", type=str, required=True, help="RTSP URL of the camera")
    parser.add_argument("--cam_name", type=str, required=True, help="Camera Name")
    parser.add_argument("--cam_id", type=str, required=True, help="Camera ID")
    parser.add_argument("--roi", type=int, nargs=4, required=True, help="ROI coordinates in format x1 y1 x2 y2")
    parser.add_argument("--threshold", type=str, required=True, help="Crowd threshold for alerting")
    args = parser.parse_args()

    main(
        rtsp=args.rtsp,
        rtmp=args.rtmp,
        cam_name=args.cam_name,
        cam_id=args.cam_id,
        roi=args.roi,
        threshold=args.threshold
    )


