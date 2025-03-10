import base
import requests
import base64
import json

# ✅ Convert Image to Base64
def encode_image_to_base64(image):
    _, buffer = base.cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# ✅ Process Stream and Broadcast Inference
def process_stream(cap, out, ai_model, cam_id, cam_name, roi, producer):
    """
    - Detects and tracks objects, streams to RTMP, and triggers intrusion alerts.
    - Sends alerts to Kafka 'fence' topic.
    - Posts alerts to HTTP API at "http://0.0.0.0:8080/api/v1/intrusion/event".
    - Ensures alerts for each track ID are sent only once.
    """
    base.logging.info(f"📌 Starting inference for Camera: {cam_name} (ID: {cam_id})")

    # Parse ROI coordinates
    roi_x1, roi_y1, roi_x2, roi_y2 = roi[0], roi[1], roi[2], roi[3]

    # Keep track of which track IDs have already triggered an intrusion alert
    intruded_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            base.logging.error("❌ Stream capture failed at frame")
            error_msg = f"Stream capture failed for camera {cam_name}"
            if producer:
                try:
                    producer.produce("notifications", error_msg)
                    producer.flush()
                except Exception as kafka_ex:
                    base.logging.error(f"❌ Failed to produce error message to Kafka: {kafka_ex}")
            break
        
        try:
            # Resize the frame for faster processing
            frame = base.cv2.resize(frame, (640, 480))

            # Draw the ROI rectangle for visualization (optional)
            base.cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

            # Perform YOLO inference with object tracking
            results = ai_model.track(frame, persist=True, verbose=False)

            # Bounding Boxes
            try:
                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy()
                    detections = results[0].boxes.data.cpu().numpy()

                    for i, row in enumerate(detections):
                        x1, y1, x2, y2 = map(int, row[:4])
                        conf = float(row[5])
                        cls = int(row[6])
                        track_id = int(track_ids[i])

                        # Only process persons (cls == 0)
                        if cls == 0:
                            # Draw bounding box
                            label = f'ID {track_id}'
                            base.cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            base.cv2.putText(frame, label, (x1, y1 - 5), base.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            # Check if the center of the bounding box is within ROI
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Send alert only if not already sent for this track_id
                            if (roi_x1 <= center_x <= roi_x2) and (roi_y1 <= center_y <= roi_y2):
                                if track_id not in intruded_ids:
                                    # Extract person image
                                    person_crop = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else None
                                    
                                    # Convert images to base64
                                    alert_frame_b64 = encode_image_to_base64(frame)  # Full frame
                                    detect_frame_b64 = encode_image_to_base64(person_crop) if person_crop is not None else ""

                                    # Create payload
                                    intrusion_payload = {
                                        "cam_name": cam_name,
                                        "cam_id": cam_id,
                                        "alert_frame": alert_frame_b64,
                                        "detect_frame": detect_frame_b64
                                    }

                                    # Send to Kafka
                                    try:
                                        producer.produce("fence", json.dumps(intrusion_payload))
                                        producer.flush()
                                        base.logging.info(f"✅ Intrusion alert sent to Kafka for track ID {track_id}")
                                    except Exception as kafka_ex:
                                        base.logging.error(f"❌ Failed to send intrusion alert to Kafka: {kafka_ex}")

                                    # Send HTTP POST request
                                    try:
                                        response = requests.post(
                                            "http://0.0.0.0:8080/api/v1/intrusion/event",
                                            json=intrusion_payload,
                                            timeout=5  # Set timeout for better reliability
                                        )
                                        if response.status_code == 200:
                                            base.logging.info(f"✅ Successfully posted alert to API: {response.json()}")
                                        else:
                                            base.logging.warning(f"⚠️ API responded with status code {response.status_code}")
                                    except Exception as api_ex:
                                        base.logging.error(f"❌ Failed to post alert to API: {api_ex}")

                                    # Mark this ID as alerted
                                    intruded_ids.add(track_id)
                else:
                    # base.logging.warning("⚠️ No valid detection results available.")
                    pass

            except Exception as e:
                base.logging.error(f"❌ Error during inference: {e}")
                error_msg = f"Inference error in camera {cam_name}: {e}"
                if producer:
                    try:
                        producer.produce("notifications", error_msg)
                        producer.flush()
                    except Exception as kafka_ex:
                        base.logging.error(f"❌ Failed to produce error message to Kafka: {kafka_ex}")
                continue

            # ✅ Broadcast processed frame to RTMP
            out.write(frame)

            # ✅ Display output frame for debugging
            base.cv2.imshow("Inference", frame)
            if base.cv2.waitKey(1) & 0xFF == 27:
                base.logging.info("❌ User terminated the process.")
                break

        except Exception as e:
            base.logging.error(f"❌ Unexpected error: {e}")
            error_msg = f"Unexpected error in camera {cam_name}: {e}"
            if producer:
                try:
                    producer.produce("notifications", error_msg)
                    producer.flush()
                except Exception as kafka_ex:
                    base.logging.error(f"❌ Failed to produce error message to Kafka: {kafka_ex}")
            continue
    
    cap.release()
    out.release()
    base.cv2.destroyAllWindows()



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
                producer.produce("notifications", error_msg)
                producer.flush()
            except Exception as kafka_ex:
                base.logging.error(f"❌ Failed to produce error message to Kafka: {kafka_ex}")


if __name__ == "__main__":
    parser = base.argparse.ArgumentParser(description="Intrusion Detection using YOLOv8 and RTSP to RTMP Streaming")
    parser.add_argument("--rtmp", type=str, required=True, help="RTMP output URL for broadcasting")
    parser.add_argument("--rtsp", type=str, required=True, help="RTSP URL of the camera")
    parser.add_argument("--cam_name", type=str, required=True, help="Camera Name")
    parser.add_argument("--cam_id", type=str, required=True, help="Camera ID")
    parser.add_argument("--roi", type=int, nargs=4, required=True, help="Entry ROI in format x1 y1 x2 y2 (plus optional 4 more if needed)")
    args = parser.parse_args()

    main(args.rtsp, args.rtmp, args.cam_name, args.cam_id, args.roi)
