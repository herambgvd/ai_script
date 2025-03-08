import base
import json
import requests
import base64

# ✅ Function to Convert Cropped Fire/Smoke Region to Base64
def encode_image_to_base64(frame, x1, y1, x2, y2):
    """Crops the detected region and encodes it to base64."""
    try:
        cropped_region = frame[y1:y2, x1:x2]
        _, buffer = base.cv2.imencode('.jpg', cropped_region)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        base.logging.error(f"❌ Error encoding image: {e}")
        return None  # Return None if encoding fails

# ✅ Process Stream and Broadcast Inference
def process_stream(cap, out, ai_model, cam_id, cam_name, roi, producer):
    """
       Process video frames to detect and track objects, stream to RTMP, and generate payloads for non-compliance.
       Fire and smoke detections trigger an alert sent to an API endpoint and a Kafka topic.
    """
    base.logging.info(f"📌 Starting inference for Camera: {cam_name} (ID: {cam_id})")

    # Define Fire & Smoke Class Indices and Confidence Thresholds
    FIRE_CLASS_INDEX = 0   # Adjust as per your model
    SMOKE_CLASS_INDEX = 1  # Adjust as per your model
    CONFIDENCE_THRESHOLD_FIRE = 0.55
    CONFIDENCE_THRESHOLD_SMOKE = 0.75

    # Fire Alert API Endpoint
    FIRE_ALERT_API = "http://0.0.0.0:8080/fire/event"

    # Starting Inference
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
            # Resize frame for processing
            frame = base.cv2.resize(frame, (640, 480))

            # Perform inference
            results = ai_model.track(frame, persist=True, verbose=False)

            # Bounding Boxes
            try:
                fire_detected = False
                fire_alert_payload = None

                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy()  
                    detections = results[0].boxes.data.cpu().numpy()

                    for i, row in enumerate(detections):
                        x1, y1, x2, y2 = map(int, row[:4])  
                        conf = float(row[5])  
                        cls = int(row[6])  
                        track_id = int(track_ids[i])

                        # ✅ Fire & Smoke Detection Logic
                        if (cls == FIRE_CLASS_INDEX and conf > CONFIDENCE_THRESHOLD_FIRE) or \
                           (cls == SMOKE_CLASS_INDEX and conf > CONFIDENCE_THRESHOLD_SMOKE):
                            fire_detected = True
                            label = "🔥 Fire" if cls == FIRE_CLASS_INDEX else "💨 Smoke"
                            base.cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for fire/smoke
                            base.cv2.putText(frame, label, (x1, y1 - 5), base.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                            # Convert Detected Region to Base64
                            base64_image = encode_image_to_base64(frame, x1, y1, x2, y2)

                            # Fire Alert Payload
                            fire_alert_payload = {
                                "camera_id": cam_id,
                                "camera_name": cam_name,
                                "detection": label,
                                "confidence": conf,
                                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                                "timestamp": base.datetime.now().isoformat(),
                                "image_base64": base64_image  # ✅ Add base64 image
                            }

                        # ✅ Person Tracking Logic
                        if cls == 0:  # Only track persons
                            label = f'{track_id}'
                            base.cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for persons
                            base.cv2.putText(frame, label, (x1, y1 - 5), base.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # ✅ Send Fire Alert if Fire/Smoke Detected
                if fire_detected:
                    try:
                        # Send Alert to API
                        response = requests.post(FIRE_ALERT_API, json=fire_alert_payload, timeout=5)
                        base.logging.info(f"🔥 Fire alert sent to API: {response.status_code}")

                        # Send Alert to Kafka
                        if producer:
                            producer.produce("fire_alerts", json.dumps(fire_alert_payload))
                            producer.flush()
                            base.logging.info(f"🔥 Fire alert sent to Kafka")
                    except Exception as alert_ex:
                        base.logging.error(f"❌ Failed to send fire alert: {alert_ex}")

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



def main(rtsp, rtmp, cam_name, cam_id, roi,model_path):
    producer = None
    try:
        # Setup Kafka Producer
        producer = base.get_kafka_producer()

        # Load The Model
        ai_model = base.load_custom_model(model_path)

        # Setup Live Capture
        cap = base.initialize_capture(rtsp_url=rtsp)
        if cap is None:  # 🚀 Stop execution if RTSP stream failed
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


parser = base.argparse.ArgumentParser(description="Person Entry-Exit Detection using YOLOv8 and RTSP to RTMP Streaming")
parser.add_argument("--rtmp", type=str, required=True, help="RTMP output URL for broadcasting")
parser.add_argument("--rtsp", type=str, required=True, help="RTSP URL of the camera")
parser.add_argument("--cam_name", type=str, required=True, help="Camera Name")
parser.add_argument("--cam_id", type=str, required=True, help="Camera ID")
parser.add_argument("--roi", type=int, nargs=4, required=True, help="Entry ROI in format x1 y1 x2 y2 x3 y3 x4 y4")
parser.add_argument("--model_path", type=str, required=True, help="Custom Model Path")
args = parser.parse_args()

if __name__ == "__main__":
    main(
        rtsp=args.rtsp,
        rtmp=args.rtmp,
        cam_name=args.cam_name,
        cam_id=args.cam_id,
        roi=args.roi,
        model_path=args.model_path
    )
