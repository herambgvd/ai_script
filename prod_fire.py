import base
import json
import requests
import base64
import numpy as np

# ✅ Function to Convert Full Image Frame to Base64
def encode_frame_to_base64(frame):
    """Encodes the full frame to base64 format."""
    try:
        _, buffer = base.cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        base.logging.error(f"❌ Error encoding full frame: {e}")
        return None  # Return None if encoding fails

# ✅ Process Stream and Broadcast Inference (Tracking-Based Alerts)
def process_stream(cap, out, ai_model, cam_id, cam_name, roi, producer):
    """
    Detect fire/smoke in ROI and send alerts, ensuring each tracking ID triggers only one alert.
    """
    base.logging.info(f"📌 Starting inference for Camera: {cam_name} (ID: {cam_id})")

    # Fire & Smoke Class Indices and Confidence Thresholds
    FIRE_CLASS_INDEX = 0  
    SMOKE_CLASS_INDEX = 1  
    CONFIDENCE_THRESHOLD_FIRE = 0.55
    CONFIDENCE_THRESHOLD_SMOKE = 0.75

    # Fire Alert API Endpoint
    FIRE_ALERT_API = "http://0.0.0.0:8080/api/v1/fire/event"

    # ✅ Track alerted IDs to avoid duplicate alerts
    alerted_track_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            base.logging.error("❌ Stream capture failed at frame")
            break

        try:
            frame = base.cv2.resize(frame, (640, 480))

            # ✅ Extract ROI coordinates
            roi_x1, roi_y1, roi_x2, roi_y2 = roi[:4]

            # ✅ Draw ROI bounding box on full frame
            base.cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)  # Blue box

            # ✅ Crop the frame to the ROI region for inference
            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            # ✅ Perform inference ONLY on the ROI
            results = ai_model.track(roi_frame, persist=True, verbose=False)

            fire_detected = False
            fire_alert_payload = None
            current_track_ids = set()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy()  
                detections = results[0].boxes.data.cpu().numpy()

                for i, row in enumerate(detections):
                    x1, y1, x2, y2 = map(int, row[:4])  # Bounding box (in cropped ROI)
                    conf = float(row[5])  
                    cls = int(row[6])  
                    track_id = int(track_ids[i])

                    # ✅ Convert bounding box coordinates from ROI to full frame
                    x1, x2 = x1 + roi_x1, x2 + roi_x1
                    y1, y2 = y1 + roi_y1, y2 + roi_y1

                    # ✅ Track current IDs in the frame
                    current_track_ids.add(track_id)

                    # ✅ Fire & Smoke Detection Logic (Restricted to ROI)
                    if (cls == FIRE_CLASS_INDEX and conf > CONFIDENCE_THRESHOLD_FIRE) or \
                       (cls == SMOKE_CLASS_INDEX and conf > CONFIDENCE_THRESHOLD_SMOKE):

                        # ✅ Send alert **ONLY IF the tracking ID has not been alerted**
                        if track_id not in alerted_track_ids:
                            fire_detected = True
                            label = "🔥 Fire" if cls == FIRE_CLASS_INDEX else "💨 Smoke"
                            base.cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for fire/smoke
                            base.cv2.putText(frame, label, (x1, y1 - 5), base.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                            # ✅ Convert Full Frame to Base64
                            full_frame_base64 = encode_frame_to_base64(frame)

                            # ✅ Fire Alert Payload
                            fire_alert_payload = {
                                "cam_id": cam_id,
                                "cam_name": cam_name,
                                "alert_frame": full_frame_base64  # ✅ Attach full frame image
                            }

                            # ✅ Add this tracking ID to the set (Avoid duplicate alerts)
                            alerted_track_ids.add(track_id)

            # ✅ Reset tracking IDs if they disappear from the frame
            alerted_track_ids.intersection_update(current_track_ids)

            # ✅ Send Fire Alert if Fire/Smoke Detected
            if fire_detected:
                try:
                    # Send Alert to API
                    response = requests.post(FIRE_ALERT_API, json=fire_alert_payload, timeout=5)
                    base.logging.info(f"🔥 Fire alert sent to API: {response.status_code}")

                    # Send Alert to Kafka
                    if producer:
                        producer.produce("fire", json.dumps(fire_alert_payload))
                        producer.flush()
                        base.logging.info(f"🔥 Fire alert sent to Kafka")
                except Exception as alert_ex:
                    base.logging.error(f"❌ Failed to send fire alert: {alert_ex}")

            # ✅ Broadcast processed frame to RTMP
            out.write(frame)

            # ✅ Display output frame for debugging
            # base.cv2.imshow("Inference", frame)
            # if base.cv2.waitKey(1) & 0xFF == 27:
            #     base.logging.info("❌ User terminated the process.")
            #     break
        except Exception as e:
            base.logging.error(f"❌ Unexpected error: {e}")
            continue

    cap.release()
    out.release()
    # base.cv2.destroyAllWindows()

# ✅ Main Function
def main(rtsp, rtmp, cam_name, cam_id, roi, model_path):
    producer = None
    try:
        # ✅ Setup Kafka Producer
        producer = base.get_kafka_producer()

        # ✅ Load The Model
        ai_model = base.load_custom_model(model_path)

        # ✅ Setup Live Capture
        cap = base.initialize_capture(rtsp_url=rtsp)
        if cap is None:
            base.logging.error(f"❌ Exiting: Unable to open RTSP stream for camera {cam_name} (ID: {cam_id})")
            return

        # ✅ Setup Live RTMP Streaming
        out = base.setup_live_broadcasting(rtmp_url=rtmp)

        # ✅ Process Stream & Perform Inference
        process_stream(cap, out, ai_model, cam_id, cam_name, roi, producer)
        base.logging.info("Processing and streaming completed.")

    except Exception as e:
        base.logging.error(f"Error in main system: {e}")

# ✅ Argument Parser
parser = base.argparse.ArgumentParser(description="Fire and Smoke Detection within ROI (RTSP to RTMP Streaming)")
parser.add_argument("--rtmp", type=str, required=True, help="RTMP output URL for broadcasting")
parser.add_argument("--rtsp", type=str, required=True, help="RTSP URL of the camera")
parser.add_argument("--cam_name", type=str, required=True, help="Camera Name")
parser.add_argument("--cam_id", type=str, required=True, help="Camera ID")
parser.add_argument("--roi", type=int, nargs=4, required=True, help="ROI in format x1 y1 x2 y2")
parser.add_argument("--model_path", type=str, required=True, help="Custom Model Path")
args = parser.parse_args()

# ✅ Run Main
if __name__ == "__main__":
    main(rtsp=args.rtsp, rtmp=args.rtmp, cam_name=args.cam_name, cam_id=args.cam_id, roi=args.roi, model_path=args.model_path)
