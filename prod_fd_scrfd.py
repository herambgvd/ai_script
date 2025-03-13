import time
import base
import cv2
import os
import numpy as np
import json
import torch
import yaml
import requests
import base64
import logging
from face_utils.detection import SCRFD
from face_utils.face_tracking.tracker.byte_tracker import BYTETracker
from face_utils.face_tracking.tracker.visualize import plot_tracking

# ✅ Setup Logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


# ✅ Load SCRFD Model with GPU Support
def load_scrfd_model(model_path):
    logging.info("🚀 Loading SCRFD model...")
    scrfd = SCRFD(model_file=model_path)
    # ✅ Ensure correct input size (Fix shape mismatch)
    scrfd.prepare(ctx_id=0, input_size=(640, 640), providers=['CUDAExecutionProvider'])  
    
    logging.info("✅ SCRFD Model Loaded with Correct Input Shape")
    return scrfd



# ✅ Load Tracking Configuration
def load_config(file_name):
    logging.info(f"🔍 Loading YAML configuration from {file_name}")
    try:
        with open(file_name, "r") as stream:
            config = yaml.safe_load(stream)
            logging.info("✅ YAML Configuration Loaded Successfully")
            return config
    except yaml.YAMLError as exc:
        logging.error(f"❌ Error loading YAML: {exc}")
        return None


# ✅ Convert Image Frame to Base64
def frame_to_base64(frame):
    try:
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        logging.error(f"❌ Error encoding frame to Base64: {e}")
        return None


# ✅ Process Stream with Tracking and ROI-based Detection
def process_stream(cap, out, ai_model, cam_id, cam_name, roi, producer):
    logging.info("🚀 Starting Face Detection and Tracking Process...")

    x1_roi, y1_roi, x2_roi, y2_roi = roi
    logging.debug(f"📌 ROI Coordinates: x1={x1_roi}, y1={y1_roi}, x2={x2_roi}, y2={y2_roi}")

    API_ENDPOINT = "http://0.0.0.0:8080/api/v1/fd/event"
    KAFKA_TOPIC = "face_detection"
    CONFIDENCE_THRESHOLD = 0.70  

    logging.info("🛠 Loading Tracker Configuration")
    tracker_config = load_config("/home/heramb/gvd_clarify_ops/face_utils/face_tracking/config/config_tacking.yaml")

    if tracker_config is None:
        logging.error("❌ Failed to load tracker configuration. Exiting.")
        return

    tracker = BYTETracker(args=tracker_config, frame_rate=30)
    logging.info("✅ BYTETracker Initialized")

    frame_id = 0
    fps = -1

    # ✅ Store processed tracking IDs to avoid duplicate payloads
    alerted_track_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.error("❌ Stream capture failed")
            break

        frame = cv2.resize(frame, (640, 480))
        logging.debug("📷 Frame captured and resized to 640x480")

        # ✅ Default `online_im` to current frame to prevent uninitialized variable issue
        online_im = frame.copy()

        # ✅ Capture Full Frame for Payload
        full_frame_base_image = frame_to_base64(frame)

        # ✅ Draw ROI Bounding Box
        cv2.rectangle(online_im, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 0, 0), 2)

        # ✅ Run Face Detection
        try:
            detections, img_info, bboxes, landmarks = ai_model.detect_tracking(image=frame, thresh=0.5)
        except Exception as e:
            logging.error(f"❌ Face Detection Error: {e}")
            continue

        if bboxes is not None and len(bboxes) > 0:
            try:
                # ✅ Convert to NumPy if needed
                if not isinstance(bboxes, np.ndarray):
                    bboxes = np.array(bboxes, dtype=np.float32)

                if len(bboxes.shape) == 1:  # If only one detection, reshape
                    bboxes = bboxes.reshape(1, -1)

                # ✅ Ensure correct format for BYTETracker
                online_targets = tracker.update(
                    torch.tensor(bboxes, dtype=torch.float32),
                    [int(img_info["height"]), int(img_info["width"])], (128, 128)
                )

                online_tlwhs = []
                online_ids = []
                online_scores = []

                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > tracker_config["aspect_ratio_thresh"]
                    if tlwh[2] * tlwh[3] > tracker_config["min_box_area"] and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)

                logging.debug(f"🎯 Tracked Faces: {online_ids}")

                # ✅ Draw bounding boxes and IDs
                for idx, tid in enumerate(online_ids):
                    bbox = bboxes[idx]
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        score = float(detections[idx][-1])  # Extract confidence score

                        # ✅ Draw bounding box and tracking ID on `online_im`
                        cv2.rectangle(online_im, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
                        cv2.putText(online_im, f"ID {tid} ({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # ✅ Extract Face Frame Without Bounding Box
                        cropped_face = frame[y1:y2, x1:x2]
                        face_detected_frame = frame_to_base64(cropped_face)

                        if tid not in alerted_track_ids:
                            payload = {
                                "cam_id": cam_id,
                                "cam_name": cam_name,
                                "face_frame": face_detected_frame
                            }

                            # ✅ Push to Kafka
                            try:
                                producer.produce(KAFKA_TOPIC, json.dumps(payload).encode('utf-8'))
                                producer.flush()
                                logging.info(f"✅ Payload sent to Kafka for Tracking ID: {tid}")
                            except Exception as e:
                                logging.error(f"❌ Kafka Push Error: {e}")

                            # ✅ Push to API
                            # try:
                            #     response = requests.post(API_ENDPOINT, json=payload, headers={"Content-Type": "application/json"})
                            #     if response.status_code == 200:
                            #         logging.info(f"✅ Payload successfully sent to API for Tracking ID: {tid}")
                            #     else:
                            #         logging.error(f"❌ API Push Error: {response.status_code} - {response.text}")
                            # except Exception as e:
                            #     logging.error(f"❌ API Push Exception: {e}")

                            # ✅ Mark tracking ID as processed
                            alerted_track_ids.add(tid)

            except Exception as e:
                logging.error(f"❌ Error in Tracking Update: {e}")
                continue

        if out:
            out.write(online_im)

        cv2.imshow("Face Tracking (BYTETracker)", online_im)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()




# ✅ Main Process
def main(rtsp, rtmp, cam_name, cam_id, roi, model_path):
    try:
        logging.info("🚀 Initializing Kafka Producer...")
        producer = base.get_kafka_producer()
        logging.info("✅ Kafka Producer Initialized")

        if model_path == "fd":
            MODEL_PATH = os.getenv('FD_PATH')
            logging.info(f"✅ Model Path Resolved: {MODEL_PATH}")

        ai_model = load_scrfd_model(MODEL_PATH)

        logging.info("🎥 Initializing Video Capture...")
        cap = base.initialize_capture(rtsp_url=rtsp)
        if cap is None:
            logging.error(f"❌ Unable to open RTSP stream for {cam_name} (ID: {cam_id})")
            return

        out = base.setup_live_broadcasting(rtmp_url=rtmp)

        process_stream(cap, out, ai_model, cam_id, cam_name, roi, producer)
        logging.info("✅ Processing and Streaming Completed")

    except Exception as e:
        logging.error(f"❌ Error in main system: {e}")

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
