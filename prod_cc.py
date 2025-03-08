import base

# ✅ Process Stream and Broadcast Inference
def process_stream(cap, out, ai_model, cam_id, cam_name, roi_entry, roi_exit, producer):
    """
       Process video frames to detect and track objects, stream to RTMP, and generate payloads for non-compliance.
       Payloads are sent to Kafka including cropped face image data as base64, but only once per tracking ID.
    """
    base.logging.info(f"📌 Starting inference for Camera: {cam_name} (ID: {cam_id})")

    # Analysis Variables
    people_in, people_out = {}, {}
    in_count, out_count = set(), set()
    previous_in_count = 0
    previous_out_count = 0
    # Preprocess tasks
    entry = [(roi_entry[i], roi_entry[i + 1]) for i in range(0, len(roi_entry), 2)]
    exit = [(roi_exit[i], roi_exit[i + 1]) for i in range(0, len(roi_exit), 2)]
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
            # logging.info("🔹 Processing frame ...")
            frame = base.cv2.resize(frame, (640, 480))
            # Perform inference
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
                        if cls == 0:  # ✅ Only process persons
                            center_x = int((x1 + x2)/2)
                            center_y = int((y1 + y2)/2)
                            point = (center_x,center_y)
                            # ✅ Check if person is inside roi_exit
                            if base.cv2.pointPolygonTest(base.np.array(exit, base.np.int32), point, False) >= 0:
                                people_out[track_id] = point  
                                base.cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                            if track_id in people_out:
                                if base.cv2.pointPolygonTest(base.np.array(entry, base.np.int32), point, False) >= 0:
                                    base.cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    base.cv2.circle(frame, point, 4, (255, 0, 255), -1)
                                    out_count.add(track_id)  

                            # ✅ Check if person is inside roi_entry
                            if base.cv2.pointPolygonTest(base.np.array(entry, base.np.int32), point, False) >= 0:
                                people_in[track_id] = point  
                                base.cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                            if track_id in people_in:
                                if base.cv2.pointPolygonTest(base.np.array(exit, base.np.int32), point, False) >= 0:
                                    base.cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    base.cv2.circle(frame, point, 4, (255, 0, 255), -1)
                                    in_count.add(track_id) 
                else:
                    base.logging.warning("⚠️ No valid detection results available.")
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

            # ✅ Draw ROI Entry (Blue)
            base.cv2.polylines(frame, [base.np.array(entry, base.np.int32)], isClosed=True, color=(255, 0, 0), thickness=1)
            base.cv2.putText(frame, "Entry", (entry[0][0], entry[0][1] - 10), 
                        base.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # ✅ Draw ROI Exit (Red)
            base.cv2.polylines(frame, [base.np.array(exit, base.np.int32)], isClosed=True, color=(0, 0, 255), thickness=1)
            base.cv2.putText(frame, "Exit", (exit[0][0], exit[0][1] - 10), 
                        base.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ✅ Display "In" & "Out" Counts
            base.cv2.putText(frame, f"IN: {len(in_count)}", (20, 40), 
                        base.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            base.cv2.putText(frame, f"OUT: {len(out_count)}", (20, 70), 
                        base.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # If the count increases, produce the message to Kafka "cc" topic
            if len(in_count) > previous_in_count or len(out_count) > previous_out_count:
                cc_msg = base.json.dumps({
                    "cam_name": cam_name,
                    "cam_id": cam_id,
                    "in_count": len(in_count),
                    "out_count": len(out_count)
                })
                if producer:
                    try:
                        producer.produce("cc", cc_msg)
                        producer.flush()
                    except Exception as kafka_ex:
                        base.logging.error(f"❌ Failed to produce count update message to Kafka: {kafka_ex}")
                # Push to the HTTP endpoint
                api_url = "http://0.0.0.0:8080/api/v1/cc/event"
                headers = {
                    "accept": "application/json",
                    "Content-Type": "application/json"
                }
                try:
                    response = base.requests.post(api_url, data=cc_msg, headers=headers, timeout=5)
                    if response.status_code == 200:
                        base.logging.info(f"✅ Successfully pushed cc_msg to API: {response.json()}")
                    else:
                        base.logging.error(f"❌ Failed to push cc_msg to API. Status Code: {response.status_code}, Response: {response.text}")
                except base.requests.RequestException as http_ex:
                    base.logging.error(f"❌ HTTP request failed: {http_ex}")
                previous_in_count = len(in_count)
                previous_out_count = len(out_count)

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


def main(rtsp, rtmp, cam_name, cam_id, roi_entry, roi_exit):
    producer = None
    try:
        # Setup Kafka Producer
        producer = base.get_kafka_producer()

        # Load The Model
        ai_model = base.load_model()

        # Setup Live Capture
        cap = base.initialize_capture(rtsp_url=rtsp)
        if cap is None:  # 🚀 Stop execution if RTSP stream failed
            base.logging.error(f"❌ Exiting: Unable to open RTSP stream for camera {cam_name} (ID: {cam_id})")
            return

        # Setup Live for RTMP streaming
        out = base.setup_live_broadcasting(rtmp_url=rtmp)

        # Process Frames and Stream to RTMP
        process_stream(cap, out, ai_model, cam_id, cam_name, roi_entry, roi_exit, producer)
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
parser.add_argument("--roi_entry", type=int, nargs=8, required=True, help="Entry ROI in format x1 y1 x2 y2 x3 y3 x4 y4")
parser.add_argument("--roi_exit", type=int, nargs=8, required=True, help="Exit ROI in format x1 y1 x2 y2 x3 y3 x4 y4")
args = parser.parse_args()

if __name__ == "__main__":
    main(args.rtsp, args.rtmp, args.cam_name, args.cam_id, args.roi_entry, args.roi_exit)
