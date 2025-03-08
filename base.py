import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import cv2
import urllib.parse
import argparse
import logging
import requests
import sys
import gi
import json
from ultralytics import YOLO
from confluent_kafka import Producer


gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Initialize GStreamer
Gst.init(None)

# Logger Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ✅ Kafka Producer Setup
def get_kafka_producer():
    try:
        logging.info("Initializing Kafka producer...")
        producer_conf = {
            'bootstrap.servers': 'localhost:9092',
            'client.id': 'notifications'
        }
        producer = Producer(producer_conf)
        logging.info("✅ Kafka producer setup completed.")
        return producer
    except Exception as e:
        logging.error(f"❌ Error setting up Kafka producer: {e}")
        return None

# ✅ Load YOLO Model
def load_model():
    try:
        logging.info("Loading YOLO model...")
        model_path = 'yolov8s.pt'
        ai_model = YOLO(model_path, task="detect")
        logging.info("✅ Model loaded successfully.")
        return ai_model
    except Exception as e:
        logging.error(f"❌ Error loading YOLO model: {e}")
        sys.exit(1)

# ✅ Initialize Video Capture with Retry
def initialize_capture(rtsp_url, max_retries=3):
    logging.info("Initializing video capture...")
    logging.info("Processing RTSP URL for GStreamer pipeline...")
    parsed_url = urllib.parse.urlparse(rtsp_url)
    username = parsed_url.username
    password = parsed_url.password

    if username and password:
        encoded_password = urllib.parse.quote(password)
        new_netloc = f"{username}:{encoded_password}@{parsed_url.hostname}:{parsed_url.port}"
        rtsp_url = urllib.parse.urlunparse((parsed_url.scheme, new_netloc, parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment))

    logging.info(f"✅ Processed RTSP URL: {rtsp_url}")
    pipeline =  f"rtspsrc location={rtsp_url} protocols=tcp latency=100 do-timestamp=true is-live=true ! rtph264depay ! h264parse ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink"

    for attempt in range(1, max_retries + 1):
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            logging.info(f"✅ Video capture initialized successfully on attempt {attempt}.")
            return cap
        logging.warning(f"⚠️ Attempt {attempt}/{max_retries} failed to open RTSP stream. Retrying...")

    logging.error("❌ Failed to open RTSP stream after multiple attempts!")
    return None

# ✅ GStreamer Pipeline for RTMP Broadcasting
def setup_live_broadcasting(rtmp_url):
    try:
        bitrate = 2500  # Default bitrate in kbps
        frame_width = 640
        frame_height = 480
        rtmp_pipeline = f'appsrc ! videoconvert ! videoscale ! x264enc bitrate={bitrate} speed-preset=ultrafast tune=zerolatency key-int-max=30 ! flvmux streamable=true ! rtmpsink location={rtmp_url} sync=false'
        out = cv2.VideoWriter(rtmp_pipeline, cv2.CAP_GSTREAMER, 0, 25, (frame_width, frame_height), True)
        logging.info(f"Video writer started successfully for RTMP URL: {rtmp_url}")
        return out
    except Exception as e:
        logging.error(f"Error setting up video writer: {e}")

# ✅  Load Custom Model
def load_custom_model(model_path):
    try:
        logging.info("Loading YOLO model...")
        ai_model = YOLO(model_path, task="detect")
        logging.info("✅ Model loaded successfully.")
        return ai_model
    except Exception as e:
        logging.error(f"❌ Error loading YOLO model: {e}")
        sys.exit(1)