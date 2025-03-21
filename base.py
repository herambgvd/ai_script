import os
import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import cv2
import urllib.parse
import argparse
import logging
import requests
import time
import sys
import gi
import json
from ultralytics import YOLO
from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()


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
        model_path = os.getenv('BASE_MODEL')
        ai_model = YOLO(model_path, task="detect")
        logging.info("✅ Model loaded successfully.")
        return ai_model
    except Exception as e:
        logging.error(f"❌ Error loading YOLO model: {e}")
        sys.exit(1)

# ✅ Initialize Video Capture with Retry
def initialize_capture(rtsp_url, max_retries=3, use_ffmpeg=False):
    """
    Initialize RTSP video capture using GStreamer.
    - If GStreamer fails, it optionally falls back to FFMPEG.
    - Allows direct webcam access if `rtsp_url` is 0.
    - Retries multiple times before failing.
    """
    logging.info("Initializing video capture...")

    # ✅ Check if using Webcam instead of RTSP
    if str(rtsp_url) == "0":
        logging.info("🎥 Using Webcam (cv2.VideoCapture(0))")
        cap = cv2.VideoCapture(0)  # Direct webcam access
        if cap.isOpened():
            logging.info("✅ Webcam initialized successfully.")
            return cap
        else:
            logging.error("❌ Failed to initialize webcam!")
            return None

    # ✅ Ensure RTSP URL is properly formatted
    parsed_url = urllib.parse.urlparse(rtsp_url)
    username, password = parsed_url.username, parsed_url.password

    if username and password:
        encoded_password = urllib.parse.quote(password, safe='')
        new_netloc = f"{username}:{encoded_password}@{parsed_url.hostname}:{parsed_url.port}"
        rtsp_url = urllib.parse.urlunparse((parsed_url.scheme, new_netloc, parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment))

    logging.info(f"✅ Processed RTSP URL: {rtsp_url}")

    # ✅ Define GStreamer Pipeline
    gst_pipeline = (
        f"rtspsrc location={rtsp_url} latency=500 protocols=tcp do-timestamp=true is-live=true "
        f"! rtph264depay ! h264parse ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true"
    )

    # ✅ Define FFMPEG Pipeline (Optional Fallback)
    ffmpeg_pipeline = rtsp_url  # OpenCV handles this internally with CAP_FFMPEG

    # ✅ Attempt to Open Video Stream with Retries
    for attempt in range(1, max_retries + 1):
        logging.info(f"🚀 Attempt {attempt}/{max_retries} to open RTSP stream...")

        # Try GStreamer First
        if not use_ffmpeg:
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            backend_used = "GStreamer"
        else:
            cap = cv2.VideoCapture(ffmpeg_pipeline, cv2.CAP_FFMPEG)
            backend_used = "FFMPEG"

        if cap.isOpened():
            logging.info(f"✅ Video capture initialized successfully using {backend_used} on attempt {attempt}.")
            return cap

        logging.warning(f"⚠️ Attempt {attempt} failed to open RTSP stream using {backend_used}. Retrying in 3 seconds...")
        cap.release()  # Ensure previous instance is closed
        time.sleep(3)  # Exponential Backoff (adjust delay if needed)

    # ✅ If GStreamer Fails, Try FFMPEG as a Fallback
    if not use_ffmpeg:
        logging.error("❌ GStreamer failed to open RTSP stream after multiple attempts! Trying FFMPEG...")
        return initialize_capture(rtsp_url, max_retries, use_ffmpeg=True)

    logging.error("❌ RTSP stream failed to open after all retries using both GStreamer and FFMPEG!")
    return None  # Return None if all attempts fail


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
        if model_path == "fire":
            MODEL_PATH = os.getenv('FIRE_PATH')
        if model_path == "box":
            MODEL_PATH = os.getenv('BOX_PATH')
        if model_path == "fd":
            MODEL_PATH = os.getenv('FACE_PATH')
        if model_path == "door":
            MODEL_PATH = os.getenv('DOOR_PATH')
        if model_path == "frs":
            MODEL_PATH = os.getenv('FACE_PATH')

        ai_model = YOLO(MODEL_PATH, task="detect")
        logging.info("✅ Model loaded successfully.")
        return ai_model
    except Exception as e:
        logging.error(f"❌ Error loading YOLO model: {e}")
        sys.exit(1)

def load_clothing_model():
    try:
        logging.info("Loading YOLO model...")
        ai_model = YOLO(os.getenv('CLOTH_PATH'), task="detect")
        logging.info("✅ Model loaded successfully.")
        return ai_model
    except Exception as e:
        logging.error(f"❌ Error loading YOLO model: {e}")
        sys.exit(1)

def load_pose_model():
    try:
        logging.info("Loading YOLO model...")
        ai_model = YOLO(os.getenv('POSE_PATH'), task="detect")
        logging.info("✅ Model loaded successfully.")
        return ai_model
    except Exception as e:
        logging.error(f"❌ Error loading YOLO model: {e}")
        sys.exit(1)