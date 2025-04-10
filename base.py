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

# Set logging level to DEBUG for detailed messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logging.debug("Importing and initializing GStreamer modules.")
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Initialize GStreamer
logging.debug("Initializing GStreamer...")
Gst.init(None)
logging.debug("GStreamer initialized successfully.")

# ✅ Kafka Producer Setup
def get_kafka_producer():
    logging.debug("Entering get_kafka_producer()")
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

# ✅ Load YOLO Model (General)
def load_model():
    logging.debug("Entering load_model()")
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
    logging.debug("Entering initialize_capture() with rtsp_url: %s", rtsp_url)
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

    # ✅ For GStreamer pipeline: format the RTSP URL (password encoding)
    if not use_ffmpeg:
        logging.debug("Parsing RTSP URL for GStreamer pipeline...")
        parsed_url = urllib.parse.urlparse(rtsp_url)
        username, password = parsed_url.username, parsed_url.password

        if username and password:
            encoded_password = urllib.parse.quote(password, safe='')
            new_netloc = f"{username}:{encoded_password}@{parsed_url.hostname}:{parsed_url.port}"
            formatted_url = urllib.parse.urlunparse((parsed_url.scheme, new_netloc, parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment))
            logging.debug("RTSP URL reformatted to: %s", formatted_url)
        else:
            logging.debug("No username/password found in RTSP URL.")
            formatted_url = rtsp_url
    else:
        # When using FFMPEG, do not reformat; use the original RTSP URL
        logging.debug("Using original RTSP URL for FFMPEG pipeline (no password formatting).")
        formatted_url = rtsp_url

    logging.info(f"Processed RTSP URL: {formatted_url}")

    # ✅ Define pipelines for each backend
    gst_pipeline = (
        f"rtspsrc location={formatted_url} latency=500 protocols=tcp do-timestamp=true is-live=true "
        f"! rtph264depay ! h264parse ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true"
    )
    logging.debug("GStreamer pipeline: %s", gst_pipeline)

    ffmpeg_pipeline = rtsp_url  # Use the original RTSP URL for FFMPEG
    logging.debug("FFMPEG pipeline: %s", ffmpeg_pipeline)

    # ✅ Attempt to open video stream with retries
    for attempt in range(1, max_retries + 1):
        logging.info(f"🚀 Attempt {attempt}/{max_retries} to open RTSP stream...")
        # if not use_ffmpeg:
        #     logging.debug("Attempting to open stream with GStreamer backend...")
        #     cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        #     backend_used = "GStreamer"
        # else:
        logging.debug("Attempting to open stream with FFMPEG backend...")
        cap = cv2.VideoCapture(ffmpeg_pipeline, cv2.CAP_FFMPEG)
        backend_used = "FFMPEG"

        if cap.isOpened():
            logging.info(f"✅ Video capture initialized successfully using {backend_used} on attempt {attempt}.")
            return cap

        logging.warning(f"⚠️ Attempt {attempt} failed to open RTSP stream using {backend_used}. Retrying in 3 seconds...")
        cap.release()  # Ensure previous instance is closed
        time.sleep(3)

    # ✅ Fallback: If GStreamer fails, try FFMPEG as a fallback
    if not use_ffmpeg:
        logging.error("❌ GStreamer failed after multiple attempts! Trying FFMPEG without URL formatting...")
        return initialize_capture(rtsp_url, max_retries, use_ffmpeg=True)

    logging.error("❌ RTSP stream failed to open after all retries using both GStreamer and FFMPEG!")
    return None

# ✅ GStreamer Pipeline for RTMP Broadcasting
def setup_live_broadcasting(rtmp_url):
    logging.debug("Entering setup_live_broadcasting() with RTMP URL: %s", rtmp_url)
    try:
        bitrate = 2500  # Default bitrate in kbps
        frame_width = 640
        frame_height = 480
        rtmp_pipeline = (
            f'appsrc ! videoconvert ! videoscale ! '
            f'x264enc bitrate={bitrate} speed-preset=ultrafast tune=zerolatency key-int-max=30 '
            f'! flvmux streamable=true ! rtmpsink location={rtmp_url} sync=false'
        )
        logging.debug("RTMP pipeline: %s", rtmp_pipeline)
        out = cv2.VideoWriter(rtmp_pipeline, cv2.CAP_GSTREAMER, 0, 25, (frame_width, frame_height), True)
        logging.info(f"Video writer started successfully for RTMP URL: {rtmp_url}")
        return out
    except Exception as e:
        logging.error(f"Error setting up video writer: {e}")
        return None

# ✅ Load Custom Model for different tasks
def load_custom_model(model_path):
    logging.debug("Entering load_custom_model() with model_path: %s", model_path)
    try:
        logging.info("Loading YOLO model for custom task...")
        if model_path == "fire":
            MODEL_PATH = os.getenv('FIRE_PATH')
        elif model_path == "box":
            MODEL_PATH = os.getenv('BOX_PATH')
        elif model_path == "fd":
            MODEL_PATH = os.getenv('FACE_PATH')
        elif model_path == "door":
            MODEL_PATH = os.getenv('DOOR_PATH')
        elif model_path == "frs":
            MODEL_PATH = os.getenv('FACE_PATH')
        else:
            logging.error("Unknown model path type provided.")
            sys.exit(1)

        logging.debug("Using MODEL_PATH: %s", MODEL_PATH)
        ai_model = YOLO(MODEL_PATH, task="detect")
        logging.info("✅ Model loaded successfully.")
        return ai_model
    except Exception as e:
        logging.error(f"❌ Error loading YOLO model: {e}")
        sys.exit(1)

def load_clothing_model():
    logging.debug("Entering load_clothing_model()")
    try:
        logging.info("Loading YOLO model for clothing detection...")
        ai_model = YOLO(os.getenv('CLOTH_PATH'), task="detect")
        logging.info("✅ Clothing model loaded successfully.")
        return ai_model
    except Exception as e:
        logging.error(f"❌ Error loading clothing model: {e}")
        sys.exit(1)

def load_pose_model():
    logging.debug("Entering load_pose_model()")
    try:
        logging.info("Loading YOLO model for pose detection...")
        ai_model = YOLO(os.getenv('POSE_PATH'), task="detect")
        logging.info("✅ Pose model loaded successfully.")
        return ai_model
    except Exception as e:
        logging.error(f"❌ Error loading pose model: {e}")
        sys.exit(1)
