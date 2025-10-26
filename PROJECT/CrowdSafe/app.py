import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import datetime
import firebase_admin
from firebase_admin import credentials, db
from math import sqrt
import os
import pandas as pd
from twilio.rest import Client
import random
import uuid

# --- Page Configuration ---
st.set_page_config(
    page_title="CrowdSafe Authority Dashboard",
    page_icon="🚨",
    layout="wide"
)

# --- Configuration (EDIT THESE VALUES) ---
# Replace placeholders with your actual data.
# WARNING: Do not commit this file to a public GitHub repository with real credentials.

# File Paths and Location
VIDEO_PATH = "local_train.mp4"
FIREBASE_KEY_PATH = "firebase-credentials.json"
LOCATION_NAME = "Potheri Railway Station"

# Firebase Database URL
FIREBASE_DB_URL = "YOUR_FIREBASE_DB_URL_HERE"

# Twilio Credentials
TWILIO_ACCOUNT_SID = "YOUR_TWILIO_ACCOUNT_SID_HERE"
TWILIO_AUTH_TOKEN = "YOUR_TWILIO_AUTH_TOKEN_HERE"
TWILIO_PHONE_NUMBER = "YOUR_TWILIO_PHONE_NUMBER_HERE"

# Unified Authority Contact List
AUTHORITY_NUMBERS = [
    "+11234567890", # Replace with actual authority numbers
    "+10987654321"
]


# --- Service Initialization ---
@st.cache_resource
def init_firebase():
    if FIREBASE_DB_URL == "YOUR_FIREBASE_DB_URL_HERE" or not os.path.exists(FIREBASE_KEY_PATH):
        return False
    try:
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
        return True
    except Exception as e:
        st.error(f"Firebase initialization failed: {e}")
        return False

@st.cache_resource
def load_yolo_model():
    try: 
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

# --- Core Functions ---
def send_sms_dispatch(message, to_numbers):
    if TWILIO_ACCOUNT_SID == "YOUR_TWILIO_ACCOUNT_SID_HERE" or not to_numbers:
        st.warning("Twilio credentials or authority numbers are not configured. Cannot send SMS.")
        return
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        for number in to_numbers:
            client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=number)
        st.toast(f"✅ Alert dispatched successfully.")
    except Exception as e:
        st.error(f"Twilio Error: {e}")

def analyze_crowd_density(boxes, proximity_threshold, cluster_threshold):
    centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes]
    num_people = len(centers)
    if num_people < 2: return 0, [False] * num_people
    neighbor_counts = [0] * num_people
    for i in range(num_people):
        for j in range(i + 1, num_people):
            if sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2) < proximity_threshold:
                neighbor_counts[i] += 1; neighbor_counts[j] += 1
    in_cluster = [count >= cluster_threshold for count in neighbor_counts]
    return sum(in_cluster), in_cluster

def get_crowd_level(dense_count, high_threshold):
    if dense_count > high_threshold: return "High"
    if dense_count > high_threshold / 2: return "Moderate"
    return "Low"

# --- Main Application UI ---
st.markdown("<h1 style='text-align: center; color: white;'>🚨 CrowdSafe Authority Dispatch Dashboard</h1>", unsafe_allow_html=True)

st.session_state.source_selection = st.selectbox(
    "**Select Data Source**",
    ("Live Video Analysis", "IoT Sensor Simulation")
)
st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("🔧 Control Panel")
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.4, 0.05, disabled=(st.session_state.source_selection == "IoT Sensor Simulation"))
st.sidebar.markdown("---")
st.sidebar.subheader("Density Cluster Settings")
proximity_threshold = st.sidebar.slider("Proximity Threshold (px)", 10, 150, 50, 5, disabled=(st.session_state.source_selection == "IoT Sensor Simulation"))
cluster_size_threshold = st.sidebar.slider("Cluster Size (neighbors)", 1, 10, 3, 1, disabled=(st.session_state.source_selection == "IoT Sensor Simulation"))
high_alert_threshold = st.sidebar.slider("High Density Alert Trigger", 5, 50, 10, 1)

st.sidebar.markdown("---")
firebase_initialized = init_firebase()
st.sidebar.info("Firebase Status: " + ("✅ Connected" if firebase_initialized else "❌ Disconnected/Not Configured"))

# --- Dashboard Layout ---
col1, col2 = st.columns([3, 2])
with col1:
    st.header("📹 Live Feed" if st.session_state.source_selection == "Live Video Analysis" else "📡 IoT Sensor Feed")
    frame_placeholder = st.empty()
with col2:
    st.header("📊 Real-Time Statistics")
    stats_placeholder = st.empty()
    st.header("🔔 Incident Status")
    response_placeholder = st.empty()

start_button = st.button(f"🚀 Start {st.session_state.source_selection}", type="primary")

if start_button:
    model = load_yolo_model()
    st.session_state.stop = False
    st.session_state.incident = None 

    st.button('🛑 Stop Analysis', on_click=lambda: setattr(st.session_state, 'stop', True))
    
    total_count, dense_count = 0, 0
    cap = None
    last_fb_update_time = time.time()
    
    if st.session_state.source_selection == "Live Video Analysis":
        if model and os.path.exists(VIDEO_PATH):
            cap = cv2.VideoCapture(VIDEO_PATH)
        else:
            st.error(f"Model loaded, but video file not found at '{VIDEO_PATH}'. Please check the path."); st.stop()

    # --- Main Processing Loop for both modes ---
    while not st.session_state.get('stop', False):
        if st.session_state.source_selection == "Live Video Analysis":
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (960, 540))
            results = model.predict(frame, imgsz=640, conf=confidence_threshold, classes=[0], verbose=False)
            boxes = [list(map(int, b.xyxy[0])) for r in results for b in r.boxes if int(b.cls[0]) == 0]
            total_count = len(boxes)
            dense_count, in_cluster = analyze_crowd_density(boxes, proximity_threshold, cluster_size_threshold)
            for i, box in enumerate(boxes):
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,0,255) if in_cluster[i] else (0,255,0), 2)
            frame_placeholder.image(frame, channels="BGR")

        elif st.session_state.source_selection == "IoT Sensor Simulation":
            if 'iot_count' not in st.session_state: st.session_state.iot_count = 5
            st.session_state.iot_count = max(0, st.session_state.iot_count + random.choice([-2,-1,0,1,1,2,3]))
            total_count, dense_count = st.session_state.iot_count, st.session_state.iot_count
            frame_placeholder.image("https://placehold.co/960x540/2d3748/ffffff?text=IoT+Sensor+Simulation\n\nLive+Data+Feed+Active", caption="Virtual data feed simulating physical sensors.")

        crowd_level = get_crowd_level(dense_count, high_alert_threshold)
        
        # --- Incident Management and Alerting Logic ---
        if crowd_level == "High" and st.session_state.incident is None:
            incident_id = str(uuid.uuid4())
            st.session_state.incident = {"id": incident_id, "dense_count": dense_count}
            alert_message = f"Urgent Alert: High-density crowd at {LOCATION_NAME}. Risk of stampede with {dense_count} individuals. Incident ID: {incident_id}"
            send_sms_dispatch(alert_message, AUTHORITY_NUMBERS)
        
        elif crowd_level != "High" and st.session_state.incident is not None:
            st.session_state.incident = None

        # --- Update UI Placeholders ---
        with stats_placeholder.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("Total People", total_count)
            c2.metric("Dense Cluster", dense_count)
            c3.metric("Status", crowd_level)

        # --- Simplified Incident Status Display ---
        with response_placeholder.container():
            incident = st.session_state.get('incident')
            if incident:
                st.info(f"**Active Incident:** Alert sent for incident {incident['id'][:6]}...")
                st.metric("Status", "Alert Dispatched to Authorities")
            else:
                st.success("No active high-density incidents.")
        
        # --- Update Firebase and Loop Delay ---
        if time.time() - last_fb_update_time > 2 and firebase_initialized:
            try:
                db.reference('current_status').set({'count': dense_count, 'status': crowd_level, 'last_update': datetime.datetime.now().isoformat()})
                last_fb_update_time = time.time()
            except Exception as e:
                pass # Fail silently if Firebase is not configured
        
        time.sleep(1 if st.session_state.source_selection == "IoT Sensor Simulation" else 0.01)

    if cap: cap.release()
    st.info("Analysis stopped.")