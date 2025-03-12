import streamlit as st
import cv2
import numpy as np
import time
from modules.facetrackingModule import FaceTracker
from modules.handtrackingModule import HandTracker

# Initialize the models
face_tracker = FaceTracker()
hand_tracker = HandTracker()

# Streamlit UI
st.title("Real-Time Face and Hand Tracking")

# Sidebar Controls
detect_hands = st.sidebar.checkbox("Enable Hand Tracking", True)
detect_faces = st.sidebar.checkbox("Enable Face Tracking", True)

# Stop button - Session state to manage stop stream
if "stop_stream" not in st.session_state:
    st.session_state.stop_stream = False

if st.sidebar.button("Stop Stream"):
    st.session_state.stop_stream = True

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error opening camera")
    st.stop()

fps_placeholder = st.empty()
frame_placeholder = st.image([])
prev_time = 0

while cap.isOpened() and not st.session_state.stop_stream:
    ret, frame = cap.read()
    if not ret:
        st.error("Error reading frame")
        break
    
    if detect_faces:
        frame = face_tracker.process_frame(frame)

    if detect_hands:
        frame = hand_tracker.process_frame(frame)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    fps_placeholder.text(f"FPS: {int(fps)}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

cap.release()
st.write("Webcam stopped.")
st.session_state.stop_stream = False