import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import time

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

def process_frame(frame, detect_hands, detect_faces, hands, face_mesh):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if detect_hands:
        hand_results = hands.process(frame_rgb)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    if detect_faces:
        face_results = face_mesh.process(frame_rgb)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_draw.DrawingSpec(thickness=1, circle_radius=1)
                )
    return frame

# Streamlit UI
st.title("Real-Time Face and Hand Tracking")

# Sidebar Controls
detect_hands = st.sidebar.checkbox("Enable Hand Tracking", True)
detect_faces = st.sidebar.checkbox("Enable Face Tracking", True)

# Stop button - Added a session state flag to manage the loop
if "stop_stream" not in st.session_state:
    st.session_state.stop_stream = False

if st.sidebar.button("Stop Stream"):
    st.session_state.stop_stream = True

# Initialize MediaPipe models
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
    
    frame = process_frame(frame, detect_hands, detect_faces, hands, face_mesh)
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    fps_placeholder.text(f"FPS: {int(fps)}")
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

cap.release()
st.write("Webcam stopped.")