import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

st.title("Real-Time Hand & Face Tracking")

# Sidebar settings
app_mode = st.sidebar.radio("Choose an option:", ["Hand Tracking", "Face Mesh"])
draw_landmarks = st.sidebar.checkbox("Show Landmarks", value=True)  # Option to toggle drawing
min_detection_confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)
min_tracking_confidence = st.sidebar.slider("Tracking Confidence", 0.1, 1.0, 0.5)

class HandFaceProcessor(VideoProcessorBase):
    def __init__(self, mode, draw_landmarks, min_detection_confidence, min_tracking_confidence):
        self.mode = mode
        self.draw_landmarks = draw_landmarks

        # Load the required model dynamically
        self.hands = None
        self.face_mesh = None

        if self.mode == "Hand Tracking":
            self.hands = mp_hands.Hands(
                max_num_hands=2, 
                min_detection_confidence=min_detection_confidence, 
                min_tracking_confidence=min_tracking_confidence
            )

        elif self.mode == "Face Mesh":
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1, 
                min_detection_confidence=min_detection_confidence, 
                min_tracking_confidence=min_tracking_confidence
            )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mode == "Hand Tracking" and self.hands:
            results = self.hands.process(img_rgb)
            if results.multi_hand_landmarks and self.draw_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        elif self.mode == "Face Mesh" and self.face_mesh:
            results = self.face_mesh.process(img_rgb)
            if results.multi_face_landmarks and self.draw_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        return frame.from_ndarray(img, format="bgr24")

# Start WebRTC Streaming with updated parameters
webrtc_streamer(
    key="realtime-tracking",
    video_processor_factory=lambda: HandFaceProcessor(
        app_mode, draw_landmarks, min_detection_confidence, min_tracking_confidence
    )
)
