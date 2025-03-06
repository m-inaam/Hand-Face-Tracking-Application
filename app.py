import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

st.title("Real-Time Hand & Face Tracking")

# Sidebar settings
app_mode = st.sidebar.radio("Choose an option:", ["Hand Tracking", "Face Mesh"])

class HandFaceProcessor(VideoProcessorBase):
    def __init__(self, mode):
        self.mode = mode
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.mode == "Hand Tracking":
            results = self.hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        elif self.mode == "Face Mesh":
            results = self.face_mesh.process(img_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        return frame.from_ndarray(img, format="bgr24")

# Start WebRTC Streaming
webrtc_streamer(key="realtime-tracking", video_processor_factory=lambda: HandFaceProcessor(app_mode))
