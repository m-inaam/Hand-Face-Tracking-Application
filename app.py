import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

# Initialize MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Streamlit App Title
st.title("Hand & Face Tracking with MediaPipe")

# Sidebar settings
st.sidebar.title("Select Mode")
app_mode = st.sidebar.radio("Choose an option:", ["About App", "Hand Tracking", "Face Mesh"])

# About App
if app_mode == "About App":
    st.markdown("""
    ## About This App
    This Streamlit application combines **Hand Tracking** and **Face Mesh Recognition** using **MediaPipe**.
    
    - The **Hand Tracking** module detects and tracks hands in real-time.
    - The **Face Mesh** module detects facial landmarks from images or video.
    """)

# Hand Tracking Module
elif app_mode == "Hand Tracking":
    st.subheader("Hand Tracking using MediaPipe")
    
    # Hand Tracking settings
    max_hands = st.sidebar.slider("Max Hands", 1, 2, 2)
    detection_conf = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5)
    tracking_conf = st.sidebar.slider("Tracking Confidence", 0.0, 1.0, 0.5)

    use_webcam = st.sidebar.checkbox("Use Webcam", True)
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    
    tffile = tempfile.NamedTemporaryFile(delete=False)
    
    if video_file_buffer:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)
    else:
        vid = cv2.VideoCapture(0 if use_webcam else "demo.mp4")

    stframe = st.empty()
    
    with mp_hands.Hands(
        max_num_hands=max_hands,
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf
    ) as hands:
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS
                    )

            stframe.image(frame, channels="BGR", use_column_width=True)

        vid.release()

# Face Mesh Module
elif app_mode == "Face Mesh":
    st.subheader("Face Mesh using MediaPipe")
    
    mode = st.sidebar.selectbox("Choose Mode", ["Run on Image", "Run on Video"])
    
    max_faces = st.sidebar.number_input("Max Faces", value=1, min_value=1)
    detection_conf = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5)
    tracking_conf = st.sidebar.slider("Tracking Confidence", 0.0, 1.0, 0.5)

    if mode == "Run on Image":
        img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if img_file_buffer:
            image = np.array(Image.open(img_file_buffer))
        else:
            st.warning("Upload an image to proceed.")
            st.stop()

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            min_detection_confidence=detection_conf
        ) as face_mesh:
            results = face_mesh.process(image)
            out_image = image.copy()
            
            for face_landmarks in results.multi_face_landmarks or []:
                mp_drawing.draw_landmarks(
                    image=out_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION
                )
            
            st.image(out_image, use_column_width=True)

    elif mode == "Run on Video":
        use_webcam = st.sidebar.checkbox("Use Webcam", True)
        video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        
        tffile = tempfile.NamedTemporaryFile(delete=False)
        
        if video_file_buffer:
            tffile.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tffile.name)
        else:
            vid = cv2.VideoCapture(0 if use_webcam else "demo.mp4")

        stframe = st.empty()
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        ) as face_mesh:
            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame)
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION
                        )

                stframe.image(frame, channels="BGR", use_column_width=True)

            vid.release()
