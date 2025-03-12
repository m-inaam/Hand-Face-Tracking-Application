import cv2
import mediapipe as mp
import logging
import time

class FaceTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize the logging system
        logging.basicConfig(level=logging.INFO)

        # Initialize the MediaPipe face mesh model
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.face_mesh_drawing_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)

        # Open the default camera
        self.cap = cv2.VideoCapture(0)

        # Check if the camera is opened successfully
        if not self.cap.isOpened():
            logging.error("Error opening camera")
            exit()

    def track_faces(self):
        # Initialize the FPS variables
        prev_time = 0
        curr_time = 0

        # Continuously read frames from the camera
        while True:
            # Capture a frame
            ret, frame = self.cap.read()

            # Check if the frame was read successfully
            if not ret:
                logging.error("Error reading frame")
                break

            # Convert the frame to RGB format for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces and facial landmarks in the frame
            results = self.face_mesh.process(frame_rgb)

            # Draw the facial landmarks on the frame
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_draw.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        #connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.face_mesh_drawing_spec,
                        connection_drawing_spec=self.face_mesh_drawing_spec
                    )

            # Calculate the FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Display the FPS in the frame
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Face Tracking', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close all windows
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_tracker = FaceTracker()
    face_tracker.track_faces()