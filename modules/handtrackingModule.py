import cv2
import mediapipe as mp
import logging
import time

class HandTracker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize the logging system
        logging.basicConfig(level=logging.INFO)

        # Initialize the MediaPipe hand detection model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Open the default camera
        self.cap = cv2.VideoCapture(0)

        # Check if the camera is opened successfully
        if not self.cap.isOpened():
            logging.error("Error opening camera")
            exit()

    def track_hands(self):
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

            # Detect hands in the frame
            results = self.hands.process(frame_rgb)

            # Draw the hand landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get the hand position
                    hand_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                    hand_y = int(hand_landmarks.landmark[0].y * frame.shape[0])
                    logging.info(f"Hand {hand_id} position: ({hand_x}, {hand_y})")

                    # Draw the hand ID on the frame
                    cv2.putText(frame, f"Hand {hand_id}", (int(hand_x), int(hand_y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Calculate the FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Display the FPS in the frame
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Hand Tracking', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close all windows
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_tracker = HandTracker()
    hand_tracker.track_hands()