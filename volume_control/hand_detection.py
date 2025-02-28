import cv2
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import numpy as np
import subprocess

def calculate_distance(p1, p2):
    return int(np.linalg.norm(np.array(p1) - np.array(p2)))

def set_volume(volume_percentage):
    # Ensure volume is between 0 and 100
    volume_percentage = max(0, min(100, volume_percentage))
    try:
        subprocess.run(['amixer', '-D', 'pulse', 'sset', 'Master', f'{volume_percentage}%'])
    except subprocess.SubprocessError:
        print("Error setting volume. Make sure 'amixer' is installed.")

def run_hand_tracking_on_webcam():
    cap = cv2.VideoCapture(index=0)
    
    # Get initial frame to get dimensions
    _, frame = cap.read()
    h, w, _ = frame.shape

    MIN_DISTANCE = 20   
    MAX_DISTANCE = 200  

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame...")
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            index_finger_pos=None
            thumb_pos=None

           
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_finger_pos = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                    thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )

            if index_finger_pos and thumb_pos:
                distance = calculate_distance(index_finger_pos, thumb_pos)
                # Calculate volume percentage based on distance
                volume_percentage = int(np.interp(distance, [MIN_DISTANCE, MAX_DISTANCE], [0, 100]))
                set_volume(volume_percentage)
                
                cv2.line(frame, thumb_pos, index_finger_pos, (255,0,0), 2)
            
         
            frame = cv2.flip(frame, 1)
            
            if index_finger_pos and thumb_pos:
                cv2.putText(frame, f"Distance: {distance} Volume: {volume_percentage}%", 
                           (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()

if __name__ == "__main__":
    run_hand_tracking_on_webcam()