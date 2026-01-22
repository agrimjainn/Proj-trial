from mediapipe_client.constants import *
from mediapipe_client.geometry import eye_aspect_ratio
from mediapipe_client.instructions import get_current_instruction, advance_instruction
import mediapipe_client.state as state

def process_landmarks(landmarks):
    if state.session_complete:
        return None

    nose = landmarks[NOSE_INDEX]
    left_eye_center = landmarks[33]
    right_eye_center = landmarks[263]

    ref_x = (left_eye_center.y + right_eye_center.y) / 2
    ref_y = (left_eye_center.x + right_eye_center.x) / 2

    delta_y = nose.y - ref_y
    delta_x = nose.x - ref_x

    current = get_current_instruction()

    if current == "head_up" and delta_y < HEAD_UP_THRESHOLD and not state.head_up_detected:
        state.head_up_detected = True
        advance_instruction()
        return "head_up"

    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)

    if current == "left_eye_blink" and left_ear < EAR_THRESHOLD:
        advance_instruction()
        return "left_eye_blink"

    if current == "right_eye_blink" and right_ear < EAR_THRESHOLD:
        advance_instruction()
        return "right_eye_blink"

    return None
