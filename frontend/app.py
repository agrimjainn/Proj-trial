import sys
import os
import streamlit as st
st.write(sys.path)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import cv2
import mediapipe as mp

from mediapipe_client.detector import process_landmarks
import mediapipe_client.state as state
from mediapipe_client.instructions import get_current_instruction

st.set_page_config(page_title="Liveness Detection", layout="centered")
st.title("🧠 Liveness Detection (Streamlit)")

frame_placeholder = st.empty()
status_placeholder = st.empty()

start = st.button("▶ Start")
stop = st.button("⛔ Stop")

if "running" not in st.session_state:
    st.session_state.running = False

if start:
    st.session_state.running = True

if stop:
    st.session_state.running = False

# -------- MediaPipe setup (ONCE) --------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "mediapipe_client/models/face_landmarker.task"

def mp_callback(result, image, timestamp_ms):
    if result.face_landmarks:
        process_landmarks(result.face_landmarks[0])

if st.session_state.running:

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=mp_callback,
        num_faces=1
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)

        while cap.isOpened() and st.session_state.running:
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            # UI overlays
            instruction = get_current_instruction()
            if instruction:
                cv2.putText(
                    frame,
                    f"Instruction: {instruction.replace('_',' ').upper()}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,255),
                    2
                )

            cv2.putText(
                frame,
                f"Step {state.current_instruction_index + 1}",
                (30,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

            frame_placeholder.image(frame, channels="BGR")

            if state.session_complete:
                status_placeholder.success("✅ LIVENESS SUCCESS")
                time.sleep(2)
                st.session_state.running = False
                break

            time.sleep(0.01)  # CRITICAL: allow async callback to fire

        cap.release()
