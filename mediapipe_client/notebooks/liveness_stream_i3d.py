# =========================================================
# LIVENESS DETECTION: MediaPipe + I3D (LIVE STREAM)
# =========================================================

import cv2
import time
import math
import os
import requests
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.models.hub import i3d_r50
import decord

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================================
# CONFIG
# =========================================================

API_BASE_URL = "http://127.0.0.1:8000/mediapipe"

FACE_MODEL_PATH = "mediapipe_client/models/face_landmarker.task"
I3D_MODEL_PATH = "mediapipe_client/models/i3d/i3d_liveness_train_head.pth"

NUM_I3D_FRAMES = 32
I3D_REAL_THRESHOLD = 0.70
I3D_FAKE_THRESHOLD = 0.60

EAR_THRESHOLD = 0.25
HEAD_UP_THRESHOLD = -0.03
HEAD_LEFT_THRESHOLD = -0.03
HEAD_RIGHT_THRESHOLD = 0.03

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# I3D SETUP
# =========================================================

def load_i3d():
    model = i3d_r50(pretrained=False)
    model.blocks[-1].proj = nn.Linear(
        model.blocks[-1].proj.in_features, 2
    )
    model.load_state_dict(
        torch.load(I3D_MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval().to(DEVICE)
    print("✅ I3D model loaded")
    return model

i3d_model = load_i3d()

FRAME_BUFFER = deque(maxlen=NUM_I3D_FRAMES)
decord.bridge.set_bridge("torch")

# =========================================================
# MEDIAPIPE SETUP
# =========================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_INDEX = 1

# =========================================================
# STATE
# =========================================================

instruction_queue = ["left_eye_blink", "right_eye_blink", "head_up"]
current_instruction_index = 0
session_complete = False

left_eye_closed = False
right_eye_closed = False
head_up_detected = False

STREAM_ID = None
last_i3d_time = 0
i3d_real_prob = 0.0
i3d_fake_prob = 0.0

# =========================================================
# HELPERS
# =========================================================

def create_stream():
    global STREAM_ID
    r = requests.post(f"{API_BASE_URL}/create-stream")
    STREAM_ID = r.json()["stream_id"]
    print("Stream created:", STREAM_ID)

def send_instruction_update(instruction):
    if STREAM_ID:
        requests.post(
            f"{API_BASE_URL}/update",
            params={"stream_id": STREAM_ID, "instruction": instruction}
        )

def distance(p1, p2):
    return math.dist([p1.x, p1.y], [p2.x, p2.y])

def eye_aspect_ratio(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * distance(p1, p4))

def get_current_instruction():
    if current_instruction_index < len(instruction_queue):
        return instruction_queue[current_instruction_index]
    return None

def advance_instruction():
    global current_instruction_index, session_complete
    current_instruction_index += 1
    if current_instruction_index >= len(instruction_queue):
        session_complete = True
        print("✅ ACTION LIVENESS COMPLETE")

# =========================================================
# MEDIAPIPE CALLBACK
# =========================================================

def on_landmarks(result, output_image, timestamp_ms):
    global left_eye_closed, right_eye_closed, head_up_detected

    if session_complete or not result.face_landmarks:
        return

    landmarks = result.face_landmarks[0]
    instruction = get_current_instruction()

    # -------- Blink --------
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)

    left_closed = left_ear < EAR_THRESHOLD
    right_closed = right_ear < EAR_THRESHOLD

    if instruction == "left_eye_blink":
        if left_closed and not left_eye_closed:
            left_eye_closed = True
        elif not left_closed and left_eye_closed:
            send_instruction_update("left_eye_blink")
            left_eye_closed = False
            advance_instruction()

    if instruction == "right_eye_blink":
        if right_closed and not right_eye_closed:
            right_eye_closed = True
        elif not right_closed and right_eye_closed:
            send_instruction_update("right_eye_blink")
            right_eye_closed = False
            advance_instruction()

    # -------- Head up --------
    nose = landmarks[NOSE_INDEX]
    eye_mid_y = (landmarks[33].y + landmarks[263].y) / 2
    delta_y = nose.y - eye_mid_y

    if instruction == "head_up" and delta_y < HEAD_UP_THRESHOLD and not head_up_detected:
        send_instruction_update("head_up")
        head_up_detected = True
        advance_instruction()

# =========================================================
# MEDIAPIPE OPTIONS
# =========================================================

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_landmarks
)

# =========================================================
# MAIN LOOP
# =========================================================

create_stream()

with FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        # -------- I3D BUFFER --------
        resized = cv2.resize(rgb, (224, 224))
        FRAME_BUFFER.append(resized)

        # Run I3D every 1.5 sec
        if len(FRAME_BUFFER) == NUM_I3D_FRAMES and time.time() - last_i3d_time > 1.5:
            clip = torch.from_numpy(
                np.stack(FRAME_BUFFER)
            ).permute(3, 0, 1, 2).float() / 255.0
            clip = clip.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                probs = torch.softmax(i3d_model(clip), dim=1)

            i3d_fake_prob = probs[0, 0].item()
            i3d_real_prob = probs[0, 1].item()
            last_i3d_time = time.time()

        # -------- UI --------
        cv2.putText(frame, f"I3D REAL: {i3d_real_prob:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"I3D FAKE: {i3d_fake_prob:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        if session_complete and i3d_real_prob > I3D_REAL_THRESHOLD:
            cv2.putText(frame, "LIVENESS PASSED", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
            cv2.imshow("Liveness", frame)
            cv2.waitKey(2000)
            break

        if i3d_fake_prob > I3D_FAKE_THRESHOLD:
            cv2.putText(frame, "SPOOF DETECTED", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        cv2.imshow("Liveness", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
