import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.models.hub import i3d_r50
import decord
import os

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = "mediapipe_client/models/i3d/i3d_liveness_train_head.pth"
VIDEO_PATH = "mediapipe_client/datasets/liveness/test/IMG_0280.mp4"

NUM_FRAMES = 32
CLASS_NAMES = ["FAKE", "REAL"]

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Model (same as training)
# --------------------------------------------------
def get_model(num_classes=2):
    model = i3d_r50(pretrained=False)
    model.blocks[-1].proj = nn.Linear(
        model.blocks[-1].proj.in_features,
        num_classes
    )
    return model

model = get_model()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.to(device)
model.eval()

print("Model loaded ✅")

# --------------------------------------------------
# Load video
# --------------------------------------------------
decord.bridge.set_bridge("torch")

vr = decord.VideoReader(VIDEO_PATH)
total_frames = len(vr)

indices = torch.linspace(
    0, total_frames - 1, NUM_FRAMES
).long()

video = vr.get_batch(indices)        # (T,H,W,C)
video = video.permute(3, 0, 1, 2)    # (C,T,H,W)
video = video.float() / 255.0

# Resize to 224x224
video = F.interpolate(
    video, size=(224, 224),
    mode="bilinear", align_corners=False
)

# Add batch dimension
video = video.unsqueeze(0).to(device)  # (1,C,T,H,W)

# --------------------------------------------------
# Inference
# --------------------------------------------------
with torch.no_grad():
    outputs = model(video)
    probs = torch.softmax(outputs, dim=1)

pred_class = torch.argmax(probs, dim=1).item()
confidence = probs[0, pred_class].item()

print("\n===== SINGLE VIDEO RESULT =====")
print(f"Predicted class : {CLASS_NAMES[pred_class]}")
print(f"Confidence      : {confidence:.4f}")
print(f"Raw probabilities: {probs.cpu().numpy()}")
