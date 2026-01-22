import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pytorchvideo.models.hub import i3d_r50
import decord
from tqdm import tqdm

MODEL_PATH = "mediapipe_client/models/i3d/i3d_liveness_train_head.pth"
VAL_PATH = "mediapipe_client/datasets/liveness/val"

NUM_FRAMES = 32
BATCH_SIZE = 2

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Dataset (NO transforms here)
# --------------------------------------------------
decord.bridge.set_bridge("torch")

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=32):
        self.samples = []
        self.num_frames = num_frames
        self.label_map = {"real": 1, "fake": 0}

        for cls in ["real", "fake"]:
            class_dir = os.path.join(root_dir, cls)
            for video in os.listdir(class_dir):
                self.samples.append(
                    (os.path.join(class_dir, video), self.label_map[cls])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        vr = decord.VideoReader(video_path)
        total_frames = len(vr)

        indices = torch.linspace(
            0, total_frames - 1, self.num_frames
        ).long()

        video = vr.get_batch(indices)       # (T, H, W, C)
        video = video.permute(3, 0, 1, 2)   # (C, T, H, W)
        video = video.float() / 255.0

        # Resize to 224x224 (I3D requirement)
        video = F.interpolate(
            video, size=(224, 224),
            mode="bilinear", align_corners=False
        )

        return video, torch.tensor(label)

# --------------------------------------------------
# Model definition (MUST match training)
# --------------------------------------------------
def get_model(num_classes=2):
    model = i3d_r50(pretrained=False)
    model.blocks[-1].proj = nn.Linear(
        model.blocks[-1].proj.in_features,
        num_classes
    )
    return model

model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully ✅")

# --------------------------------------------------
# Validation loader
# --------------------------------------------------
val_dataset = VideoDataset(VAL_PATH, NUM_FRAMES)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

criterion = nn.CrossEntropyLoss()

# --------------------------------------------------
# Validation loop
# --------------------------------------------------
total_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for videos, labels in tqdm(val_loader, desc="Validating"):
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

val_loss = total_loss / len(val_loader)
val_acc = correct / total

print("\n===== VALIDATION RESULT =====")
print(f"Validation Loss     : {val_loss:.4f}")
print(f"Validation Accuracy : {val_acc:.4f}")