import cv2
import mediapipe as mp
from mediapipe_client.landmarker import create_landmarker
from mediapipe_client.detector import process_landmarks

class WebcamProcessor:
    def __init__(self, model_path):
        self.mp_face = mp.tasks.vision.FaceLandmarker
        self.latest_result = None

        def callback(result, image, timestamp_ms):
            if result.face_landmarks:
                self.latest_result = process_landmarks(
                    result.face_landmarks[0]
                )
        
        options = create_landmarker(model_path, callback)
        self.landmarker = self.mp_face.create_from_options(options)

    def process_frame(self, frame):
        mp_image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data=frame
        )

        self.landmarker.detect_async(mp_image, int(cv2.getTickCount()))
        return self.latest_result