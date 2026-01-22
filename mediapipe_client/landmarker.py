import mediapipe as mp

# Aliases for clarity
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def create_landmarker(model_path: str, callback):
    """
    Creates MediaPipe FaceLandmarkerOptions for LIVE_STREAM mode
    """

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=callback,
        num_faces=1
    )

    return options
