from fastapi import FastAPI
from app.api.mediapipe_routes import router

app = FastAPI(title="MediaPipe Liveness API")

app.include_router(router)

@app.get("/")
def root():
    return {"status":"API running"}