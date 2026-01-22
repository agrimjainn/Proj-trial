from fastapi import APIRouter
from uuid import uuid4

router = APIRouter(prefix="/mediapipe")

ALL_INSTRUCTIONS = [
    "head_up",
    "head_left",
    "head_right",
    "left_eye_blink",
    "right_eye_blink"
]

#Temporary in memory
STREAM_STORE = {}

# CREATING STREAM AND SELECT INSTRUCTIONS

@router.post("/create-stream")
def create_stream():
    stream_id = str(uuid4())

    # TODO: Change the fixed to random logic
    selected = ["left_eye_blink", "right_eye_blink", "head_up"]

    expected = {i:i in selected for i in ALL_INSTRUCTIONS}
    count = {i:0 for i in ALL_INSTRUCTIONS}

    STREAM_STORE[stream_id] = {
        "expected": expected,
        "count": count,
        "completed": False
    }

    return {
        "stream_id": stream_id,
        "expected": expected
    }


# UPDATING COUNTS FROM MEDIAPIPE
@router.post("/update")
def update_instruction(stream_id: str, instruction: str):
    stream = STREAM_STORE.get(stream_id)

    if not stream:
        return {"error":"Invalud stream_id"}
    
    if not stream["expected"].get(instruction):
        return {"message":"Instruction not expected"}
    
    stream["count"][instruction] += 1

    return {
        "instruction":instruction,
        "count":stream["count"][instruction]
    }

# CHECK COMPLITION STATUS
@router.get("/status/{stream_id}")
def check_status(stream_id: str):
    stream = STREAM_STORE.get(stream_id)

    if not stream:
        return {"error": "Invalid stream_id"}
    
    for instr, required in stream["expected"].items():
        if required and stream["count"][instr] < 1:
            return{
                "completed": False,
                "expected": stream["expected"],
                "count": stream["count"]
            }
        
    stream["completed"] = True

    return{
        "completed": True,
        "expected": stream["expected"],
        "count": stream["count"]
    }