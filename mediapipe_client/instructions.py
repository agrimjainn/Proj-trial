from mediapipe_client.state import current_instruction_index, session_complete

instruction_queue = [
    "left_eye_blink",
    "right_eye_blink",
    "head_up"
]

def get_current_instruction():
    if current_instruction_index < len(instruction_queue):
        return instruction_queue[current_instruction_index]
    return None

def advance_instruction():
    global current_instruction_index, session_complete
    current_instruction_index += 1
    if current_instruction_index >= len(instruction_queue):
        session_complete = True
