import math

def distance(p1, p2):
    return math.dist([p1.x, p1.y], [p2.x, p2.y])

def eye_aspect_ratio(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    vertical1 = distance(p2, p6)
    vertical2 = distance(p3, p5)
    horizontal = distance(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)
