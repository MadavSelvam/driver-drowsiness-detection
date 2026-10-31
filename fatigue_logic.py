
from collections import deque

HISTORY_FRAMES = 15

SEVERE_WINDOW = 5
SEVERE_CLOSED_MIN_COUNT = 3
SEVERE_CLOSED_MIN_RATIO = 0.50

MILD_YAWN_WINDOW = 7
MILD_YAWN_MIN_COUNT = 3
MILD_YAWN_MIN_RATIO = 0.40


def eye_closed_event(left_closed, right_closed):
    return bool(left_closed and right_closed)


def yawn_event(yawn_probability, threshold=0.35):
    return yawn_probability >= threshold


def is_severe_eye_fatigue(history):
    recent = list(history)[-SEVERE_WINDOW:]
    if len(recent) < SEVERE_WINDOW:
        return False

    closed_count = sum(recent)

    return (
        closed_count >= SEVERE_CLOSED_MIN_COUNT
        and closed_count / len(recent) >= SEVERE_CLOSED_MIN_RATIO
    )


def is_repeated_yawn(history):
    recent = list(history)[-MILD_YAWN_WINDOW:]
    if len(recent) < MILD_YAWN_WINDOW:
        return False

    yawn_count = sum(recent)

    return (
        yawn_count >= MILD_YAWN_MIN_COUNT
        and yawn_count / len(recent) >= MILD_YAWN_MIN_RATIO
    )


def determine_fatigue_state(eye_history, yawn_history):
    if is_severe_eye_fatigue(eye_history):
        return "Severe"

    if is_repeated_yawn(yawn_history):
        return "Mild"

    return "Normal"
