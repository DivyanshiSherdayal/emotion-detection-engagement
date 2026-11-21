"""Easy-to-change settings in one place (no hard-coded numbers elsewhere)."""

import os

_source = os.environ.get("CAMERA_SOURCE", "0")
CAMERA_SOURCE = int(_source) if _source.isdigit() else _source   # 0 = default webcam

SMOOTHING_WINDOW = 5                 # how many recent predictions are smoothed together
EMOTION_INTERVAL_SECONDS = 0.7       # run the AI model about this often
DB_WRITE_INTERVAL_SECONDS = 2        # save one analytics row this often during a session
FORGET_AFTER_NO_FACE_SECONDS = 3     # clear smoothing memory after the face is gone this long
