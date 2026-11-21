"""
Wraps DeepFace so the rest of the app only ever sees a simple dictionary,
never DeepFace's internal format. Also implements temporal smoothing.
"""

import os

# DeepFace needs the older Keras 2 API; the `tf-keras` package provides it.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from deepface import DeepFace


class EmotionDetector:
    """Runs DeepFace's PRE-TRAINED emotion model on one cropped face."""

    def warm_up(self):
        """
        The first call downloads/loads the model weights (slow, needs internet
        once). Doing it at startup keeps the first real frame fast.
        Returns True if the model is ready.
        """
        try:
            blank_face = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.analyze(blank_face, actions=["emotion"],
                             enforce_detection=False, detector_backend="skip",
                             silent=True)
            return True
        except Exception as error:
            print(f"[EmotionDetector] Model failed to load: {error}")
            return False

    def analyze(self, face_image):
        """
        Returns {"emotion": "Happy", "confidence": 82.4, "scores": {...}}
        or None if the frame could not be analyzed (we never crash on bad frames).
        """
        if face_image is None or face_image.size == 0:
            return None
        try:
            result = DeepFace.analyze(face_image, actions=["emotion"],
                                      enforce_detection=False,
                                      detector_backend="skip", silent=True)
            if isinstance(result, list):
                result = result[0]

            scores = {name.capitalize(): float(value)
                      for name, value in result["emotion"].items()}
            top_emotion = result["dominant_emotion"].capitalize()
            return {"emotion": top_emotion,
                    "confidence": scores[top_emotion],  # model probability, 0-100
                    "scores": scores}
        except Exception as error:
            print(f"[EmotionDetector] Skipped a frame: {error}")
            return None
