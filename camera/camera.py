"""
Webcam access + OpenCV face detection.

Design note (good viva answer): ONE background thread reads the webcam.
It publishes the newest annotated JPEG here, and the web page simply streams
whatever the latest JPEG is. This avoids two parts of the program fighting
over the same camera.
"""

import threading

import cv2
import numpy as np


class Camera:
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(
                "Face detector file could not be loaded: " + cascade_path +
                "\nFix: pip install opencv-python==4.10.0.84"
            )
        self._lock = threading.Lock()
        self._latest_jpeg = self._make_placeholder("Starting camera...")

    # ---------- reading ----------
    def is_available(self):
        return self.capture is not None and self.capture.isOpened()

    def read_frame(self):
        """Returns one mirrored BGR frame, or None if the camera fails."""
        if not self.is_available():
            return None
        success, frame = self.capture.read()
        if not success or frame is None:
            return None
        return cv2.flip(frame, 1)  # mirror view feels natural on a dashboard

    # ---------- face detection ----------
    def detect_face(self, frame):
        """
        Haar-cascade face detection (built into OpenCV - no download needed).
        Returns (largest_face_box, face_count); box is (x, y, w, h) or None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None, 0

        # MVP = one person: use the largest face. We still COUNT all faces, so
        # multi-person support can be added later without redesigning.
        largest = max(faces, key=lambda f: int(f[2]) * int(f[3]))
        x, y, w, h = (int(v) for v in largest)
        return (x, y, w, h), len(faces)

    @staticmethod
    def crop_face(frame, face_box, margin=0.2):
        """Crops the face (plus a small margin) so the AI model sees only the face."""
        x, y, w, h = face_box
        pad_x, pad_y = int(w * margin), int(h * margin)
        top, left = max(0, y - pad_y), max(0, x - pad_x)
        bottom = min(frame.shape[0], y + h + pad_y)
        right = min(frame.shape[1], x + w + pad_x)
        return frame[top:bottom, left:right]

    # ---------- publishing for the live view ----------
    def publish(self, frame, face_box, face_count, label=""):
        """Draws the face box + label on a copy of the frame and stores it as JPEG."""
        view = frame.copy()
        if face_box is not None:
            x, y, w, h = face_box
            green = (94, 190, 45)
            cv2.rectangle(view, (x, y), (x + w, y + h), green, 2)
            if label:
                cv2.rectangle(view, (x, max(0, y - 28)), (x + max(w, 170), y), green, -1)
                cv2.putText(view, label, (x + 6, max(14, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2, cv2.LINE_AA)
        if face_count > 1:
            cv2.putText(view, "Multiple faces - analyzing the largest", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        ok, buffer = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._latest_jpeg = buffer.tobytes()

    def publish_placeholder(self, text):
        jpeg = self._make_placeholder(text)
        with self._lock:
            self._latest_jpeg = jpeg

    def get_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    @staticmethod
    def _make_placeholder(text):
        image = np.full((480, 640, 3), (250, 244, 240), dtype=np.uint8)
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(image, text, (max(10, (640 - text_w) // 2), 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (110, 90, 70), 2, cv2.LINE_AA)
        ok, buffer = cv2.imencode(".jpg", image)
        return buffer.tobytes() if ok else b""

    def release(self):
        if self.capture is not None:
            self.capture.release()
