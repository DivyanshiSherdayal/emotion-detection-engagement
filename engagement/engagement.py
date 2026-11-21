"""
Converts observable visual signals into one 0-100 "estimated engagement score".

IMPORTANT: this is NOT a measure of real attention or emotion. It is a
transparent combination of three visual signals:

    score = 0.4 * face_presence + 0.3 * orientation + 0.3 * consistency

face_presence  100 if a face is visible in this frame, else 0.
orientation    How centered the face is in the frame - a rough, explainable
               proxy for "facing the camera" (NOT real head-pose tracking).
consistency    How stable the recent expression predictions are (from the
               temporal smoother). A rapidly changing result scores lower.

Why these weights? Face presence is the most reliable signal we have, so it
gets the largest share. Orientation and consistency are weaker, indirect
proxies, so they share the rest equally. They are a starting point, not a
proven psychological model - change WEIGHTS below to experiment (sum = 1.0).
"""


class EngagementCalculator:
    WEIGHTS = {"face_presence": 0.4, "orientation": 0.3, "consistency": 0.3}

    def calculate(self, face_detected, face_box, frame_shape, consistency_ratio):
        if not face_detected or face_box is None:
            return 0

        presence = 100
        orientation = self._orientation_score(face_box, frame_shape)
        consistency = max(0.0, min(1.0, consistency_ratio)) * 100

        score = (self.WEIGHTS["face_presence"] * presence
                 + self.WEIGHTS["orientation"] * orientation
                 + self.WEIGHTS["consistency"] * consistency)
        return int(round(max(0, min(100, score))))

    @staticmethod
    def _orientation_score(face_box, frame_shape):
        """1.0 when the face is horizontally centered, falling to 0 at the frame edge."""
        x, _y, w, _h = face_box
        frame_width = frame_shape[1]
        offset = abs((x + w / 2) - frame_width / 2)
        closeness = 1 - offset / (frame_width / 2)
        return max(0.0, closeness) * 100
