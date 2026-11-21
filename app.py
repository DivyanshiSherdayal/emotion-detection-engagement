"""
Flask backend for: AI-Based Emotion Detection and Student Engagement Analysis.

Pipeline (same as the workflow slide):
  Webcam -> Face detection (OpenCV) -> Pre-trained model (DeepFace)
  -> Temporal smoothing -> Estimated engagement -> SQLite -> Dashboard

Run:  python app.py      then open  http://127.0.0.1:5000
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import csv
import io
import threading
import time
from datetime import datetime

from flask import Flask, Response, jsonify, render_template, request

import config
from camera.camera import Camera
from database import db
from emotion.detector import EmotionDetector
from emotion.smoothing import TemporalSmoother
from engagement.engagement import EngagementCalculator

app = Flask(__name__)

# ---------------- shared state (background thread writes, routes read) -------
state_lock = threading.Lock()
state = {
    "camera_available": False,
    "model_status": "loading",      # loading | ready | error
    "face_detected": False,
    "face_count": 0,
    "emotion": "N/A",
    "confidence": 0.0,
    "scores": {},
    "engagement_score": 0,
    "session_active": False,
    "session_id": None,
    "session_start_epoch": None,
}

camera = Camera(config.CAMERA_SOURCE)
detector = EmotionDetector()
smoother = TemporalSmoother(window_size=config.SMOOTHING_WINDOW)
engagement_calc = EngagementCalculator()


def update_state(**changes):
    with state_lock:
        state.update(changes)


def processing_loop():
    """Background thread: read frame -> detect face -> analyze -> score -> save."""
    update_state(model_status="loading")
    model_ready = detector.warm_up()
    update_state(model_status="ready" if model_ready else "error")

    last_analysis = 0.0
    last_db_write = 0.0
    no_face_since = None

    while True:
        if not camera.is_available():
            camera.publish_placeholder("Camera not available")
            update_state(camera_available=False, face_detected=False, face_count=0,
                         emotion="N/A", engagement_score=0)
            time.sleep(1.0)
            continue

        frame = camera.read_frame()
        if frame is None:  # invalid / dropped frame: skip it, never crash
            time.sleep(0.2)
            continue

        face_box, face_count = camera.detect_face(frame)
        face_detected = face_box is not None
        now = time.time()

        # Forget old predictions if the face has been gone for a while.
        if face_detected:
            no_face_since = None
        else:
            no_face_since = no_face_since or now
            if now - no_face_since > config.FORGET_AFTER_NO_FACE_SECONDS:
                smoother.reset()

        # Run the (slower) AI model only every EMOTION_INTERVAL seconds.
        if face_detected and model_ready and now - last_analysis >= config.EMOTION_INTERVAL_SECONDS:
            result = detector.analyze(camera.crop_face(frame, face_box))
            if result is not None:
                smoother.add(result["emotion"], result["scores"])
            last_analysis = now

        shown_emotion = smoother.get_smoothed() if face_detected else "No face"
        confidence = smoother.get_confidence() if face_detected else 0.0
        scores = smoother.get_average_scores() if face_detected else {}
        score = engagement_calc.calculate(face_detected, face_box, frame.shape,
                                          smoother.get_consistency_ratio())

        label = f"{shown_emotion} {confidence:.0f}%" if shown_emotion not in ("N/A", "No face") else ""
        camera.publish(frame, face_box, face_count, label)

        with state_lock:
            state.update(camera_available=True, face_detected=face_detected,
                         face_count=face_count, emotion=shown_emotion,
                         confidence=round(confidence, 1), scores=scores,
                         engagement_score=score)
            session_active = state["session_active"]
            session_id = state["session_id"]

        if session_active and shown_emotion != "N/A" \
                and now - last_db_write >= config.DB_WRITE_INTERVAL_SECONDS:
            try:
                db.insert_record(session_id, datetime.now().isoformat(timespec="seconds"),
                                 shown_emotion, confidence, score)
            except Exception as error:  # a DB problem must not stop the camera
                print(f"[DB] Could not save record: {error}")
            last_db_write = now

        time.sleep(0.03)


# ------------------------------- routes --------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def stream():
        while True:
            jpeg = camera.get_jpeg()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.05)
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(dict(state))


@app.route("/api/start_session", methods=["POST"])
def api_start_session():
    with state_lock:
        if state["session_active"]:
            return jsonify({"error": "A session is already running"}), 409
    try:
        session_id = db.create_session(datetime.now().isoformat(timespec="seconds"))
    except Exception as error:
        return jsonify({"error": f"Database error: {error}"}), 500
    smoother.reset()
    update_state(session_active=True, session_id=session_id, session_start_epoch=time.time())
    return jsonify({"status": "started", "session_id": session_id})


@app.route("/api/end_session", methods=["POST"])
def api_end_session():
    with state_lock:
        session_id = state["session_id"]
        was_active = state["session_active"]
    update_state(session_active=False)
    if session_id is None or not was_active:
        return jsonify({"error": "No session is running"}), 409
    db.end_session(session_id, datetime.now().isoformat(timespec="seconds"))
    return jsonify({"session_id": session_id, "summary": build_summary(session_id)})


def pick_session_id():
    """Which session should the dashboard show? ?session_id=... else current, else latest."""
    requested = request.args.get("session_id", type=int)
    if requested is not None:
        return requested
    with state_lock:
        current = state["session_id"]
    return current if current is not None else db.get_latest_session_id()


@app.route("/api/analytics")
def api_analytics():
    session_id = pick_session_id()
    if session_id is None or db.get_session(session_id) is None:
        return jsonify({"session_id": None, "timeline": [], "summary": build_summary(None)})
    records = db.get_records(session_id)
    return jsonify({
        "session_id": session_id,
        "timeline": [{"timestamp": r["timestamp"], "emotion": r["detected_emotion"],
                      "engagement_score": r["engagement_score"]} for r in records],
        "summary": build_summary(session_id),
    })


@app.route("/api/sessions")
def api_sessions():
    with state_lock:
        active_id = state["session_id"] if state["session_active"] else None
    result = []
    for s in db.get_sessions():
        result.append({
            "id": s["id"],
            "start_time": s["start_time"],
            "active": s["id"] == active_id,
            "duration_seconds": session_duration(s),
            "observation_count": s["observation_count"],
            "average_engagement": round(s["average_engagement"] or 0, 1),
        })
    return jsonify(result)


@app.route("/api/export_csv")
def api_export_csv():
    session_id = pick_session_id()
    if session_id is None or db.get_session(session_id) is None:
        return jsonify({"error": "No such session"}), 404
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["timestamp", "detected_emotion", "confidence_percent", "estimated_engagement_score"])
    for r in db.get_records(session_id):
        writer.writerow([r["timestamp"], r["detected_emotion"], round(r["confidence"], 1), r["engagement_score"]])
    return Response(buffer.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=session_{session_id}.csv"})


@app.route("/api/clear_history", methods=["POST"])
def api_clear_history():
    with state_lock:
        if state["session_active"]:
            return jsonify({"error": "End the running session first"}), 409
        state["session_id"] = None
    db.clear_history()
    return jsonify({"status": "cleared"})


def session_duration(session_row):
    """Seconds between start and end (or until now, if the session is still running)."""
    start = datetime.fromisoformat(session_row["start_time"])
    end = datetime.fromisoformat(session_row["end_time"]) if session_row["end_time"] else datetime.now()
    return max(0, int((end - start).total_seconds()))


def build_summary(session_id):
    empty = {"duration_seconds": 0, "observation_count": 0, "average_engagement": 0,
             "face_visible_percent": 0, "emotion_distribution": {}}
    session_row = db.get_session(session_id) if session_id is not None else None
    if session_row is None:
        return empty

    records = db.get_records(session_id)
    total = len(records)
    summary = dict(empty, duration_seconds=session_duration(session_row), observation_count=total)
    if total == 0:
        return summary

    counts = {}
    for r in records:
        counts[r["detected_emotion"]] = counts.get(r["detected_emotion"], 0) + 1
    face_visible = sum(1 for r in records if r["detected_emotion"] != "No face")

    summary["emotion_distribution"] = {name: round(c / total * 100, 1) for name, c in counts.items()}
    summary["average_engagement"] = round(sum(r["engagement_score"] for r in records) / total, 1)
    summary["face_visible_percent"] = round(face_visible / total * 100, 1)
    return summary


if __name__ == "__main__":
    db.init_db()
    threading.Thread(target=processing_loop, daemon=True).start()
    print("\n  Dashboard: http://127.0.0.1:5000\n  (first start downloads the AI model - needs internet once)\n")
    # use_reloader=False: the reloader would start a 2nd process that opens the webcam twice.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
