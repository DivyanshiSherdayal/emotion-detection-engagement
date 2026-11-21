# EmoSense — AI-Based Emotion Detection and Student Engagement Analysis System

Mini Project (BCC 351) · Group G20 · IMS Engineering College, Ghaziabad
Supervisor: Dr. Ramesh Kumar Verma

A web dashboard that uses the laptop webcam to detect a face, estimate the **facial expression**
with a **pre-trained** model (DeepFace), smooth the result over time, and compute an
**estimated engagement score**. Results are stored in SQLite and shown with Chart.js.

> **Important:** these are AI-based *estimates* of visible facial expressions. They do **not** reveal a
> person's true emotion or real attention, and must not be used for grading or disciplinary decisions.

---

## 1. Setup (once)

```bash
python -m venv venv
venv\Scripts\activate            # Windows      (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt  # takes a few minutes (TensorFlow is large)
```

## 2. Run

```bash
python app.py
```

Open **http://127.0.0.1:5000** in Chrome/Edge.

* **First run needs internet once** — DeepFace downloads its small emotion-model weights.
  The header pill shows "Loading model…" and turns green ("Model ready") when done.
  *Do this at home before the demo, not in the viva hall.*
* Tick the consent box → **Start Session** → talk/react naturally → **End Session** to see the summary.
* Every session is saved. Use **Session History** to **View** an old session's charts/summary or download its readings with **CSV** (opens in Excel). **Clear history** deletes everything.

## 3. How it works (pipeline)

| Step | Where | What happens |
|---|---|---|
| 1. Webcam | `camera/camera.py` | OpenCV reads frames (one background thread owns the camera) |
| 2. Face detection | `camera/camera.py` | OpenCV Haar cascade finds the largest face |
| 3. Expression model | `emotion/detector.py` | DeepFace (pre-trained CNN) → 7 probabilities |
| 4. Temporal smoothing | `emotion/smoothing.py` | Most common label of the last 5 analyses is shown |
| 5. Engagement estimate | `engagement/engagement.py` | Weighted formula below |
| 6. Storage | `database/db.py` | SQLite: table `sessions` + table `session_analytics`; one reading every 2 s during a session (no images) |
| 7. Dashboard | `templates/`, `static/` | Flask serves the page; JS polls `/api/status` and `/api/analytics` |

**Temporal smoothing:** `Happy, Happy, Neutral, Happy, Happy` → displayed `Happy`. Confidence shown is the
model's average probability for the *displayed* expression over the same window.

**Estimated engagement score (0–100):**

```
score = 0.4 × face presence  +  0.3 × orientation  +  0.3 × consistency
```
* *face presence* — 100 if a face is visible, else 0
* *orientation* — how centered the face is horizontally (a rough proxy for "facing the camera"; **not** head-pose tracking)
* *consistency* — how stable the recent expression predictions are

Weights are in `engagement/engagement.py` (`WEIGHTS`) and are a transparent starting point, not a validated
psychological model. Other settings (smoothing window, how often the AI runs, DB write interval) are in `config.py`.

## 4. Folder structure

```
app.py                 Flask app + background processing thread
config.py              Easy-to-change settings
camera/camera.py       Webcam + face detection
emotion/detector.py    DeepFace wrapper (pre-trained model)
emotion/smoothing.py   Temporal smoothing
engagement/engagement.py  Engagement formula
database/db.py         SQLite helper
templates/index.html   Dashboard page
static/css, static/js  Styling and dashboard logic
tests/test_core.py     Automated tests
data/                  database.db is created here
```

## 5. Tests

```bash
python -m unittest discover tests -v     # 14 tests: smoothing, engagement 0–100 bounds, database/sessions
```

Manual checks: cover the camera → "Camera not available" message; step out of frame → "No face detected",
engagement 0, app keeps running; two people in frame → largest face analysed with a warning on the video.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| "Camera not available" | Close Zoom/Teams/other apps using the webcam; allow camera access in OS privacy settings; try `set CAMERA_SOURCE=1` (Windows) / `export CAMERA_SOURCE=1` then run again |
| Model stays on "Loading…" | Needs internet for the first run; check Wi-Fi, then restart `python app.py` |
| `pip install` fails on TensorFlow | Use 64-bit Python 3.10–3.12 (`python --version`) |
| `ImportError: DLL load failed` (Windows) | Install the Microsoft Visual C++ Redistributable |
| Charts are empty / "Chart.js could not be loaded" | No internet to the CDN. Save https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js as `static/js/vendor/chart.umd.min.js` (works offline afterwards) |
| Slow / laggy | Increase `EMOTION_INTERVAL_SECONDS` in `config.py` (e.g. 1.0) |

## 7. Privacy & limitations

* Everything runs locally; **no raw images or video are stored** — only timestamp, expression, confidence, engagement score.
* Use only with the participant's **consent** (the dashboard requires a consent tick).
* Facial expression ≠ emotion ≠ attention. Accuracy varies with lighting, camera angle, glasses, skin tone and culture.
* Prototype optimised for **one person**; multiple faces are detected and counted but only the largest is analysed.
* Not for grading, discipline or any high-stakes decision.

## 8. Possible future work
MediaPipe face-mesh for real head-pose, a stronger Hugging Face expression model, multi-face support,
and exporting session reports as PDF, and comparing sessions side by side.
