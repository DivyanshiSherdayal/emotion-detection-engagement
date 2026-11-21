"""
Basic automated tests (no webcam or AI model needed).
Run from the project folder:   python -m unittest discover tests -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import db
from emotion.smoothing import TemporalSmoother
from engagement.engagement import EngagementCalculator


class SmootherTests(unittest.TestCase):
    def test_single_odd_frame_does_not_change_result(self):
        smoother = TemporalSmoother(window_size=5)
        for label in ["Happy", "Happy", "Neutral", "Happy", "Happy"]:
            smoother.add(label)
        self.assertEqual(smoother.get_smoothed(), "Happy")
        self.assertAlmostEqual(smoother.get_consistency_ratio(), 0.8)

    def test_old_predictions_drop_out_of_the_window(self):
        smoother = TemporalSmoother(window_size=3)
        for label in ["Sad", "Sad", "Sad", "Happy", "Happy", "Happy"]:
            smoother.add(label)
        self.assertEqual(smoother.get_smoothed(), "Happy")

    def test_empty_smoother_is_safe(self):
        smoother = TemporalSmoother()
        self.assertEqual(smoother.get_smoothed(), "N/A")
        self.assertEqual(smoother.get_consistency_ratio(), 0.0)

    def test_confidence_matches_displayed_emotion(self):
        smoother = TemporalSmoother(window_size=2)
        smoother.add("Happy", {"Happy": 80.0, "Sad": 20.0})
        smoother.add("Happy", {"Happy": 60.0, "Sad": 40.0})
        self.assertEqual(smoother.get_confidence(), 70.0)


class EngagementTests(unittest.TestCase):
    shape = (480, 640, 3)

    def test_no_face_scores_zero(self):
        self.assertEqual(EngagementCalculator().calculate(False, None, self.shape, 1.0), 0)

    def test_centered_stable_face_scores_100(self):
        box = (220, 100, 200, 200)  # centered horizontally in a 640-wide frame
        self.assertEqual(EngagementCalculator().calculate(True, box, self.shape, 1.0), 100)

    def test_score_always_between_0_and_100(self):
        calc = EngagementCalculator()
        for x in (0, 100, 300, 600):
            for consistency in (0.0, 0.5, 1.0, 5.0, -2.0):
                score = calc.calculate(True, (x, 50, 100, 100), self.shape, consistency)
                self.assertTrue(0 <= score <= 100)

    def test_off_center_face_scores_lower(self):
        calc = EngagementCalculator()
        centered = calc.calculate(True, (220, 100, 200, 200), self.shape, 1.0)
        edge = calc.calculate(True, (0, 100, 200, 200), self.shape, 1.0)
        self.assertLess(edge, centered)


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.original_path = db.DB_PATH
        db.DB_PATH = os.path.join(self.folder, "test.db")
        db.init_db()

    def tearDown(self):
        db.DB_PATH = self.original_path

    def test_session_with_readings(self):
        session_id = db.create_session("2026-11-20T10:00:00")
        db.insert_record(session_id, "2026-11-20T10:00:02", "Happy", 82.5, 90)
        db.insert_record(session_id, "2026-11-20T10:00:04", "Neutral", 60.0, 70)
        records = db.get_records(session_id)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["detected_emotion"], "Happy")

    def test_sessions_are_kept_separately(self):
        first = db.create_session("2026-11-20T10:00:00")
        db.insert_record(first, "2026-11-20T10:00:02", "Happy", 80, 90)
        second = db.create_session("2026-11-20T11:00:00")
        db.insert_record(second, "2026-11-20T11:00:02", "Sad", 55, 60)
        db.insert_record(second, "2026-11-20T11:00:04", "Sad", 57, 80)
        self.assertEqual(len(db.get_records(first)), 1)
        self.assertEqual(len(db.get_records(second)), 2)
        listing = db.get_sessions()          # newest first
        self.assertEqual([s["id"] for s in listing], [second, first])
        self.assertEqual(listing[0]["observation_count"], 2)
        self.assertAlmostEqual(listing[0]["average_engagement"], 70.0)

    def test_end_session_and_latest(self):
        session_id = db.create_session("2026-11-20T10:00:00")
        db.end_session(session_id, "2026-11-20T10:05:00")
        self.assertEqual(db.get_session(session_id)["end_time"], "2026-11-20T10:05:00")
        self.assertEqual(db.get_latest_session_id(), session_id)

    def test_unfinished_session_gets_an_end_time(self):
        session_id = db.create_session("2026-11-20T10:00:00")
        db.insert_record(session_id, "2026-11-20T10:00:06", "Happy", 80, 90)
        db.close_unfinished_sessions()
        self.assertEqual(db.get_session(session_id)["end_time"], "2026-11-20T10:00:06")

    def test_clear_history(self):
        session_id = db.create_session("2026-11-20T10:00:00")
        db.insert_record(session_id, "2026-11-20T10:00:02", "Happy", 80, 90)
        db.clear_history()
        self.assertEqual(db.get_sessions(), [])
        self.assertEqual(db.get_records(session_id), [])

    def test_old_table_without_session_id_is_upgraded(self):
        import sqlite3
        db.DB_PATH = os.path.join(self.folder, "old.db")
        conn = sqlite3.connect(db.DB_PATH)
        conn.execute("CREATE TABLE session_analytics (id INTEGER PRIMARY KEY, timestamp TEXT, "
                     "detected_emotion TEXT, confidence REAL, engagement_score REAL)")
        conn.commit(); conn.close()
        db.init_db()                         # must not crash and must add session_id
        session_id = db.create_session("2026-11-20T10:00:00")
        db.insert_record(session_id, "2026-11-20T10:00:02", "Happy", 80, 90)
        self.assertEqual(len(db.get_records(session_id)), 1)


if __name__ == "__main__":
    unittest.main()
