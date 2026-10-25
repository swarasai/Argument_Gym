"""Offline tests for topics, report payload and history. Run: python3 test_speak.py"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import history
from report import build_message, pick_numbers
from topics import CATEGORIES, pick_topic
from voice import VoiceTracker, estimate_pitch


class SpeakingTests(unittest.TestCase):
    def test_topics(self):
        for category in CATEGORIES:
            self.assertIn(pick_topic(category), CATEGORIES[category])
            last = CATEGORIES[category][0]
            self.assertNotEqual(pick_topic(category, avoid=last), last)

    def test_report_payload(self):
        summary = {"duration_s": 60, "total_words": 100, "irrelevant": 1}
        self.assertEqual(pick_numbers(summary), {"duration_s": 60, "total_words": 100})
        self.assertIn("My topic", build_message("My topic", "Quick thinking", 60, summary, transcript="hello"))

    def test_history(self):
        with tempfile.TemporaryDirectory() as path:
            with patch.object(history, "ROOT", Path(path)):
                item = {"when": "2026-10-25T10:00:00", "topic": "test", "summary": {"total_words": 20}}
                file = history.save_record("test_user", item)
                self.assertEqual(len(history.load_records("test_user")), 1)
                history.delete_record("test_user", file.name)
                self.assertEqual(history.load_records("test_user"), [])

    def test_short_session_is_saved(self):
        import sys
        # Keep the offline test independent of microphone hardware and API keys.
        with patch.dict(sys.modules, {"sounddevice": MagicMock(), "coach_test": MagicMock()}):
            from speak import make_record
        with tempfile.TemporaryDirectory() as path:
            with patch.object(history, "ROOT", Path(path)):
                class ShortCoach:
                    total_words = 2
                    def summary(self):
                        return {"duration_s": 30, "total_words": 2}
                    def transcript_text(self):
                        return "hello there"
                entry, location = make_record("student", "Sample topic", "Storytelling", 30, ShortCoach())
                self.assertTrue(location.exists())
                self.assertIn("Limited analysis", entry["report"])
                self.assertEqual(len(history.load_records("student")), 1)

    def test_voice_empty(self):
        self.assertIsNone(VoiceTracker().summary()["pitch_variation_semitones"])
        self.assertIsNone(estimate_pitch(bytes(3200)))


if __name__ == "__main__":
    unittest.main()
