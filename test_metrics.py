"""Offline tests for the existing metrics interface. Run: python3 test_metrics.py"""
import unittest
from metrics import Coach, analyze_words, match_score


class MetricsTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(analyze_words([], 10)["filler_words"], 0)

    def test_fillers(self):
        words = [(w, i * 0.5, i * 0.5 + 0.35) for i, w in enumerate("um I basically like it".split())]
        result = analyze_words(words, 3)
        self.assertEqual(result["filler_words"], 1)
        self.assertGreaterEqual(result["possible_filler_words"], 1)

    def test_match(self):
        self.assertEqual(match_score("I like chocolate", "i LIKE chocolate!")["pct"], 100)

    def test_session(self):
        coach = Coach(-25.0, keep_text=True, track_voice=True)
        coach.on_transcript({"type": "transcript", "duration": 2, "text": "Hello friends",
            "words": [{"word": "Hello", "start": 0, "end": 0.5},
                      {"word": "friends", "start": 0.5, "end": 1.0}]})
        self.assertEqual(coach.total_words, 2)
        self.assertIn("pitch_variation_semitones", coach.summary())
        self.assertEqual(coach.transcript_text(), "Hello friends")


if __name__ == "__main__":
    unittest.main()
