"""Offline tests: python3 -m unittest test_debate.py"""
import unittest
from unittest.mock import patch
from debate import grade_answer, next_challenge, compact_summary


class DebateTests(unittest.TestCase):
    def test_short_audio_is_not_hallucinated(self):
        with patch('debate._call', side_effect=AssertionError('should not call AI')):
            self.assertTrue(grade_answer('x', 'x', 'hello', 1)['limited'])

    def test_score_validation(self):
        with patch('debate._call', return_value={'scores': {'claim': 4, 'reasoning': 8,
                              'evidence': True, 'organization': 3}}):
            self.assertEqual(grade_answer('x', 'x', 'one two three four five', 1)['scores'],
                             {'claim': 4, 'organization': 3})

    def test_repeat_challenge_fallback(self):
        with patch('debate._call', return_value={'question': 'already asked', 'focus': 'evidence'}):
            self.assertNotEqual(next_challenge('x', 'x', 'text', {'scores': {}}, ['already asked'])['question'], 'already asked')

    def test_summary_of_rounds(self):
        turns = [{'summary': {'duration_s': 30, 'total_words': 60, 'fillers_per_min': 2,
                              'pace_excluding_pauses_wpm': 120, 'long_pauses': 1}},
                 {'summary': {'duration_s': 30, 'total_words': 50, 'fillers_per_min': 4,
                              'pace_excluding_pauses_wpm': 100, 'long_pauses': 2}}]
        s = compact_summary(turns)
        self.assertEqual((s['total_words'], s['long_pauses'], s['fillers_per_min']), (110, 3, 3))


if __name__ == '__main__':
    unittest.main()
