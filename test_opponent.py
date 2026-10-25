"""Offline tests: python3 -m unittest test_opponent.py"""
import unittest
from unittest.mock import patch, MagicMock

from opponent import synthesize, speak_opponent


class OpponentTests(unittest.TestCase):
    def test_empty_line_is_rejected(self):
        with self.assertRaises(ValueError):
            synthesize('   ')

    def test_synthesize_sends_pcm_request(self):
        pcm = b'\x01\x00\x02\x00'
        response = MagicMock()
        response.read.return_value = pcm
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        with patch.dict('os.environ', {'CARTESIA_API_KEY': 'test-key'}, clear=False):
            with patch('opponent.urllib.request.urlopen', return_value=response) as opener:
                audio = synthesize('Name one example.')
        self.assertEqual(audio.tolist(), [1, 2])
        request = opener.call_args[0][0]
        self.assertTrue(request.full_url.endswith('/tts/bytes'))
        self.assertEqual(request.get_method(), 'POST')

    def test_speak_waits_for_playback(self):
        with patch('opponent.synthesize', return_value=[1, 2, 3]):
            with patch('opponent.sd.play') as play:
                with patch('opponent.sd.wait') as wait:
                    speak_opponent('Your claim is unsupported.')
        play.assert_called_once()
        wait.assert_called_once()


if __name__ == '__main__':
    unittest.main()
