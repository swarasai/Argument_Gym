"""Spoken debate opponent: Cartesia TTS played on the computer running Streamlit."""
import json
import os
import ssl
import urllib.error
import urllib.request

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

TTS_URL = 'https://api.cartesia.ai/tts/bytes'
TTS_RATE = 24000
TTS_VERSION = '2026-08-14'
DEFAULT_VOICE = '47c38ca4-5f35-497b-b1a3-415245fb35e1'  # Daniel, en-US
DEFAULT_MODEL = 'sonic-3.6'


def _api_key():
    load_dotenv()
    key = os.getenv('CARTESIA_API_KEY')
    if not key:
        raise RuntimeError('Add CARTESIA_API_KEY to your .env file to hear the opponent.')
    return key


def synthesize(text):
    """Return int16 mono PCM for one opponent line."""
    transcript = ' '.join((text or '').split())
    if not transcript:
        raise ValueError('Opponent has nothing to say.')
    key = _api_key()
    payload = {
        'model_id': os.getenv('CARTESIA_TTS_MODEL', DEFAULT_MODEL),
        'transcript': transcript,
        'voice': os.getenv('CARTESIA_VOICE_ID', DEFAULT_VOICE),
        'language': 'en',
        'output_format': {
            'container': 'raw',
            'encoding': 'pcm_s16le',
            'sample_rate': TTS_RATE,
        },
        'generation_config': {'speed': 1.0, 'emotion': 'confident'},
    }
    request = urllib.request.Request(
        TTS_URL,
        data=json.dumps(payload).encode('utf-8'),
        method='POST',
        headers={
            'Authorization': f'Bearer {key}',
            'X-API-Key': key,
            'Cartesia-Version': TTS_VERSION,
            'Content-Type': 'application/json',
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45, context=SSL_CTX) as response:
            pcm = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')[:400]
        raise RuntimeError(f'Opponent voice request failed ({exc.code}): {detail}') from exc
    if len(pcm) < 2:
        raise RuntimeError('Opponent voice returned empty audio.')
    return np.frombuffer(pcm, dtype='<i2')


def speak_opponent(text):
    """Play the opponent line through this computer's speakers and wait until it finishes."""
    audio = synthesize(text)
    sd.play(audio, TTS_RATE)
    sd.wait()
