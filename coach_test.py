"""
Step 3: live speech coaching in the terminal.

Run:  python coach_test.py     (Ctrl+C to stop; saves a session file in ./sessions)
"""

import asyncio
import json
import os
import ssl
import statistics
import time

import sounddevice as sd
import websockets
from dotenv import load_dotenv

from metrics import ACTIVE_BELOW_BASELINE_DB, Coach, chunk_db  # noqa: F401

try:  # fixes the Mac "CERTIFICATE_VERIFY_FAILED" error if certifi is installed
    import certifi

    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = None

load_dotenv()
API_KEY = os.environ["CARTESIA_API_KEY"]

SAMPLE_RATE = 16000
CHUNK_MS = 100
FINALIZE_EVERY = 1.5     # seconds; lower = cues arrive sooner, but words may get split
STATUS_EVERY = 1.5       # seconds between status lines in the terminal

URL = (
    "wss://api.cartesia.ai/stt/websocket"
    f"?model=ink-whisper&encoding=pcm_s16le&sample_rate={SAMPLE_RATE}"
)
HEADERS = {
    "X-API-Key": API_KEY,
    "Authorization": f"Bearer {API_KEY}",
    "Cartesia-Version": "2026-03-01",
}


def calibrate():
    """Record 5 s of normal-volume speech to learn what 'your normal volume' sounds like."""
    input("Calibration: press Enter, then read a sentence aloud at your normal volume (5 sec)... ")
    audio = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    raw = audio.tobytes()
    size = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2  # bytes per chunk
    dbs = [chunk_db(raw[i : i + size]) for i in range(0, len(raw) - size + 1, size)]
    loud = [d for d in dbs if d > max(dbs) - 20]  # chunks within 20 dB of the peak = speech
    if len(loud) < 10:
        print("Didn't hear much, using a default level. Check your mic and try again later.")
        return -25.0
    baseline = statistics.median(loud)
    print(f"Baseline volume: {baseline:.1f} dBFS\n")
    return baseline


async def stream(coach):
    loop = asyncio.get_running_loop()
    audio_q = asyncio.Queue()

    def on_audio(indata, frames, time_info, status):
        loop.call_soon_threadsafe(audio_q.put_nowait, bytes(indata))

    async with websockets.connect(URL, additional_headers=HEADERS, ssl=SSL_CTX) as ws:
        print("Listening... talk normally. Ctrl+C to stop.\n")

        async def send_audio():
            last_finalize = time.monotonic()
            while True:
                chunk = await audio_q.get()
                coach.on_chunk(chunk)
                await ws.send(chunk)
                if time.monotonic() - last_finalize >= FINALIZE_EVERY:
                    await ws.send("finalize")
                    last_finalize = time.monotonic()

        async def receive():
            last_print = 0.0
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(data, dict) or data.get("type") != "transcript":
                    continue
                status = coach.on_transcript(data)
                text = (data.get("text") or "").strip()
                if text:
                    print(f">> {text}")
                if status["cue"]:
                    print(f"   *** {status['cue'].upper()} ***")
                if time.monotonic() - last_print >= STATUS_EVERY:
                    wpm = "--" if status["wpm"] is None else status["wpm"]
                    vol = "--" if status["db_rel"] is None else f"{status['db_rel']:+.0f} dB"
                    print(f"   pace {wpm} wpm [{status['pace_state']}]   volume {vol} [{status['vol_state']}]")
                    last_print = time.monotonic()

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=int(SAMPLE_RATE * CHUNK_MS / 1000),
            callback=on_audio,
        ):
            await asyncio.gather(send_audio(), receive())


if __name__ == "__main__":
    coach = Coach(baseline_db=calibrate())
    try:
        asyncio.run(stream(coach))
    except KeyboardInterrupt:
        pass
    finally:
        print("\nSession summary:")
        print(json.dumps(coach.summary(), indent=2))
        print("Saved to", coach.save())
