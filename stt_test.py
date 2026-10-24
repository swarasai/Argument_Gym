"""
Step 1: speak into your Mac mic, see text appear in the terminal.

Uses Cartesia's streaming speech-to-text WebSocket (ink-whisper model).
Run:  python stt_test.py      (Ctrl+C to quit)
"""

import asyncio
import json
import os
import time

import sounddevice as sd
import websockets
from dotenv import load_dotenv

load_dotenv()  # reads CARTESIA_API_KEY from your .env file
API_KEY = os.environ["CARTESIA_API_KEY"]

SAMPLE_RATE = 16000        # Hz, what we send to Cartesia
CHUNK_MS = 100             # send audio in 100 ms pieces
FINALIZE_EVERY = 2.0       # seconds; how often we ask for a transcript
DEBUG = True               # prints every raw message from Cartesia (leave on for the first run)

URL = (
    "wss://api.cartesia.ai/stt/websocket"
    f"?model=ink-whisper&encoding=pcm_s16le&sample_rate={SAMPLE_RATE}"
)
HEADERS = {
    "X-API-Key": API_KEY,
    "Authorization": f"Bearer {API_KEY}",
    "Cartesia-Version": "2026-03-01",
}


async def main():
    loop = asyncio.get_running_loop()
    audio_q: asyncio.Queue[bytes] = asyncio.Queue()

    # The mic runs on its own thread; this hands each chunk to our async code.
    def on_audio(indata, frames, time_info, status):
        if status:
            print("mic status:", status)
        loop.call_soon_threadsafe(audio_q.put_nowait, bytes(indata))

    async with websockets.connect(URL, additional_headers=HEADERS) as ws:
        print("Connected. Start talking... (Ctrl+C to stop)\n")

        async def send_audio():
            last_finalize = time.monotonic()
            while True:
                chunk = await audio_q.get()
                await ws.send(chunk)
                # Cartesia's manual endpoint only returns text when we say "finalize".
                if time.monotonic() - last_finalize >= FINALIZE_EVERY:
                    await ws.send("finalize")
                    last_finalize = time.monotonic()

        async def receive_text():
            async for msg in ws:
                if DEBUG:
                    print("RAW:", msg)
                try:
                    data = json.loads(msg)
                except (TypeError, json.JSONDecodeError):
                    continue
                text = data.get("text") if isinstance(data, dict) else None
                if text and text.strip():
                    print(f">> {text.strip()}")

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=int(SAMPLE_RATE * CHUNK_MS / 1000),
            callback=on_audio,
        ):
            await asyncio.gather(send_audio(), receive_text())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")