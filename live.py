"""
Live test: mic -> Cartesia -> collect fragments until the user pauses -> interpret().

Fixes the problem where one sentence came back as several short fragments.
Fragments are joined into one transcript, and only after the user has been quiet
for PAUSE_SECONDS does the whole thing go to the LLM.

Run:  python3 live.py      (Ctrl+C to quit)
Needs: interpret.py in the same folder, CARTESIA_API_KEY and ANTHROPIC_API_KEY in .env
"""

import asyncio
import json
import os
import time

import sounddevice as sd
import websockets
from dotenv import load_dotenv

from interpret import interpret

load_dotenv()
API_KEY = os.environ["CARTESIA_API_KEY"]

SAMPLE_RATE = 16000
CHUNK_MS = 100
FINALIZE_EVERY = 1.0     # seconds; how often we ask Cartesia for text
PAUSE_SECONDS = 2.5      # quiet this long = the user finished the sentence (tune this!)
DEBUG = False

SITUATION = "ordering coffee at a cafe"
PHRASES = [
    "Can I get a large oat milk latte",
    "Thank you very much",
    "How much is that",
    "Where is the bathroom",
]

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

    fragments = []             # text pieces heard since the last sentence
    state = {"last_text": 0.0}  # when the most recent text arrived
    confirmed = []             # accepted sentences (used once the confirm flow exists)

    def on_audio(indata, frames, time_info, status):
        if status:
            print("mic status:", status)
        loop.call_soon_threadsafe(audio_q.put_nowait, bytes(indata))

    async with websockets.connect(URL, additional_headers=HEADERS) as ws:
        print("Connected. Talk, then pause. (Ctrl+C to stop)\n")

        async def send_audio():
            last_finalize = time.monotonic()
            while True:
                chunk = await audio_q.get()
                await ws.send(chunk)
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
                    fragments.append(text.strip())
                    state["last_text"] = time.monotonic()
                    # live caption: everything heard so far this sentence
                    print("  ...", " ".join(fragments))

        async def watch_for_pause():
            while True:
                await asyncio.sleep(0.2)
                quiet_for = time.monotonic() - state["last_text"]
                if fragments and quiet_for >= PAUSE_SECONDS:
                    raw = " ".join(fragments)
                    fragments.clear()
                    print(f"\nHEARD:    {raw}")
                    result = await asyncio.to_thread(
                        interpret, raw, SITUATION, PHRASES, confirmed
                    )
                    if not result["best"]:
                        print("GUESS:    (couldn't tell, would ask the user to repeat)\n")
                    else:
                        print(f'GUESS:    {result["best"]}   ({result["confidence"]})')
                        print(f'ALTS:     {result["alternates"]}\n')

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=int(SAMPLE_RATE * CHUNK_MS / 1000),
            callback=on_audio,
        ):
            await asyncio.gather(send_audio(), receive_text(), watch_for_pause())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")