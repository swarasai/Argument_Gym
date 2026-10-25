"""
Read-aloud practice: show a sentence, the user reads it, and we report how many of its
words the speech recognizer picked up.

IMPORTANT wording: this measures what Cartesia's recognizer understood. It is NOT a
clinical intelligibility score, and it should never be presented as one.

Run:  python3 practice.py      (Ctrl+C to stop)
Saves only the target sentence and the score (not what was heard) to practice/practice_log.json
"""

import asyncio
import json
import os
import random
import time
from datetime import datetime

import sounddevice as sd
import websockets

from coach_test import CHUNK_MS, HEADERS, SAMPLE_RATE, SSL_CTX, URL
from metrics import match_score

TARGETS = [
    "The weather is nice today",
    "Could you please say that again more slowly",
    "I would like to make an appointment for Tuesday",
    "Thank you for taking the time to talk with me",
    "Where is the nearest pharmacy",
    "I am going to the park with my friends",
]

FINALIZE_EVERY = 1.0    # seconds; how often we ask Cartesia for text
END_SILENCE_S = 3.0     # stop listening after this long with no new words
MAX_SECONDS = 25.0      # never listen longer than this
LOG_PATH = os.path.join("practice", "practice_log.json")


async def listen_once():
    """Listen until the user finishes reading. Returns the text the recognizer heard."""
    loop = asyncio.get_running_loop()
    audio_q = asyncio.Queue()
    heard = []
    state = {"last_text_at": None}

    def on_audio(indata, frames, time_info, status):
        loop.call_soon_threadsafe(audio_q.put_nowait, bytes(indata))

    async with websockets.connect(URL, additional_headers=HEADERS, ssl=SSL_CTX) as ws:

        async def send_audio():
            last_finalize = time.monotonic()
            while True:
                chunk = await audio_q.get()
                await ws.send(chunk)
                if time.monotonic() - last_finalize >= FINALIZE_EVERY:
                    await ws.send("finalize")
                    last_finalize = time.monotonic()

        async def receive():
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(data, dict) or data.get("type") != "transcript":
                    continue
                text = (data.get("text") or "").strip()
                if text:
                    heard.append(text)
                    state["last_text_at"] = time.monotonic()

        started = time.monotonic()
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=int(SAMPLE_RATE * CHUNK_MS / 1000),
            callback=on_audio,
        ):
            tasks = [asyncio.create_task(send_audio()), asyncio.create_task(receive())]
            try:
                while True:
                    await asyncio.sleep(0.2)
                    now = time.monotonic()
                    last = state["last_text_at"]
                    if now - started > MAX_SECONDS:
                        break
                    if last is not None and now - last >= END_SILENCE_S:
                        break
                    if any(t.done() for t in tasks):  # connection dropped or a task crashed
                        break
            finally:
                for t in tasks:
                    t.cancel()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        print("Connection problem:", r)
    return " ".join(heard)


def log_result(target, pct):
    os.makedirs("practice", exist_ok=True)
    log = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            log = json.load(f)
    log.append({"when": datetime.now().isoformat(timespec="seconds"), "target": target, "match_pct": pct})
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def main():
    print("Read-aloud practice. Read each sentence at a comfortable pace.")
    print("The score shows how many words the speech recognizer picked up.")
    print("It is not a clinical measure.\n")
    while True:
        target = random.choice(TARGETS)
        print(f'Read this aloud:\n\n    "{target}"\n')
        input("Press Enter, then start reading... ")
        print("Listening...")
        heard = asyncio.run(listen_once())
        if not heard.strip():
            print("\nDidn't hear anything. Check your mic and try again.")
        else:
            result = match_score(target, heard)
            shown = " ".join(w if ok else f"[{w}]" for w, ok in zip(result["target_words"], result["matched"]))
            got = sum(result["matched"])
            print(f"\nRecognizer heard: {heard}")
            print(f"Target:           {shown}   ([word] = not picked up)")
            print(f"Picked up {got} of {len(result['matched'])} words ({result['pct']}%)")
            log_result(target, result["pct"])
        if input("\nEnter = another sentence, q = quit: ").strip().lower() == "q":
            break
    print("Saved to", LOG_PATH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")