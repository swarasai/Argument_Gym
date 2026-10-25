"""Timed speaking practice. Run: python3 speak.py [--user maya] [--live] [--save-transcript]."""
import argparse
import asyncio
import json
import time
from datetime import datetime

import sounddevice as sd
import websockets

from coach_test import CHUNK_MS, HEADERS, SAMPLE_RATE, SSL_CTX, URL, calibrate
from history import load_last, save_record
from metrics import Coach
from report import write_report
from topics import CATEGORIES, pick_topic

FINALIZE_EVERY = 1.0
GRACE_S = 2.5
MIN_WORDS = 15
DURATIONS = [30, 60, 90, 120]


def choose(prompt, options):
    for i, item in enumerate(options, 1):
        print(f"  {i}. {item}")
    while True:
        reply = input(f"{prompt} (1-{len(options)}): ").strip()
        if reply.isdecimal() and 1 <= int(reply) <= len(options):
            return int(reply) - 1
        print("Choose one of the numbered options.")


async def record(coach, seconds, live=False, on_tick=None):
    """Keep original Cartesia stream protocol and Coach event interfaces."""
    loop = asyncio.get_running_loop()
    audio_q = asyncio.Queue(maxsize=80)

    def on_audio(indata, frames, time_info, status):
        if status:
            print("microphone status:", status)
        data = bytes(indata)
        def add():
            if audio_q.full():
                try:
                    audio_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            audio_q.put_nowait(data)
        loop.call_soon_threadsafe(add)

    async with websockets.connect(URL, additional_headers=HEADERS, ssl=SSL_CTX) as ws:
        async def send_audio():
            last_finalize = time.monotonic()
            while True:
                pcm = await audio_q.get()
                coach.on_chunk(pcm)
                await ws.send(pcm)
                if time.monotonic() - last_finalize >= FINALIZE_EVERY:
                    await ws.send("finalize")
                    last_finalize = time.monotonic()

        async def receive():
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if not isinstance(data, dict):
                    continue
                if data.get("type") == "error":
                    raise RuntimeError(str(data.get("message", data)))
                if data.get("type") != "transcript":
                    continue
                state = coach.on_transcript(data)
                if live:
                    text = (data.get("text") or "").strip()
                    if text:
                        print(">>", text)
                    if state["cue"]:
                        print("COACH:", state["cue"])

        with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                               blocksize=int(SAMPLE_RATE * CHUNK_MS / 1000), callback=on_audio):
            sender = asyncio.create_task(send_audio())
            receiver = asyncio.create_task(receive())
            print("\nStart speaking now.\n")
            started = time.monotonic()
            elapsed = 0.0
            if on_tick:
                on_tick(seconds, seconds)
            try:
                while True:
                    elapsed = time.monotonic() - started
                    if on_tick:
                        on_tick(max(0.0, seconds - elapsed), seconds)
                    if elapsed >= seconds:
                        break
                    if sender.done() or receiver.done():
                        for task in (sender, receiver):
                            if task.done():
                                task.result()  # surface errors rather than hiding them
                        raise RuntimeError("Audio streaming ended unexpectedly")
                    await asyncio.sleep(0.2)
            finally:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
                if not receiver.done():
                    try:
                        await ws.send("finalize")
                        await asyncio.sleep(GRACE_S)
                    except websockets.exceptions.ConnectionClosed:
                        pass
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
    coach.stream_time = max(coach.stream_time, float(min(elapsed, seconds)))
    if on_tick:
        on_tick(0, seconds)


def make_record(user, topic, category, seconds, coach, save_transcript=False):
    """Analyze, compare with history, obtain report and save one JSON session."""
    summary = coach.summary()
    last = load_last(user)
    previous = last.get("summary") if last else None
    transcript = coach.transcript_text()
    if coach.total_words < MIN_WORDS:
        feedback = (f"**Limited analysis:** Only {coach.total_words} words were recognized. "
                    f"At least {MIN_WORDS} recognized words are recommended for useful coherence feedback. "
                    "This practice attempt was saved, but content and delivery scores may be unreliable. "
                    "Try speaking longer and check the microphone if words are missing.")
    else:
        feedback = write_report(topic, category, seconds, summary, previous, transcript)
    entry = {"when": datetime.now().isoformat(timespec="seconds"), "category": category,
             "topic": topic, "target_s": seconds, "summary": summary, "report": feedback}
    if save_transcript:
        entry["transcript"] = transcript
    path = save_record(user, entry)
    return entry, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="me")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--save-transcript", action="store_true")
    args = parser.parse_args()
    print("Speaking coach — the transcript is sent to Claude to provide feedback.")
    print("Transcript text is saved locally only with --save-transcript.\n")
    category = list(CATEGORIES)[choose("Category", list(CATEGORIES))]
    topic = pick_topic(category)
    while True:
        print("\nTopic:", topic)
        if input("Enter to accept, n for another: ").lower().strip() != "n":
            break
        topic = pick_topic(category, avoid=topic)
    seconds = DURATIONS[choose("Length", [f"{x} seconds" for x in DURATIONS])]
    baseline = calibrate()
    coach = Coach(baseline_db=baseline, keep_text=True, track_voice=True)
    input("Press Enter to start...")
    asyncio.run(record(coach, seconds, args.live))
    print("\nAnalyzing...")
    entry, path = make_record(args.user, topic, category, seconds, coach, args.save_transcript)
    print("\n", entry["report"], "\n\nSaved:", path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
