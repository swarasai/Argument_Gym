"""
Speech coaching logic: pace + volume tracking, cues with cooldown, session summary.
No microphone or network code in here, so a dashboard or the Raspberry Pi can reuse it.
"""

import json
import math
import os
import statistics
import time
from collections import deque
from datetime import datetime

# ---- Tune these ------------------------------------------------------------
FAST_WPM = 160            # above this, burst speaking rate counts as "too fast"
PACE_WINDOW_S = 10.0      # look at the last 10 seconds of speech
MIN_WORDS_FOR_PACE = 5    # need at least this many words in the window to judge pace
QUIET_BELOW_BASELINE_DB = 6.0   # quieter than (your calibrated level - 6 dB) = "too quiet"
ACTIVE_BELOW_BASELINE_DB = 15.0 # chunks quieter than (baseline - 15 dB) count as silence
COOLDOWN_S = 10.0         # minimum seconds between cues, so it doesn't nag
# ----------------------------------------------------------------------------


def chunk_db(pcm_bytes):
    """Loudness of one 16-bit mono audio chunk, in dB relative to full scale (0 = max)."""
    import numpy as np

    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return -100.0
    rms = math.sqrt(float(np.mean(x * x)))
    return 20 * math.log10(max(rms, 1.0) / 32768)


class Coach:
    def __init__(self, baseline_db):
        self.baseline_db = baseline_db
        self.stream_time = 0.0          # seconds of audio Cartesia has processed so far
        self.words = deque()            # (start, end) of recent words, in stream time
        self.total_words = 0
        self.recent_db = deque(maxlen=30)   # last ~3 seconds of chunk loudness
        self.last_cue_at = -1e9
        self.cues_given = {"slow_down": 0, "louder": 0}
        self.samples = []               # timeline for the daily report

    # ---- feed it data ------------------------------------------------------
    def on_chunk(self, pcm_bytes):
        self.recent_db.append(chunk_db(pcm_bytes))

    def on_transcript(self, msg):
        """msg = one parsed Cartesia message with type == 'transcript'."""
        # Every transcript (even empty ones = silence) reports how much audio it covered.
        self.stream_time += float(msg.get("duration", 0) or 0)
        for w in msg.get("words", []) or []:
            self.words.append((w["start"], w["end"]))
            self.total_words += 1
            self.stream_time = max(self.stream_time, w["end"])
        cutoff = self.stream_time - PACE_WINDOW_S
        while self.words and self.words[0][1] < cutoff:
            self.words.popleft()

        status = self.status()
        self.samples.append(
            {"t": round(self.stream_time, 1), "wpm": status["wpm"], "db_rel": status["db_rel"]}
        )
        return status

    # ---- read the current state -------------------------------------------
    def pace_wpm(self):
        """Speaking rate over the recent burst of speech, or None if not enough words."""
        if len(self.words) < MIN_WORDS_FOR_PACE:
            return None
        span = self.words[-1][1] - self.words[0][0]
        if span <= 0.5:
            return None
        return len(self.words) / span * 60

    def volume_db(self):
        """Average loudness of the last ~3 s of actual speech, or None if silent."""
        active = [d for d in self.recent_db if d > self.baseline_db - ACTIVE_BELOW_BASELINE_DB]
        if len(active) < 5:
            return None
        return sum(active) / len(active)

    def status(self):
        wpm = self.pace_wpm()
        db = self.volume_db()
        db_rel = None if db is None else db - self.baseline_db
        pace_state = "unknown" if wpm is None else ("fast" if wpm > FAST_WPM else "ok")
        vol_state = (
            "unknown" if db_rel is None else ("quiet" if db_rel < -QUIET_BELOW_BASELINE_DB else "ok")
        )

        cue = None
        now = time.monotonic()
        if now - self.last_cue_at >= COOLDOWN_S:
            if pace_state == "fast":
                cue = "Slow down a little"
                self.cues_given["slow_down"] += 1
            elif vol_state == "quiet":
                cue = "A bit louder"
                self.cues_given["louder"] += 1
            if cue:
                self.last_cue_at = now

        return {
            "wpm": None if wpm is None else round(wpm),
            "db_rel": None if db_rel is None else round(db_rel, 1),
            "pace_state": pace_state,
            "vol_state": vol_state,
            "cue": cue,
        }

    # ---- end of session ----------------------------------------------------
    def summary(self):
        wpms = [s["wpm"] for s in self.samples if s["wpm"] is not None]
        dbs = [s["db_rel"] for s in self.samples if s["db_rel"] is not None]
        return {
            "duration_s": round(self.stream_time),
            "total_words": self.total_words,
            "avg_wpm_when_speaking": round(statistics.mean(wpms)) if wpms else None,
            "pct_time_too_fast": round(100 * sum(w > FAST_WPM for w in wpms) / len(wpms)) if wpms else None,
            "avg_volume_vs_baseline_db": round(statistics.mean(dbs), 1) if dbs else None,
            "pct_time_too_quiet": round(100 * sum(d < -QUIET_BELOW_BASELINE_DB for d in dbs) / len(dbs)) if dbs else None,
            "cues_given": self.cues_given,
            "fast_wpm_threshold": FAST_WPM,
        }

    def save(self, folder="sessions"):
        """Saves numbers only (no audio, no transcript text) to a JSON file."""
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".json")
        with open(path, "w") as f:
            json.dump({"summary": self.summary(), "timeline": self.samples}, f, indent=2)
        return path
