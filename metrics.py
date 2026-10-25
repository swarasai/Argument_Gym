"""
Speech coaching logic: pace + volume tracking, pauses, repeats, cues with cooldown,
session summary. No microphone or network code in here, so a dashboard or the
Raspberry Pi can reuse it.
"""
 
import difflib
import json
import math
import os
import re
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
 
LONG_PAUSE_S = 1.0        # a gap between words this long counts as a long pause ("block")
STOPPED_TALKING_S = 6.0   # a gap this long means the person stopped talking, not a block
FILLERS = {"um", "uh", "uhm", "umm", "er", "erm", "hmm", "mm"}
SENTENCE_END = (".", "?", "!")   # a long gap after these is a normal sentence break
 
# Words that are OFTEN (not always) filler. Counted separately from true fillers because the
# speech recognizer usually removes um/uh but keeps these, and some uses are legitimate.
POSSIBLE_FILLER_WORDS = {"basically", "literally", "actually"}
POSSIBLE_FILLER_PHRASES = {("you", "know"), ("i", "mean"), ("kind", "of"), ("sort", "of")}
# "like" is only counted when it does NOT follow one of these ("I like pizza", "feels like rain").
LIKE_NOT_FILLER_AFTER = {
    "i", "you", "we", "they", "to", "would", "don't", "dont", "didn't", "didnt",
    "doesn't", "who", "might", "could", "should", "look", "looks", "feel", "feels",
    "sound", "sounds", "seem", "seems", "just",
}
# ----------------------------------------------------------------------------
 
 
def chunk_db(pcm_bytes):
    """Loudness of one 16-bit mono audio chunk, in dB relative to full scale (0 = max)."""
    import numpy as np
 
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return -100.0
    rms = math.sqrt(float(np.mean(x * x)))
    return 20 * math.log10(max(rms, 1.0) / 32768)
 
 
# ---- Word-level analysis (pure functions, easy to test) ----------------------
def _clean(word):
    """Lowercase and strip punctuation/dashes at the edges: "I--" -> "i", "card." -> "card"."""
    return re.sub(r"^[^\w']+|[^\w']+$", "", word.lower())
 
 
def _is_stutter_token(word):
    """True for hyphenated repeats such as "b-b-ball" or "I-I-I". Rough heuristic."""
    parts = [p for p in word.lower().strip(".,?!;:").split("-") if p]
    return len(parts) >= 2 and parts[0] == parts[1]
 
 
def analyze_words(words, total_time):
    """
    words      -- list of (text, start, end) in order; times are seconds of stream time
    total_time -- seconds of audio covered by the session
    Returns numbers only (nothing from the transcript is kept).
 
    ROUGH ESTIMATES: the speech-to-text often removes repeats and fillers, so
    repeated_words and filler_words are likely to undercount.
    """
    if not words:
        return {
            "long_pauses": 0,
            "longest_pause_s": 0.0,
            "speaking_time_s": 0.0,
            "silence_time_s": round(max(total_time, 0.0), 1),
            "speech_share_pct": None,
            "pace_excluding_pauses_wpm": None,
            "repeated_words": 0,
            "filler_words": 0,
            "possible_filler_words": 0,
        }
 
    # Split the speech into "runs" separated by long pauses.
    runs = []                  # (start, end) of each stretch of continuous speech
    blocks = []                # long pauses in the middle of speaking, in seconds
    run_start = words[0][1]
    prev_text, prev_end = words[0][0], words[0][2]
    for text, start, end in words[1:]:
        gap = start - prev_end
        if gap >= LONG_PAUSE_S:
            runs.append((run_start, prev_end))
            run_start = start
            normal_sentence_break = prev_text.rstrip().endswith(SENTENCE_END)
            if gap < STOPPED_TALKING_S and not normal_sentence_break:
                blocks.append(gap)
        prev_text, prev_end = text, end
    runs.append((run_start, prev_end))
 
    speaking_time = sum(e - s for s, e in runs)
    silence_time = max(total_time - speaking_time, 0.0)
    share = min(100, round(100 * speaking_time / total_time)) if total_time > 0 else None
    pace = round(len(words) / speaking_time * 60) if speaking_time >= 2.0 else None
 
    # Repeats and fillers, counted from the words as the recognizer wrote them.
    cleaned = [_clean(t) for t, _, _ in words]
    fillers = sum(c in FILLERS for c in cleaned)
    repeats = 0
    for i, (text, _, _) in enumerate(words):
        if _is_stutter_token(text):
            repeats += 1
        elif i > 0 and cleaned[i] and cleaned[i] == cleaned[i - 1] and cleaned[i] not in FILLERS:
            repeats += 1
 
    possible = 0
    for i, c in enumerate(cleaned):
        if c == "like":
            if i == 0 or cleaned[i - 1] not in LIKE_NOT_FILLER_AFTER:
                possible += 1
        elif c in POSSIBLE_FILLER_WORDS:
            possible += 1
        if i > 0 and (cleaned[i - 1], c) in POSSIBLE_FILLER_PHRASES:
            possible += 1
 
    return {
        "long_pauses": len(blocks),
        "longest_pause_s": round(max(blocks), 1) if blocks else 0.0,
        "speaking_time_s": round(speaking_time, 1),
        "silence_time_s": round(silence_time, 1),
        "speech_share_pct": share,
        "pace_excluding_pauses_wpm": pace,
        "repeated_words": repeats,
        "filler_words": fillers,
        "possible_filler_words": possible,
    }
 
 
def match_score(target, heard):
    """
    How many words of the target sentence the speech recognizer picked up, in order.
    This measures what the recognizer understood, NOT clinical intelligibility.
 
    Returns {"pct": int, "matched": [bool per target word], "target_words": [str]}
    """
    t = [w for w in (_clean(x) for x in target.split()) if w]
    h = [w for w in (_clean(x) for x in heard.split()) if w]
    matched = [False] * len(t)
    if t and h:
        sm = difflib.SequenceMatcher(None, t, h, autojunk=False)
        for a, _, size in sm.get_matching_blocks():
            for i in range(a, a + size):
                matched[i] = True
    pct = round(100 * sum(matched) / len(t)) if t else 0
    return {"pct": pct, "matched": matched, "target_words": t}
 
 
class Coach:
    def __init__(self, baseline_db, keep_text=False, track_voice=False):
        self.baseline_db = baseline_db
        self.voice = None
        if track_voice:  # pitch and loudness variety; needs numpy
            from voice import VoiceTracker
 
            self.voice = VoiceTracker()
        self.keep_text = keep_text      # keep transcript text in memory (never saved to disk)
        self.stream_time = 0.0          # seconds of audio Cartesia has processed so far
        self.words = deque()            # (start, end) of recent words, in stream time
        self.all_words = []             # (text, start, end) for the whole session, in memory only
        self.texts = []                 # transcript pieces, only filled when keep_text=True
        self.total_words = 0
        self.recent_db = deque(maxlen=30)   # last ~3 seconds of chunk loudness
        self.last_cue_at = -1e9
        self.cues_given = {"slow_down": 0, "louder": 0}
        self.samples = []               # timeline for the daily report
        self.extra = {}                 # extra fields merged into the summary (e.g. coherence)
 
    # ---- feed it data ------------------------------------------------------
    def on_chunk(self, pcm_bytes):
        db = chunk_db(pcm_bytes)
        self.recent_db.append(db)
        if self.voice:
            self.voice.on_chunk(pcm_bytes, db, self.baseline_db - ACTIVE_BELOW_BASELINE_DB)
 
    def on_transcript(self, msg):
        """msg = one parsed Cartesia message with type == 'transcript'."""
        # Every transcript (even empty ones = silence) reports how much audio it covered.
        self.stream_time += float(msg.get("duration", 0) or 0)
        text = (msg.get("text") or "").strip()
        if self.keep_text and text:
            self.texts.append(text)
        for w in msg.get("words", []) or []:
            self.words.append((w["start"], w["end"]))
            self.all_words.append((w.get("word", ""), w["start"], w["end"]))
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
 
    def transcript_text(self):
        """Whole session transcript, only available when keep_text=True."""
        return " ".join(self.texts)
 
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
        out = {
            "duration_s": round(self.stream_time),
            "total_words": self.total_words,
            "avg_wpm_when_speaking": round(statistics.mean(wpms)) if wpms else None,
            "pct_time_too_fast": round(100 * sum(w > FAST_WPM for w in wpms) / len(wpms)) if wpms else None,
            "avg_volume_vs_baseline_db": round(statistics.mean(dbs), 1) if dbs else None,
            "pct_time_too_quiet": round(100 * sum(d < -QUIET_BELOW_BASELINE_DB for d in dbs) / len(dbs)) if dbs else None,
            "pace_variation_wpm": round(statistics.pstdev(wpms)) if len(wpms) >= 3 else None,
            "cues_given": self.cues_given,
            "fast_wpm_threshold": FAST_WPM,
        }
        word_stats = analyze_words(self.all_words, self.stream_time)
        out.update(word_stats)
        minutes = self.stream_time / 60
        out["fillers_per_min"] = (
            round((word_stats["filler_words"] + word_stats["possible_filler_words"]) / minutes, 1)
            if minutes >= 0.25
            else None
        )
        if self.voice:
            out.update(self.voice.summary())
        out.update(self.extra)
        out["notes"] = (
            "filler_words (um/uh) and repeated_words are rough estimates and likely undercount, "
            "because speech-to-text often removes them. possible_filler_words counts words like "
            "'like', 'basically' and 'you know', which can be legitimate uses. "
            "Pitch/loudness variety comes from the audio signal and is a rough signal. "
            "pct_time_* values count transcript messages (about 1.5 s each), not exact seconds."
        )
        return out
 
    def save(self, folder="sessions"):
        """Saves numbers only (no audio, no transcript text) to a JSON file."""
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".json")
        with open(path, "w") as f:
            json.dump({"summary": self.summary(), "timeline": self.samples}, f, indent=2)
        return path
 