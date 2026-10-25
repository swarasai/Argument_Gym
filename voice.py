"""Coarse pitch and amplitude analysis; measures variability, never emotional tone."""
import math
import statistics
import numpy as np


def estimate_pitch(pcm, sr=16000):
    x = np.frombuffer(pcm, dtype='<i2').astype(np.float64)
    if len(x) < 400 or np.sqrt(np.mean(x*x)) < 150:
        return None
    x -= x.mean()
    # Reject low-periodicity/noisy frames. Work on only one 100-ms frame.
    corr = np.correlate(x, x, mode='full')[len(x)-1:]
    if corr[0] <= 0:
        return None
    minlag, maxlag = int(sr/350), min(int(sr/75), len(x)-1)
    local = corr[minlag:maxlag+1]
    if not len(local):
        return None
    peaks = [i for i in range(1, len(local)-1) if local[i] > local[i-1] and local[i] >= local[i+1]]
    if not peaks:
        return None
    peak = max(peaks, key=lambda i: local[i])
    lag = peak + minlag
    if corr[lag] / corr[0] < 0.62:
        return None
    return round(sr/lag, 2)


def pitch_band(variation):
    if variation is None:
        return None
    return 'flat' if variation < 1.5 else 'moderate' if variation < 3.5 else 'expressive'


class VoiceTracker:
    def __init__(self):
        self.pitches = []
        self.levels = []

    def on_chunk(self, pcm, db, active_floor):
        if db <= active_floor:
            return
        self.levels.append(db)
        pitch = estimate_pitch(pcm)
        if pitch:
            self.pitches.append(pitch)

    def summary(self):
        levels = self.levels
        pitches = self.pitches
        variation = None
        pitch_range = None
        if len(pitches) >= 8:
            notes = [12*math.log2(x/100) for x in pitches]
            variation = round(statistics.pstdev(notes), 2)
            pitch_range = round(float(np.percentile(notes, 95)-np.percentile(notes, 5)), 2)
        return {
            'pitch_variation_semitones': variation,
            'pitch_range_semitones': pitch_range,
            'pitch_variety_band': pitch_band(variation),
            'loudness_variation_db': round(statistics.pstdev(levels), 2) if len(levels) >= 8 else None,
        }
