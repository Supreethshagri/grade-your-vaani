from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from grade_your_vaani.dataset import DATA_DIR

RAW_DIR = DATA_DIR / "raw_audio"
OUT_DIR = DATA_DIR / "audio"
TARGET_SR = 16000


def to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # average the channels into one
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def main() -> None:
    if not RAW_DIR.is_dir():
        raise SystemExit(f"Put your original recordings in {RAW_DIR}")
    files = sorted(RAW_DIR.glob("*.wav"))
    if not files:
        raise SystemExit(f"No .wav files found in {RAW_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in files:
        audio, sr = sf.read(src, dtype="float32")
        channels = audio.shape[1] if audio.ndim == 2 else 1
        out = to_mono_16k(audio, sr)
        sf.write(OUT_DIR / src.name, out, TARGET_SR, subtype="PCM_16")
        print(f"{src.name}: {sr} Hz, {channels} ch -> {TARGET_SR} Hz, 1 ch")


if __name__ == "__main__":
    main()