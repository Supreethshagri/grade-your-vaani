import io
import zlib
from functools import lru_cache

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

from grade_your_vaani.dataset import DATA_DIR
from grade_your_vaani.prepare_audio import to_mono_16k

SR = 16000
TELEPHONY_SR = 8000
MU = 255
NOISY_SNR_DB = 10.0
NOISE_FILE = DATA_DIR / "noise" / "background.wav"
CONDITIONS = ("clean", "telephony", "telephony_noisy")

# Classic telephone band: 300-3400 Hz
PHONE_BAND = butter(4, [300, 3400], btype="bandpass", fs=TELEPHONY_SR, output="sos")


def clip_seed(clip_id: str) -> int:
    # crc32 gives the same number every run; Python's hash() changes between runs
    return zlib.crc32(clip_id.encode("utf-8"))


def mu_law_roundtrip(audio: np.ndarray) -> np.ndarray:
    """Encode to 8-bit mu-law and decode back (continuous mu-law, close to G.711)."""
    audio = np.clip(audio, -1.0, 1.0)
    encoded = np.sign(audio) * np.log1p(MU * np.abs(audio)) / np.log1p(MU)
    quantized = np.round((encoded + 1) / 2 * MU) / MU * 2 - 1  # 256 levels = 8 bits
    return np.sign(quantized) * ((1 + MU) ** np.abs(quantized) - 1) / MU


def telephony(audio: np.ndarray) -> np.ndarray:
    narrow = resample_poly(audio, 1, 2)          # 16k -> 8k, anti-aliased
    narrow = sosfiltfilt(PHONE_BAND, narrow)     # keep only the phone band
    narrow = mu_law_roundtrip(narrow)            # 8-bit codec noise
    return resample_poly(narrow, 2, 1).astype(np.float32)  # back to 16k for the model


@lru_cache(maxsize=1)
def _load_noise_file() -> np.ndarray:
    noise, sr = sf.read(NOISE_FILE, dtype="float32")
    noise = to_mono_16k(noise, sr)
    if np.mean(noise ** 2) == 0:
        raise ValueError(f"{NOISE_FILE} is silent")
    return noise


def _noise_like(n: int, rng: np.random.Generator) -> np.ndarray:
    if not NOISE_FILE.is_file():
        return rng.standard_normal(n).astype(np.float32)
    noise = _load_noise_file()
    if len(noise) < n:
        noise = np.tile(noise, n // len(noise) + 1)
    start = int(rng.integers(0, len(noise) - n + 1))
    return noise[start:start + n]


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    # Note: power is measured over the whole clip, including the silence at the edges
    signal_power = float(np.mean(audio ** 2))
    if signal_power == 0:
        raise ValueError("audio is silent; can't set an SNR")
    noise = _noise_like(len(audio), rng)
    noise_power = float(np.mean(noise ** 2))
    scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    return np.clip(audio + noise * scale, -1.0, 1.0).astype(np.float32)


def apply_condition(audio: np.ndarray, condition: str, seed: int) -> np.ndarray:
    if condition == "clean":
        return audio
    if condition == "telephony":
        return telephony(audio)
    if condition == "telephony_noisy":
        # Noise happens in the caller's room, so it's added BEFORE the phone line
        rng = np.random.default_rng(seed)
        return telephony(add_noise(audio, NOISY_SNR_DB, rng))
    raise ValueError(f"unknown condition: {condition}")


def to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


if __name__ == "__main__":
    from grade_your_vaani.dataset import PROJECT_ROOT, load_manifest

    clip = load_manifest(DATA_DIR / "manifest.csv")[0]
    audio, sr = sf.read(clip.audio_path, dtype="float32")
    audio = to_mono_16k(audio, sr)

    out_dir = PROJECT_ROOT / "results" / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Noise source:", NOISE_FILE.name if NOISE_FILE.is_file() else "white noise")
    for condition in CONDITIONS:
        path = out_dir / f"{clip.clip_id}_{condition}.wav"
        path.write_bytes(to_wav_bytes(apply_condition(audio, condition, clip_seed(clip.clip_id))))
        print(f"wrote {path}")