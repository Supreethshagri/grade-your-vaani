import hashlib
import json
import os
import time
from datetime import datetime

from grade_your_vaani.dataset import PROJECT_ROOT
from grade_your_vaani.providers.groq_whisper import transcribe

CACHE_DIR = PROJECT_ROOT / "cache"

_last_call_start = 0.0


def _cache_key(audio_bytes: bytes, model: str, language: str, prompt: str) -> str:
    h = hashlib.sha256()
    h.update(hashlib.sha256(audio_bytes).digest())
    h.update(json.dumps([model, language, prompt], ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def cached_transcribe(audio_bytes: bytes, filename: str, model: str, language: str,
                      prompt: str = "", use_cache: bool = True,
                      min_interval_s: float = 0.0) -> tuple[str, float, bool]:
    """Returns (text, latency_s, came_from_cache)."""
    global _last_call_start
    path = CACHE_DIR / f"{_cache_key(audio_bytes, model, language, prompt)}.json"

    if use_cache and path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data["text"], float(data["latency_s"]), True
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # corrupted cache file: treat it as a miss and call the API again

    # Pacing: wait so API calls start at least min_interval_s apart.
    # The wait happens BEFORE the timer starts, so it never counts as latency.
    wait = _last_call_start + min_interval_s - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call_start = time.monotonic()

    start = time.perf_counter()
    text = transcribe(audio_bytes, filename, model, language, prompt)
    latency_s = time.perf_counter() - start

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"model": model, "language": language, "prompt": prompt, "text": text,
             "latency_s": round(latency_s, 3),
             "created_at": datetime.now().isoformat(timespec="seconds")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a Ctrl+C can't leave a half-written cache file
    return text, latency_s, False