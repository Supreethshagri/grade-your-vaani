import csv
from pathlib import Path

import soundfile as sf
from pydantic import BaseModel, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (PROJECT_ROOT / "data").resolve()
ALLOWED_LANGUAGES = {"en", "hi", "kn"}
REQUIRED_COLUMNS = {"clip_id", "audio_path", "reference", "language", "tags"}


class Clip(BaseModel):
    clip_id: str
    audio_path: Path
    reference: str
    language: str
    tags: list[str] = []
    sample_rate: int
    channels: int
    duration_s: float

    @field_validator("clip_id", "reference")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be empty")
        return v

    @field_validator("language")
    @classmethod
    def known_language(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ALLOWED_LANGUAGES:
            raise ValueError(f"must be one of {sorted(ALLOWED_LANGUAGES)}")
        return v

    @field_validator("tags", mode="before")
    @classmethod
    def split_tags(cls, v):
        if isinstance(v, str):
            return [t.strip() for t in v.split(";") if t.strip()]
        return v or []


def resolve_audio_path(raw: str) -> Path:
    if not raw:
        raise ValueError("audio_path is empty")
    path = (DATA_DIR / raw).resolve()
    if not path.is_relative_to(DATA_DIR):
        raise ValueError(f"audio path points outside the data folder: {raw}")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"only .wav files are supported: {raw}")
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {raw}")
    return path


def load_manifest(manifest_path: Path) -> list[Clip]:
    clips: list[Clip] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    with open(manifest_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")

        for line_no, raw_row in enumerate(reader, start=2):
            row = {k: (v or "").strip() for k, v in raw_row.items() if k}
            try:
                path = resolve_audio_path(row["audio_path"])
                info = sf.info(str(path))
                clip = Clip(
                    clip_id=row["clip_id"],
                    audio_path=path,
                    reference=row["reference"],
                    language=row["language"],
                    tags=row["tags"],
                    sample_rate=info.samplerate,
                    channels=info.channels,
                    duration_s=info.duration,
                )
                if clip.clip_id in seen_ids:
                    raise ValueError(f"duplicate clip_id: {clip.clip_id}")
                seen_ids.add(clip.clip_id)
                clips.append(clip)
            except (ValueError, FileNotFoundError, RuntimeError) as e:
                errors.append(f"line {line_no}: {e}")

    if errors:
        raise ValueError("manifest has problems:\n" + "\n".join(errors))
    if not clips:
        raise ValueError("manifest has no clips")
    return clips


def audio_warnings(clip: Clip) -> list[str]:
    warnings = []
    if clip.sample_rate != 16000:
        warnings.append(f"sample rate is {clip.sample_rate}, expected 16000")
    if clip.channels != 1:
        warnings.append(f"{clip.channels} channels, expected mono")
    if clip.duration_s < 1:
        warnings.append("shorter than 1s")
    if clip.duration_s > 30:
        warnings.append("longer than 30s")
    return warnings


if __name__ == "__main__":
    try:
        clips = load_manifest(DATA_DIR / "manifest.csv")
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e))

    print(f"Loaded {len(clips)} clips\n")
    seconds_by_lang: dict[str, float] = {}
    for clip in clips:
        warnings = audio_warnings(clip)
        status = "WARN: " + "; ".join(warnings) if warnings else "ok"
        print(f"{clip.clip_id:<10} {clip.language:<3} {clip.duration_s:>5.1f}s  {status}")
        seconds_by_lang[clip.language] = seconds_by_lang.get(clip.language, 0) + clip.duration_s

    print()
    for lang, secs in sorted(seconds_by_lang.items()):
        print(f"{lang}: {secs:.1f}s of audio")