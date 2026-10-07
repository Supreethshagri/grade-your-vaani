import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SUPPORTED_MODELS = ("whisper-large-v3", "whisper-large-v3-turbo")

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        # max_retries: the SDK waits and retries on rate limits (429) and server errors
        _client = Groq(api_key=key, timeout=60.0, max_retries=5)
    return _client


def transcribe(audio_bytes: bytes, filename: str, model: str, language: str,
               prompt: str = "") -> str:
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model: {model}")
    extra = {"prompt": prompt} if prompt else {}
    result = get_client().audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=model,
        language=language,
        temperature=0.0,
        response_format="json",
        **extra,
    )
    return (result.text or "").strip()