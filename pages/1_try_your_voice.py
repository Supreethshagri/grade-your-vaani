import io
import os
import threading
import time
from collections import deque

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from grade_your_vaani.checks import check_transcript
from grade_your_vaani.conditions import SR, apply_condition, clip_seed, to_wav_bytes
from grade_your_vaani.display import DEL_STYLE, INS_STYLE, diff_html
from grade_your_vaani.metrics import score_clip
from grade_your_vaani.normalize import normalize
from grade_your_vaani.prepare_audio import to_mono_16k
from grade_your_vaani.providers.groq_whisper import SUPPORTED_MODELS, transcribe

LANGS = {"hi": "Hindi", "kn": "Kannada", "en": "English"}
TRY_CONDITIONS = {"clean": "As recorded", "telephony_noisy": "Phone line + background noise"}
MIN_SECONDS, MAX_SECONDS = 1.0, 30.0
MAX_BYTES = 5 * 1024 * 1024
MAX_REF_CHARS = 300
MAX_TRIES_PER_SESSION = 5
MAX_TRIES_PER_MINUTE = 4  # 4 tries x 4 calls = 16 calls/min, under Groq's free-tier limit
MAX_TRIES_PER_HOUR = 20

st.set_page_config(page_title="Try your voice | Grade Your Vaani", page_icon="🎙️")


@st.cache_resource
def _shared_limiter() -> dict:
    # cache_resource = ONE object shared by every visitor, not one per session
    return {"lock": threading.Lock(), "times": deque()}


def take_global_slot() -> bool:
    limiter = _shared_limiter()
    now = time.monotonic()
    with limiter["lock"]:  # visitors run in parallel threads, so updates must be locked
        times = limiter["times"]
        while times and now - times[0] > 3600:
            times.popleft()
        last_minute = sum(1 for t in times if now - t <= 60)
        if len(times) >= MAX_TRIES_PER_HOUR or last_minute >= MAX_TRIES_PER_MINUTE:
            return False
        times.append(now)
        return True


def load_audio(uploaded) -> tuple[np.ndarray | None, str | None]:
    if uploaded is None:
        return None, "Record or upload some audio first."
    if uploaded.size > MAX_BYTES:
        return None, "That file is over 5 MB."
    try:
        audio, sr = sf.read(io.BytesIO(uploaded.getvalue()), dtype="float32")
    except Exception:
        return None, "Couldn't read that audio. Use WAV, FLAC or OGG."
    audio = to_mono_16k(audio, sr)
    seconds = len(audio) / SR
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        return None, f"The audio is {seconds:.1f} seconds long; it needs to be between 1 and 30."
    if float(np.max(np.abs(audio))) < 1e-3:
        return None, "The recording is silent. Check your microphone."
    return audio, None


def grade(audio: np.ndarray, reference: str, language: str) -> list[dict]:
    norm_ref = normalize(reference, language)
    processed = {c: apply_condition(audio, c, clip_seed("try")) for c in TRY_CONDITIONS}
    results = []
    for model in SUPPORTED_MODELS:
        for condition, label in TRY_CONDITIONS.items():
            start = time.perf_counter()
            try:
                hyp = transcribe(to_wav_bytes(processed[condition]), f"try_{condition}.wav",
                                 model, language)
            except Exception as e:
                # Server log gets the error type only, never the user's audio or text
                print(f"try page: {model} / {condition} failed: {type(e).__name__}")
                results.append({"model": model, "condition": label, "error": True})
                continue
            latency = time.perf_counter() - start
            norm_hyp = normalize(hyp, language)
            score = score_clip(norm_ref, norm_hyp)
            results.append({
                "model": model, "condition": label, "error": False,
                "norm_ref": norm_ref, "norm_hyp": norm_hyp,
                "wer": score.wer, "cer": score.cer, "latency": latency,
                "flags": check_transcript(norm_ref, norm_hyp, language, 0, False),
            })
    return results


st.title("Try it with your own voice")
st.write("Say a sentence in Hindi, Kannada or English, type exactly what you said, and see how "
         "both Whisper models handle your voice: as recorded, and after passing it through a "
         "simulated phone line with background noise.")

if not os.getenv("GROQ_API_KEY"):
    st.warning("Live testing is turned off on this deployment.")
    st.stop()

st.session_state.setdefault("tries", 0)

language = st.selectbox("Language", list(LANGS), format_func=LANGS.get)
reference = st.text_input(
    "What you'll say, typed exactly", max_chars=MAX_REF_CHARS,
    help="Write Hindi and Kannada words in their own script, and acronyms like EMI or UPI "
         "in English letters.",
)
source = st.radio("Audio", ["Record in the browser", "Upload a file"], horizontal=True)
if source == "Record in the browser":
    audio_file = st.audio_input("Record (1 to 30 seconds)")
else:
    audio_file = st.file_uploader("WAV, FLAC or OGG (1 to 30 seconds)", type=["wav", "flac", "ogg"])
st.caption("Your audio is sent to Groq's API to be transcribed. This app doesn't save your "
           "audio or your text.")

out_of_tries = st.session_state["tries"] >= MAX_TRIES_PER_SESSION
if st.button("Grade my voice", type="primary", disabled=out_of_tries):
    audio, error = None, None
    if not normalize(reference, language):
        error = "Type what you said first: the score compares the transcript against it."
    else:
        audio, error = load_audio(audio_file)

    if error:
        st.error(error)
    elif not take_global_slot():
        st.warning("The live demo is busy right now. It's rate-limited to protect the free "
                   "API quota, so try again in a minute.")
    else:
        st.session_state["tries"] += 1
        with st.spinner("Transcribing with both models..."):
            st.session_state["last"] = grade(audio, reference, language)

last = st.session_state.get("last")
if last:
    st.subheader("Results")
    rows = []
    for r in last:
        if r["error"]:
            rows.append({"Model": r["model"], "Audio": r["condition"], "WER": "failed",
                         "CER": "", "Latency": "", "Problems": ""})
        else:
            rows.append({"Model": r["model"], "Audio": r["condition"],
                         "WER": f"{r['wer']:.0%}", "CER": f"{r['cer']:.0%}",
                         "Latency": f"{r['latency']:.2f}s",
                         "Problems": ", ".join(r["flags"]) or "none"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.markdown(f'Differences: <span style="{DEL_STYLE}">missed</span> / '
                f'<span style="{INS_STYLE}">heard instead</span>', unsafe_allow_html=True)
    for r in last:
        if r["error"]:
            continue
        st.markdown(f"**{r['model']} · {r['condition']}**")
        st.markdown(f'<div style="line-height:2.2;font-size:1.05rem">'
                    f'{diff_html(r["norm_ref"], r["norm_hyp"])}</div>', unsafe_allow_html=True)

    st.caption("One recording isn't a benchmark: scores on a single clip swing a lot. Latency here "
               "is measured from the server running this page, not from Bengaluru like the benchmark.")

tries_left = MAX_TRIES_PER_SESSION - st.session_state["tries"]
st.caption(f"{tries_left} of {MAX_TRIES_PER_SESSION} tries left in this session.")