import difflib
import html
import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from grade_your_vaani.normalize import normalize

ROOT = Path(__file__).resolve().parent
PUBLISHED = ROOT / "published" / "runs"
FINDINGS = ROOT / "published" / "findings.md"
AUDIO_DIR = ROOT / "data" / "audio"

LANGS = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
CONDITIONS = ["clean", "telephony", "telephony_noisy"]
VOCAB_LABELS = {"none": "No prompt", "plain": "Plain prompt", "localized": "Localized prompt"}
SAFE_CLIP_ID = re.compile(r"^[A-Za-z0-9_-]+$")
DEL_STYLE = "background:#fdd;color:#900;text-decoration:line-through;padding:0 2px"
INS_STYLE = "background:#dfd;color:#060;padding:0 2px"

st.set_page_config(page_title="Grade Your Vaani", page_icon="🎙️", layout="wide")


@st.cache_data
def load_runs() -> list[dict]:
    runs = []
    if not PUBLISHED.is_dir():
        return runs
    for run_dir in sorted(PUBLISHED.iterdir()):
        try:
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            results = pd.read_csv(run_dir / "results.csv", encoding="utf-8-sig", keep_default_na=False)
            flags = pd.read_csv(run_dir / "flags.csv", encoding="utf-8-sig", keep_default_na=False)
        except (OSError, ValueError):
            continue  # skip anything incomplete instead of crashing the page
        summary.setdefault("vocab", "none")
        runs.append({"summary": summary, "results": results, "flags": flags})
    return runs


def find_run(runs: list[dict], model: str, vocab: str) -> dict | None:
    matches = [r for r in runs if r["summary"]["model"] == model and r["summary"]["vocab"] == vocab]
    return matches[-1] if matches else None


def accuracy_frame(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        s = run["summary"]
        for condition, langs in s.get("by_condition", {}).items():
            for lang, m in langs.items():
                rows.append({"model": s["model"], "vocab": s["vocab"], "condition": condition,
                             "language": LANGS.get(lang, lang), "WER": m["wer"], "CER": m["cer"],
                             "clips": m["clips"]})
    return pd.DataFrame(rows)


def broken_share(run: dict) -> float | None:
    flags = run["flags"]
    if flags.empty or "flags" not in flags:
        return None
    return float((flags["flags"] != "").mean())


def diff_html(ref: str, hyp: str) -> str:
    a, b = ref.split(), hyp.split()
    parts = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            parts.append(html.escape(" ".join(a[i1:i2])))
            continue
        if i2 > i1:
            parts.append(f'<span style="{DEL_STYLE}">{html.escape(" ".join(a[i1:i2]))}</span>')
        if j2 > j1:
            parts.append(f'<span style="{INS_STYLE}">{html.escape(" ".join(b[j1:j2]))}</span>')
    return " ".join(parts)


runs = load_runs()
if not runs:
    st.warning("No published results yet. Run the benchmark, then "
               "`python -m grade_your_vaani.publish <run folders>`.")
    st.stop()

acc = accuracy_frame(runs)
models = sorted(acc["model"].unique())

st.title("Grade Your Vaani")
st.caption("How well does Whisper understand Hindi, Kannada and English: on clean audio, "
           "over a phone line, and with background noise?")
st.info("Small benchmark: 10 clips (4 Hindi, 4 Kannada, 2 English) from one speaker, a simulated "
        "G.711 phone line, and white noise at 10 dB SNR. Treat the results as directional, not definitive.")
st.page_link("pages/1_Try_your_voice.py", label="Try it with your own voice", icon="🎙️")
if FINDINGS.is_file():
    st.subheader("Key findings")
    st.markdown(FINDINGS.read_text(encoding="utf-8"))

# 1. Accuracy by condition
st.header("1. Accuracy by audio condition")
col1, col2 = st.columns(2)
model = col1.selectbox("Model", models)
metric = col2.radio("Metric", ["WER", "CER"], horizontal=True,
                    help="WER counts whole words. CER counts characters, which is fairer for "
                         "Kannada, where words often merge.")
view = acc[(acc["model"] == model) & (acc["vocab"] == "none")]
if view.empty:
    st.write("No baseline (no-prompt) run published for this model.")
else:
    chart = alt.Chart(view).mark_bar().encode(
        x=alt.X("condition:N", sort=CONDITIONS, title=None),
        xOffset="language:N",
        y=alt.Y(f"{metric}:Q", title=metric, axis=alt.Axis(format="%")),
        color=alt.Color("language:N", title="Language"),
        tooltip=["language", "condition", alt.Tooltip(f"{metric}:Q", format=".1%"), "clips"],
    )
    st.altair_chart(chart)

# 2. Speed vs accuracy
st.header("2. large-v3 vs turbo: speed vs accuracy")
table = []
for m in models:
    run = find_run(runs, m, "none")
    if not run:
        continue
    s = run["summary"]
    row = {"Model": m}
    for code, lang in LANGS.items():
        metrics = s["by_condition"].get("clean", {}).get(code)
        if metrics:
            row[f"{lang} WER"] = f"{metrics['wer']:.1%}"
    lat = s.get("latency", {}).get("clean")
    if lat:
        row["Median latency"] = f"{lat['latency_median_s']:.2f}s"
        row["RTF"] = f"{lat['rtf_median']:.3f}" if lat.get("rtf_median") is not None else "n/a"
    table.append(row)
st.dataframe(pd.DataFrame(table), hide_index=True)
st.caption("Clean audio, no prompt. Latency is end-to-end API time from Bengaluru (upload, queue "
           "and model), measured with paced calls after a warm-up request. RTF = latency ÷ audio length.")

# 3. Vocabulary experiment
st.header("3. Does a vocabulary prompt help? (EMI, UPI, KYC, PAN, Aadhaar)")
rows = []
for m in models:
    for vocab, label in VOCAB_LABELS.items():
        run = find_run(runs, m, vocab)
        if not run:
            continue
        s = run["summary"]
        clean = s["by_condition"].get("clean", {})
        kw = s.get("keywords", {}).get("clean", {})
        row = {"Model": m, "Prompt": label,
               "Keyword recall": f"{kw['recall']:.0%}" if kw.get("recall") is not None else "n/a"}
        for code, lang in LANGS.items():
            if code in clean:
                row[f"{lang} WER"] = f"{clean[code]['wer']:.1%}"
        share = broken_share(run)
        row["Broken outputs"] = f"{share:.0%}" if share is not None else "n/a"
        rows.append(row)
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption("Keyword recall and WER are on clean audio. 'Broken outputs' covers all three conditions: "
           "wrong script, translated into English, repeated characters, dropped or invented text, "
           "or the prompt echoed back.")

# 4. Transcript explorer
st.header("4. Look at the transcripts yourself")
labels = {f"{r['summary']['model']} · {VOCAB_LABELS.get(r['summary']['vocab'], r['summary']['vocab'])}": r
          for r in runs}
col1, col2, col3 = st.columns(3)
run = labels[col1.selectbox("Run", list(labels))]
res = run["results"]
clip_id = col2.selectbox("Clip", sorted(res["clip_id"].unique()))
condition = col3.selectbox("Condition", [c for c in CONDITIONS if c in set(res["condition"])])

match = res[(res["clip_id"] == clip_id) & (res["condition"] == condition)]
if match.empty:
    st.write("No result for this combination.")
else:
    row = match.iloc[0]
    if row["error"]:
        st.error(f"This call failed: {row['error']}")
    else:
        lang = row["language"]
        fl = run["flags"]
        fmatch = fl[(fl["clip_id"] == clip_id) & (fl["condition"] == condition)]
        flag_text = fmatch.iloc[0]["flags"] if not fmatch.empty else ""

        m1, m2, m3 = st.columns(3)
        m1.metric("WER", f"{float(row['wer']):.0%}")
        m2.metric("CER", f"{float(row['cer']):.0%}")
        m3.metric("Flags", flag_text.replace(";", ", ") if flag_text else "none")

        st.text(f"Reference:  {row['reference']}")
        st.text(f"Transcript: {row['hypothesis']}")
        st.markdown("**What was scored** (after normalization): "
                    f'<span style="{DEL_STYLE}">missed</span> / '
                    f'<span style="{INS_STYLE}">heard instead</span>', unsafe_allow_html=True)
        diff = diff_html(normalize(row["reference"], lang), normalize(row["hypothesis"], lang))
        st.markdown(f'<div style="line-height:2.2;font-size:1.1rem">{diff}</div>',
                    unsafe_allow_html=True)

        audio_path = AUDIO_DIR / f"{clip_id}.wav"
        if SAFE_CLIP_ID.match(clip_id) and audio_path.is_file():
            st.audio(str(audio_path))
            st.caption("Original recording (clean). The phone-line and noisy versions are generated from it.")

st.divider()
st.caption("Built with Python, Groq Whisper (large-v3 and large-v3-turbo), jiwer, NumPy/SciPy and Streamlit.")