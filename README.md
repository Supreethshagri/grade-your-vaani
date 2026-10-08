# Grade Your Vaani

How well does Whisper understand Hindi, Kannada and English, on clean audio, over a phone line, and with background noise?

Most ASR benchmarks use clean English audio. Indian customer calls are noisy, code-mixed and full of banking acronyms, so I built a small benchmark to see how a popular model actually holds up there.

It runs the same recordings through Whisper large-v3 and large-v3-turbo (via Groq) under three audio conditions. It scores them with WER/CER after careful normalization and measures latency. It also tests whether vocabulary hints help with terms like EMI, UPI and KYC, and flags outputs that are broken rather than just inaccurate.

**Live demo:** https://grade-your-vaani.onrender.com
It has a results dashboard, plus a page where you can record your own voice and see how both models do.

> Hosted on Render's free tier. If it has been idle, the first load takes about a minute.

## Key findings

- **English was easy; Indian languages weren't.** On clean audio, large-v3 scored 11% WER on English, 24% on Hindi and 68% on Kannada (28% CER).
- **The phone line itself barely hurt; noise did.** A simulated G.711 phone line left Hindi WER at 24%. Adding 10 dB background noise pushed it to 45%, and pushed Kannada CER from 28% to 57%. English didn't move.
- **turbo is about 3x faster, but much worse on Indian languages.** Its median latency was ~0.25 s vs ~0.65-0.82 s for large-v3, but Hindi WER more than doubled (24% → 55%).
- **Vocabulary prompts are language-dependent.** A prompt listing banking terms took English to 0% WER and raised keyword recall from 33% to 67%. It also made Whisper translate every Kannada clip into English (Kannada WER 68% → 124%).
- **On unintelligible audio, the model can echo the prompt.** turbo once returned the entire vocabulary list as the "transcript", so a call where nobody said NEFT would get tagged NEFT.
- **Raw WER overstates errors.** Without normalization, English WER was 28%, and almost all of that was punctuation and number formatting.

**What I'd recommend from this data:** use large-v3 for Hindi and Kannada. turbo is fine for English when speed matters. Use vocabulary prompts for English (and cautiously for Hindi), but not for Kannada.

## What it measures

| | |
|---|---|
| Models | whisper-large-v3, whisper-large-v3-turbo (Groq API) |
| Languages | Hindi, Kannada, English, including code-mixed sentences like "ನನ್ನ EMI ಮುಂದಿನ ಸೋಮವಾರ ಕಟ್ ಆಗುತ್ತೆ" |
| Audio conditions | clean · phone line (8 kHz, 300-3400 Hz, μ-law) · phone line + white noise at 10 dB SNR |
| Metrics | WER and CER (raw and normalized), keyword recall and false alarms, latency, real-time factor, broken-output flags |

## Results

Normalized WER / CER, no prompt:

| Condition | Language | large-v3 | turbo |
|---|---|---|---|
| Clean | English | 11.1% / 2.1% | 5.6% / 1.0% |
| | Hindi | 24.1% / 7.6% | 55.2% / 32.8% |
| | Kannada | 68.0% / 28.2% | 80.0% / 38.5% |
| Phone line | English | 5.6% / 1.0% | 0.0% / 0.0% |
| | Hindi | 24.1% / 7.6% | 65.5% / 35.9% |
| | Kannada | 68.0% / 32.0% | 76.0% / 37.8% |
| Phone + noise | English | 11.1% / 2.1% | 11.1% / 5.2% |
| | Hindi | 44.8% / 25.2% | 65.5% / 43.5% |
| | Kannada | 84.0% / 57.0% | 92.0% / 72.4% |

For Kannada, read CER first. Kannada joins words all the time ("ಕಟ್ ಆಗುತ್ತೆ" vs "ಕಟ್ಟಾಗುತ್ತೆ"), and WER counts every word-boundary difference as a full error.

### Vocabulary prompt experiment (clean audio)

The prompt lists EMI, UPI, KYC, PAN and Aadhaar, plus five terms that never appear in the test set (NEFT, IMPS, RTGS, CIBIL, GST) to catch false alarms. "Localized" adds a short "bank call" intro in the audio's own language before the terms.

| Model | Prompt | Keyword recall | English WER | Hindi WER | Kannada WER | Broken outputs* |
|---|---|---|---|---|---|---|
| large-v3 | none | 33% | 11.1% | 24.1% | 68.0% | 0% |
| large-v3 | plain | 67% | 0.0% | 20.7% | 124.0% | 40% |
| large-v3 | localized | 56% | 0.0% | 20.7% | 60.0% | 13% |
| turbo | none | 33% | 5.6% | 55.2% | 80.0% | 3% |
| turbo | plain | 56% | 0.0% | 51.7% | 104.0% | 53% |
| turbo | localized | 56% | 0.0% | 48.3% | 80.0% | 17% |

\* Share of all 30 transcripts in the run (all three conditions) flagged as wrong script, translated, repeated, truncated, padded, or echoing the prompt.

Keyword recall rose under the plain prompt partly *because* Kannada was translated into English ("your kyc has not been completed yet"). Looking at recall alone, that would have counted as a win, which is why every run also reports WER and broken-output flags.

## How it works

```
data/manifest.csv + recordings
  -> prepare_audio   resample to 16 kHz mono (anti-aliased, originals untouched)
  -> dataset         validate every clip before any API call
  -> conditions      clean / phone line / phone line + noise, built in memory
  -> Groq Whisper    cached by audio hash + model + language + prompt
  -> normalize       Unicode NFC, punctuation, digits, ordinals, known spelling variants
  -> metrics         corpus-level WER/CER, keyword recall, latency, RTF
  -> checks          flag broken outputs
  -> publish         copy chosen runs to published/
  -> app.py          Streamlit dashboard + live "try your voice" page
```

## Design decisions

- **Corpus-level WER**, not the average of per-clip WER, so a 3-word clip doesn't count as much as a 30-word one.
- **One rule for normalization:** only remove differences a human would say don't matter. रुपए/रुपये is normalized because both spellings are correct. Aadhar/Aadhaar is not, because getting domain terms right is the point.
- **Noise is added before the phone line,** because background noise is in the caller's room and goes through the codec together with the speech.
- **Reproducible noise:** seeds come from `crc32(clip_id)`, not Python's `hash()`, which changes on every run.
- **Content-addressed cache** (SHA-256 of the audio plus settings). Changing how results are scored never costs an API call, and re-recording a clip automatically misses the old cache entry.
- **Clean latency numbers:** calls are paced to stay under the free tier's ~20 requests/minute, and a throwaway warm-up call absorbs connection setup. Before I added these, rate-limit retries silently inflated a third of the measurements.
- **Distractor vocabulary terms:** a real client hands over their whole vocabulary list, not just the words in your test set, so the prompt test includes terms that are never spoken.
- **Live demo privacy:** user audio is processed in memory and never cached or saved, and the server logs only the error type. A shared rate limit (4 tries/minute, 20/hour across all visitors) plus a per-session limit protects the API quota.

## Limitations

This is a small, honest benchmark, not a definitive one:

- **10 clips (4 Hindi, 4 Kannada, 2 English), one speaker, and only 9 domain-keyword occurrences.** One clip can move a percentage a lot, so treat the numbers as directional.
- **The phone line is simulated** (continuous μ-law, close to G.711). Real calls add compressed mobile codecs, packet loss and echo, so real-world results are likely worse.
- **The noise is synthetic white noise,** and SNR is computed over the whole clip, including the silence at the start and end.
- **Groq's Whisper endpoint is batch-only,** so this measures batch latency, not streaming. Latency is end-to-end from Bengaluru, and it varied by up to ~2x between runs at different times of day.
- **Broken outputs aren't repeatable.** Normal transcripts were identical across runs, but failures changed: one clip came out in Khmer script in one run and as repeated "ಠ" in another. The cache keeps one sample.
- **The broken-output checks are heuristics** with documented thresholds. Known misses: a word repeated only twice, and words leaking from the localized prompt's intro phrase.
- Only Whisper via Groq is tested; no other ASR providers.

## Run it yourself (Windows)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with `GROQ_API_KEY=your_key`, put recordings in `data\raw_audio\`, and list them in `data\manifest.csv`. Then:

```
python -m grade_your_vaani.prepare_audio
python -m grade_your_vaani.dataset
python -m grade_your_vaani.run --pace 3.1
python -m grade_your_vaani.run --model whisper-large-v3-turbo --pace 3.1
python -m grade_your_vaani.run --vocab plain --pace 3.1
python -m grade_your_vaani.checks
python -m grade_your_vaani.publish <run folder names>
streamlit run app.py
```

Useful flags: `--conditions clean` (fewer API calls), `--no-cache` (fresh latency), `--vocab none|plain|localized`.

## Project structure

```
grade-your-vaani/
├── app.py                        results dashboard
├── pages/1_Try_your_voice.py     live demo page
├── .streamlit/config.toml        upload size limit
├── data/
│   ├── manifest.csv              clips, transcripts, language, tags
│   ├── vocab.txt                 domain terms for the prompt experiment
│   └── audio/                    16 kHz mono recordings
├── published/
│   ├── findings.md
│   └── runs/                     results shown on the dashboard
└── grade_your_vaani/
    ├── dataset.py                manifest + audio validation
    ├── prepare_audio.py          resampling and downmixing
    ├── conditions.py             phone line, μ-law, noise
    ├── providers/groq_whisper.py Groq API client
    ├── cache.py                  transcript cache, pacing, latency
    ├── normalize.py              text normalization
    ├── metrics.py                WER, CER, keyword recall
    ├── vocab.py                  vocabulary prompts
    ├── checks.py                 broken-output detection
    ├── publish.py                choose runs for the dashboard
    ├── display.py                transcript diff rendering
    └── run.py                    benchmark CLI
```

## What I'd do next

- Grow to ~100 clips with more speakers (adding a subset of Google's FLEURS) and real recorded background noise.
- Add another ASR provider behind the same interface for a cross-vendor comparison.
- Test real 8 kHz call recordings instead of simulated ones.

## Stack

Python, Groq Whisper API, NumPy, SciPy, soundfile, jiwer, Pydantic, Streamlit, Altair, Render.
