import argparse
import csv
import json
from datetime import datetime
from statistics import median

import numpy as np
import soundfile as sf

from grade_your_vaani.cache import cached_transcribe
from grade_your_vaani.conditions import (
    CONDITIONS, NOISE_FILE, NOISY_SNR_DB, SR, apply_condition, clip_seed, to_wav_bytes,
)
from grade_your_vaani.dataset import DATA_DIR, PROJECT_ROOT, load_manifest
from grade_your_vaani.metrics import (
    ClipScore, KeywordScore, corpus_cer, corpus_wer, keyword_score, score_clip,
)
from grade_your_vaani.normalize import NORMALIZER_VERSION, normalize
from grade_your_vaani.prepare_audio import to_mono_16k
from grade_your_vaani.providers.groq_whisper import SUPPORTED_MODELS
from grade_your_vaani.vocab import VOCAB_STYLES, build_prompt, load_vocab

RESULTS_DIR = PROJECT_ROOT / "results"
FIELDS = ["clip_id", "language", "condition", "model", "vocab", "reference", "hypothesis",
          "wer_raw", "cer_raw", "wer", "cer", "kw_expected", "kw_found", "kw_false_alarms",
          "audio_s", "latency_s", "rtf", "cached", "error"]
SLOW_CALL_S = 2.0  # rough heuristic: a short clip taking this long probably waited on a retry


def summarize(scores: list[ClipScore]) -> tuple[float, float]:
    return round(corpus_wer(scores), 4), round(corpus_cer(scores), 4)


def latency_stats(latencies: list[float], durations: list[float]) -> dict:
    rtfs = [lat / dur for lat, dur in zip(latencies, durations) if dur > 0]
    return {
        "calls": len(latencies),
        "latency_median_s": round(median(latencies), 3),
        "latency_max_s": round(max(latencies), 3),
        "rtf_median": round(median(rtfs), 3) if rtfs else None,
    }


def keyword_summary(scores: list[KeywordScore]) -> dict:
    expected = sum(s.expected for s in scores)
    found = sum(s.found for s in scores)
    return {
        "expected": expected,
        "found": found,
        "recall": round(found / expected, 4) if expected else None,
        "false_alarms": sum(s.false_alarms for s in scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade Your Vaani: run an ASR benchmark")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="whisper-large-v3")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore cached transcripts and call the API again")
    parser.add_argument("--pace", type=float, default=0.0,
                        help="minimum seconds between API calls (3.1 keeps under 20 requests/min)")
    parser.add_argument("--vocab", choices=VOCAB_STYLES, default="none",
                        help="none, plain (terms only) or localized (terms in the audio's language)")
    args = parser.parse_args()
    if args.pace < 0:
        raise SystemExit("--pace can't be negative")

    try:
        clips = load_manifest(DATA_DIR / "manifest.csv")
        terms = load_vocab()
        languages = sorted({clip.language for clip in clips})
        prompts = {lang: build_prompt(terms, lang, args.vocab) for lang in languages}
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e))
    kw_terms = [normalize(t, "en") for t in terms]

    suffix = "" if args.vocab == "none" else f"_vocab-{args.vocab}"
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{args.model}{suffix}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    if args.no_cache or args.pace > 0:
        # Warm-up: the first request also pays for connection setup (DNS, TCP, TLS).
        # One throwaway call means no benchmark clip absorbs that cost.
        try:
            cached_transcribe(to_wav_bytes(np.zeros(SR, dtype=np.float32)), "warmup.wav",
                              args.model, "en", use_cache=False, min_interval_s=args.pace)
        except Exception as e:
            print(f"Warm-up call failed, continuing: {e}")

    rows: list[dict] = []
    raw_scores: dict[tuple[str, str], list[ClipScore]] = {}
    norm_scores: dict[tuple[str, str], list[ClipScore]] = {}
    kw_by_cond: dict[str, list[KeywordScore]] = {}
    lat_by_cond: dict[str, tuple[list[float], list[float]]] = {}
    failures = cache_hits = slow_calls = total = 0

    for clip in clips:
        audio, sr = sf.read(clip.audio_path, dtype="float32")
        audio = to_mono_16k(audio, sr)

        for condition in args.conditions:
            total += 1
            base = {"clip_id": clip.clip_id, "language": clip.language, "condition": condition,
                    "model": args.model, "vocab": args.vocab, "reference": clip.reference}
            try:
                processed = apply_condition(audio, condition, clip_seed(clip.clip_id))
                audio_s = len(processed) / SR
                hypothesis, latency_s, cached = cached_transcribe(
                    to_wav_bytes(processed), f"{clip.clip_id}_{condition}.wav",
                    args.model, clip.language, prompt=prompts[clip.language],
                    use_cache=not args.no_cache, min_interval_s=args.pace,
                )
            except Exception as e:  # one failed call shouldn't kill the whole run
                failures += 1
                print(f"{clip.clip_id:<8} {condition:<16} FAILED: {e}")
                rows.append({**base, "hypothesis": "", "wer_raw": "", "cer_raw": "",
                             "wer": "", "cer": "", "kw_expected": "", "kw_found": "",
                             "kw_false_alarms": "", "audio_s": "", "latency_s": "",
                             "rtf": "", "cached": "", "error": str(e)})
                continue

            cache_hits += cached
            if not cached and latency_s > SLOW_CALL_S:
                slow_calls += 1

            raw = score_clip(clip.reference, hypothesis)
            norm_ref = normalize(clip.reference, clip.language)
            norm_hyp = normalize(hypothesis, clip.language)
            norm = score_clip(norm_ref, norm_hyp)
            kw = keyword_score(norm_ref, norm_hyp, kw_terms)

            key = (condition, clip.language)
            raw_scores.setdefault(key, []).append(raw)
            norm_scores.setdefault(key, []).append(norm)
            kw_by_cond.setdefault(condition, []).append(kw)
            lats, durs = lat_by_cond.setdefault(condition, ([], []))
            lats.append(latency_s)
            durs.append(audio_s)

            rows.append({**base, "hypothesis": hypothesis,
                         "wer_raw": round(raw.wer, 4), "cer_raw": round(raw.cer, 4),
                         "wer": round(norm.wer, 4), "cer": round(norm.cer, 4),
                         "kw_expected": kw.expected, "kw_found": kw.found,
                         "kw_false_alarms": kw.false_alarms,
                         "audio_s": round(audio_s, 2), "latency_s": round(latency_s, 3),
                         "rtf": round(latency_s / audio_s, 3) if audio_s > 0 else "",
                         "cached": "yes" if cached else "no", "error": ""})

            kw_note = f"  kw {kw.found}/{kw.expected}" if kw.expected else ""
            if kw.false_alarms:
                kw_note += f" (+{kw.false_alarms} false)"
            source = "cached" if cached else f"{latency_s:.2f}s"
            print(f"{clip.clip_id:<8} {condition:<16} WER {norm.wer:6.1%}  "
                  f"CER {norm.cer:6.1%}{kw_note}  [{source}]  hyp: {norm_hyp}")

    with open(run_dir / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_condition: dict[str, dict] = {}
    for condition in args.conditions:
        for lang in sorted(lang for (c, lang) in norm_scores if c == condition):
            wer_raw, cer_raw = summarize(raw_scores[(condition, lang)])
            wer, cer = summarize(norm_scores[(condition, lang)])
            by_condition.setdefault(condition, {})[lang] = {
                "clips": len(norm_scores[(condition, lang)]),
                "wer_raw": wer_raw, "cer_raw": cer_raw, "wer": wer, "cer": cer,
            }

    keywords = {cond: keyword_summary(scores) for cond, scores in kw_by_cond.items()}
    latency = {cond: latency_stats(lats, durs) for cond, (lats, durs) in lat_by_cond.items()}
    successes = total - failures
    if cache_hits == 0:
        latency_source = "fresh API calls"
    elif cache_hits == successes:
        latency_source = "cache (original call times)"
    else:
        latency_source = "mixed fresh and cached"

    summary = {
        "run_id": run_id,
        "model": args.model,
        "normalizer": NORMALIZER_VERSION,
        "vocab": args.vocab,
        "prompts": prompts,
        "vocab_terms": terms,
        "conditions": args.conditions,
        "noisy_snr_db": NOISY_SNR_DB,
        "noise_source": NOISE_FILE.name if NOISE_FILE.is_file() else "white noise",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "calls": total,
        "cache_hits": cache_hits,
        "failures": failures,
        "pace_s": args.pace,
        "latency_source": latency_source,
        "slow_calls": slow_calls,
        "by_condition": by_condition,
        "keywords": keywords,
        "latency": latency,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nNormalized WER / CER by condition")
    for condition, langs in by_condition.items():
        print(f"  {condition}")
        for lang, s in langs.items():
            print(f"    {lang}: WER {s['wer']:6.1%}   CER {s['cer']:6.1%}   ({s['clips']} clips)")

    print(f"\nDomain keywords (vocab: {args.vocab})")
    for condition, k in keywords.items():
        recall = f"{k['recall']:.0%}" if k["recall"] is not None else "n/a"
        print(f"  {condition:<16} found {k['found']}/{k['expected']} ({recall})   "
              f"false alarms {k['false_alarms']}")

    print(f"\nLatency per call ({latency_source})")
    for condition, s in latency.items():
        rtf = f"{s['rtf_median']:.3f}" if s["rtf_median"] is not None else "n/a"
        print(f"  {condition:<16} median {s['latency_median_s']:.2f}s   "
              f"max {s['latency_max_s']:.2f}s   RTF {rtf}")

    if slow_calls:
        print(f"\nWarning: {slow_calls} fresh calls took over {SLOW_CALL_S}s, possibly "
              f"rate-limit retries. Re-run with --no-cache --pace 3.1 for clean timings.")
    print(f"\nAPI calls: {successes - cache_hits}   cache hits: {cache_hits}   failures: {failures}")
    print(f"Saved to {run_dir}")

    if failures == total:
        raise SystemExit("All calls failed. Check the errors above.")


if __name__ == "__main__":
    main()