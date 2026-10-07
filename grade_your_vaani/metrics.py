from dataclasses import dataclass

import jiwer


@dataclass
class ClipScore:
    word_errors: int
    ref_words: int
    char_errors: int
    ref_chars: int

    @property
    def wer(self) -> float:
        return self.word_errors / self.ref_words

    @property
    def cer(self) -> float:
        return self.char_errors / self.ref_chars


def score_clip(reference: str, hypothesis: str) -> ClipScore:
    reference = reference.strip()
    hypothesis = hypothesis.strip()
    if not reference:
        raise ValueError("reference is empty")

    if not hypothesis:
        # Model returned nothing: every reference word and character counts as deleted
        return ClipScore(
            word_errors=len(reference.split()),
            ref_words=len(reference.split()),
            char_errors=len(reference),
            ref_chars=len(reference),
        )

    w = jiwer.process_words(reference, hypothesis)
    c = jiwer.process_characters(reference, hypothesis)
    return ClipScore(
        word_errors=w.substitutions + w.deletions + w.insertions,
        ref_words=w.substitutions + w.deletions + w.hits,
        char_errors=c.substitutions + c.deletions + c.insertions,
        ref_chars=c.substitutions + c.deletions + c.hits,
    )


def corpus_wer(scores: list[ClipScore]) -> float:
    if not scores:
        raise ValueError("no scores")
    return sum(s.word_errors for s in scores) / sum(s.ref_words for s in scores)


def corpus_cer(scores: list[ClipScore]) -> float:
    if not scores:
        raise ValueError("no scores")
    return sum(s.char_errors for s in scores) / sum(s.ref_chars for s in scores)

@dataclass
class KeywordScore:
    expected: int
    found: int
    false_alarms: int


def keyword_score(norm_ref: str, norm_hyp: str, terms: list[str]) -> KeywordScore:
    ref_words = set(norm_ref.split())
    hyp_words = set(norm_hyp.split())
    expected = [t for t in terms if t in ref_words]
    return KeywordScore(
        expected=len(expected),
        found=sum(1 for t in expected if t in hyp_words),
        false_alarms=sum(1 for t in terms if t in hyp_words and t not in ref_words),
    )