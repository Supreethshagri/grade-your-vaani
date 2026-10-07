from grade_your_vaani.dataset import DATA_DIR

VOCAB_FILE = DATA_DIR / "vocab.txt"
MAX_PROMPT_CHARS = 800  # Whisper only uses roughly the last 224 tokens of a prompt
VOCAB_STYLES = ("none", "plain", "localized")

# Generic "bank call" intro in each language, so the prompt matches the audio's language
PROMPT_PREFIX = {"en": "Bank call:", "hi": "बैंक कॉल:", "kn": "ಬ್ಯಾಂಕ್ ಕರೆ:"}


def load_vocab() -> list[str]:
    if not VOCAB_FILE.is_file():
        raise FileNotFoundError(f"vocabulary file not found: {VOCAB_FILE}")
    terms: list[str] = []
    lines = VOCAB_FILE.read_text(encoding="utf-8-sig").splitlines()
    for line_no, line in enumerate(lines, start=1):
        term = line.strip()
        if not term or term.startswith("#"):
            continue
        if " " in term:
            raise ValueError(f"vocab line {line_no}: use single-word terms only ({term!r})")
        if term not in terms:
            terms.append(term)
    if not terms:
        raise ValueError(f"{VOCAB_FILE} has no terms")
    return terms


def build_prompt(terms: list[str], language: str, style: str) -> str:
    if style == "none":
        return ""
    prompt = ", ".join(terms)
    if style == "localized":
        if language not in PROMPT_PREFIX:
            raise ValueError(f"no localized prompt prefix for language {language!r}")
        prompt = f"{PROMPT_PREFIX[language]} {prompt}"
    elif style != "plain":
        raise ValueError(f"unknown vocab style: {style}")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError("vocabulary prompt is too long; Whisper ignores text beyond ~224 tokens")
    return prompt