import re
import unicodedata

NORMALIZER_VERSION = "v2"

ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}
ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b")

# Only spellings that are BOTH genuinely correct.
# Never add an entry just because the model got a word wrong on your test set.
VARIANTS: dict[str, dict[str, str]] = {
    "hi": {"रुपए": "रुपये", "दुबारा": "दोबारा"},
    "kn": {},
    "en": {},
}


def _ascii_digits(text: str) -> str:
    # १५०० and ೧೫೦೦ are the same number as 1500
    return "".join(str(unicodedata.decimal(ch)) if unicodedata.category(ch) == "Nd" else ch
                   for ch in text)


def _strip_punctuation(text: str) -> str:
    out = []
    for i, ch in enumerate(text):
        if not unicodedata.category(ch).startswith("P"):
            out.append(ch)
            continue
        between_digits = 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit()
        if between_digits and ch == ",":
            continue            # 1,500 and 1,50,000 -> 1500 and 150000
        if between_digits and ch == ".":
            out.append(ch)      # keep decimals like 2.5
            continue
        out.append(" ")
    return "".join(out)


def normalize(text: str, language: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch not in ZERO_WIDTH)
    text = _ascii_digits(text)
    text = text.lower()
    text = _strip_punctuation(text)
    text = ORDINAL.sub(r"\1", text)
    variants = VARIANTS.get(language, {})
    return " ".join(variants.get(word, word) for word in text.split())


if __name__ == "__main__":
    samples = [
        ("Please transfer 2500 rupees via UPI to Ramesh.", "en"),
        ("My Aadhar and Paan details were updated on 12th March.", "en"),
        ("मैंने कल 15,000 रुपए UPSA बेजे थे।", "hi"),
        ("मैंने कल १५०० रुपये भेजे थे", "hi"),
        ("Interest is 2.5 percent on 1,50,000", "en"),
        ("ದೈವಿಟು ನಿಮ್ಮ ವಿಳಾಸವನ್ನು ಮತ್ತಮೆ ಹೇಳಿ.", "kn"),
    ]
    for text, lang in samples:
        print(f"{text}\n  -> {normalize(text, lang)}\n")