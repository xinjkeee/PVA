import re


_NON_TEXT_RE = re.compile(r"[^0-9a-zа-яё ]+", flags=re.IGNORECASE)


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = _NON_TEXT_RE.sub(" ", text)
    return " ".join(text.split())


def words(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()
