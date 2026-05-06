from collections.abc import Sequence

from .text import normalize_text, words


def levenshtein(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + int(ref_item != hyp_item)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def error_rates(reference: str, hypothesis: str) -> dict[str, float | int | str]:
    reference_normalized = normalize_text(reference)
    hypothesis_normalized = normalize_text(hypothesis)

    reference_words = words(reference_normalized)
    hypothesis_words = words(hypothesis_normalized)
    word_errors = levenshtein(reference_words, hypothesis_words)

    reference_chars = list(reference_normalized)
    hypothesis_chars = list(hypothesis_normalized)
    char_errors = levenshtein(reference_chars, hypothesis_chars)

    word_count = len(reference_words)
    char_count = len(reference_chars)

    return {
        "reference_normalized": reference_normalized,
        "hypothesis_normalized": hypothesis_normalized,
        "word_errors": word_errors,
        "word_count": word_count,
        "wer": word_errors / word_count if word_count else 0.0,
        "char_errors": char_errors,
        "char_count": char_count,
        "cer": char_errors / char_count if char_count else 0.0,
    }
