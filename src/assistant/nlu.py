from dataclasses import dataclass, field
from datetime import datetime

from src.asr.text import normalize_text


@dataclass
class NLUResult:
    intent: str
    response_text: str
    confidence: float
    slots: dict[str, str] = field(default_factory=dict)


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def interpret_text(text: str, now: datetime | None = None) -> NLUResult:
    normalized = normalize_text(text)
    current_time = now or datetime.now()

    if not normalized:
        return NLUResult(
            intent="empty",
            response_text="Я не смогла разобрать фразу. Повторите, пожалуйста.",
            confidence=0.0,
        )

    if _contains_any(normalized, ("привет", "здравствуй", "добрый день", "доброе утро")):
        return NLUResult(
            intent="greeting",
            response_text="Привет! Я на связи.",
            confidence=0.95,
        )

    if _contains_any(normalized, ("который час", "сколько времени", "текущее время")):
        return NLUResult(
            intent="time",
            response_text=f"Сейчас {current_time:%H:%M}.",
            confidence=0.9,
        )

    if _contains_any(normalized, ("какое сегодня число", "какая дата", "сегодня дата")):
        return NLUResult(
            intent="date",
            response_text=f"Сегодня {current_time:%d.%m.%Y}.",
            confidence=0.9,
        )

    if _contains_any(normalized, ("как тебя зовут", "кто ты", "что ты умеешь")):
        return NLUResult(
            intent="about",
            response_text=(
                "Я прототип голосового ассистента: слышу wake word, распознаю речь "
                "и отвечаю простыми правилами."
            ),
            confidence=0.85,
        )

    if _contains_any(normalized, ("погода", "температура", "дождь")):
        return NLUResult(
            intent="weather_stub",
            response_text="Погода пока не подключена. В следующей версии добавим внешний источник данных.",
            confidence=0.75,
        )

    if _contains_any(normalized, ("спасибо", "благодарю")):
        return NLUResult(
            intent="thanks",
            response_text="Пожалуйста!",
            confidence=0.9,
        )

    if _contains_any(normalized, ("пока", "до свидания", "выключись")):
        return NLUResult(
            intent="goodbye",
            response_text="Хорошо, до встречи.",
            confidence=0.85,
        )

    return NLUResult(
        intent="fallback",
        response_text=f"Я услышала: {text}. Пока отвечаю только на простые команды.",
        confidence=0.35,
    )
