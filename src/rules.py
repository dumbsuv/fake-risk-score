import re
from typing import Any


URGENCY_MARKERS = [
    "срочно",
    "экстренно",
    "немедленно",
    "только что",
    "важно",
    "молния",
    "пока не удалили",
    "успейте",
]

SENSATIONAL_MARKERS = [
    "шок",
    "сенсация",
    "ужас",
    "катастрофа",
    "скандал",
    "все скрывают",
    "об этом молчат",
    "сми молчат",
    "правду скрывают",
    "никто не говорит",
]

UNCERTAIN_SOURCE_MARKERS = [
    "говорят",
    "по слухам",
    "источник сообщил",
    "источники сообщили",
    "инсайдеры сообщили",
    "очевидцы сообщают",
    "знакомый рассказал",
    "из надежных источников",
    "есть информация",
]

SHARE_CALL_MARKERS = [
    "перешли всем",
    "перешлите всем",
    "сделай репост",
    "сделайте репост",
    "распространите",
    "поделитесь",
    "максимальный репост",
    "не дайте скрыть",
]


def normalize_text(text: str) -> str:
    """
    Приводит текст к нижнему регистру и убирает лишние пробелы.
    """

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_markers(text: str, markers: list[str]) -> list[str]:
    """
    Возвращает список маркеров, найденных в тексте.
    """

    normalized_text = normalize_text(text)
    found = []

    for marker in markers:
        if marker in normalized_text:
            found.append(marker)

    return found


def has_excessive_punctuation(text: str) -> bool:
    """
    Проверяет наличие чрезмерной пунктуации.
    Например: !!!, ???, ?!
    """

    return bool(re.search(r"(!{2,}|\?{2,}|!\?|\?!)", str(text)))


def has_many_uppercase_words(text: str) -> bool:
    """
    Проверяет наличие большого количества слов капсом.
    Для русскоязычных новостей это может быть признаком эмоционального давления.
    """

    words = re.findall(r"\b[А-ЯЁA-Z]{3,}\b", str(text))
    return len(words) >= 3


def has_link(text: str) -> bool:
    """
    Проверяет, есть ли в тексте ссылка или похожий на ссылку фрагмент.
    """

    normalized_text = normalize_text(text)

    link_patterns = [
        "http://",
        "https://",
        "www.",
        ".ru",
        ".com",
        ".org",
        ".рф",
    ]

    return any(pattern in normalized_text for pattern in link_patterns)


def calculate_rule_score(text: str) -> dict[str, Any]:
    """
    Рассчитывает риск по правиловому модулю.

    Возвращает:
    - rule_score: число от 0 до 1;
    - triggered_rules: список сработавших правил;
    - details: подробности по группам признаков.
    """

    urgency = find_markers(text, URGENCY_MARKERS)
    sensational = find_markers(text, SENSATIONAL_MARKERS)
    uncertain_source = find_markers(text, UNCERTAIN_SOURCE_MARKERS)
    share_call = find_markers(text, SHARE_CALL_MARKERS)

    excessive_punctuation = has_excessive_punctuation(text)
    many_uppercase_words = has_many_uppercase_words(text)
    no_link = not has_link(text)

    triggered_rules = []
    points = 0.0

    if urgency:
        triggered_rules.append("маркеры срочности")
        points += 0.20

    if sensational:
        triggered_rules.append("маркеры сенсационности")
        points += 0.20

    if uncertain_source:
        triggered_rules.append("неопределенный источник")
        points += 0.20

    if share_call:
        triggered_rules.append("призыв к распространению")
        points += 0.20

    if excessive_punctuation:
        triggered_rules.append("избыточная пунктуация")
        points += 0.10

    if many_uppercase_words:
        triggered_rules.append("избыточное использование капса")
        points += 0.05

    if no_link:
        triggered_rules.append("отсутствие ссылок")
        points += 0.05

    rule_score = min(points, 1.0)

    return {
        "rule_score": rule_score,
        "triggered_rules": triggered_rules,
        "details": {
            "urgency_markers": urgency,
            "sensational_markers": sensational,
            "uncertain_source_markers": uncertain_source,
            "share_call_markers": share_call,
            "excessive_punctuation": excessive_punctuation,
            "many_uppercase_words": many_uppercase_words,
            "has_link": not no_link,
        },
    }


if __name__ == "__main__":
    test_text = """
    Срочно! СМИ молчат! Очевидцы сообщают, что произошло нечто ужасное.
    Перешлите всем, пока это не удалили!!!
    """

    result = calculate_rule_score(test_text)

    print("Rule score:", result["rule_score"])
    print("Triggered rules:", result["triggered_rules"])
    print("Details:", result["details"])