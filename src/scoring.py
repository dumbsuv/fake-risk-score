from typing import Any

from config import (
    SCORING_WEIGHTS,
    PAIR_SCORING_WEIGHTS,
    LOW_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
)

from hf_models import HuggingFaceModels
from rules import calculate_rule_score


def get_risk_level(fake_score: float) -> str:
    """
    Переводит числовой fake_score в уровень риска.
    """

    if fake_score < LOW_RISK_THRESHOLD:
        return "низкий"

    if fake_score < HIGH_RISK_THRESHOLD:
        return "средний"

    return "высокий"


def get_recommendation(risk_level: str) -> str:
    """
    Формирует текстовую рекомендацию по результату анализа.
    """

    if risk_level == "низкий":
        return (
            "Публикация не содержит выраженных признаков фейковости по выбранным "
            "моделям и правилам. Дополнительная проверка может быть выполнена "
            "при высокой значимости темы."
        )

    if risk_level == "средний":
        return (
            "Публикация содержит отдельные признаки подозрительности. "
            "Рекомендуется проверить источник, дату публикации, первоисточник "
            "и фактические утверждения."
        )

    return (
        "Публикация содержит выраженные признаки подозрительности. "
        "Требуется дополнительная проверка источника, контекста, фактических "
        "утверждений и возможных признаков манипулятивной подачи."
    )


def calculate_fake_score(
    semantic_fake_risk: float,
    emotion_risk: float,
    toxicity_risk: float,
    rule_score: float,
    contradiction_risk: float | None = None,
) -> float:
    """
    Рассчитывает итоговый риск-скор.

    Если contradiction_risk отсутствует, используется формула для цельного текста.

    Если contradiction_risk есть, используется формула для пары:
    заголовок + основной текст.
    """

    if contradiction_risk is None:
        fake_score = (
            SCORING_WEIGHTS["semantic_fake_risk"] * semantic_fake_risk
            + SCORING_WEIGHTS["emotion_risk"] * emotion_risk
            + SCORING_WEIGHTS["toxicity_risk"] * toxicity_risk
            + SCORING_WEIGHTS["rule_score"] * rule_score
        )
    else:
        fake_score = (
            PAIR_SCORING_WEIGHTS["contradiction_risk"] * contradiction_risk
            + PAIR_SCORING_WEIGHTS["semantic_fake_risk"] * semantic_fake_risk
            + PAIR_SCORING_WEIGHTS["emotion_risk"] * emotion_risk
            + PAIR_SCORING_WEIGHTS["toxicity_risk"] * toxicity_risk
            + PAIR_SCORING_WEIGHTS["rule_score"] * rule_score
        )

    return min(max(fake_score, 0.0), 1.0)


class FakeRiskScorer:
    """
    Основной класс скорингового инструмента.

    Он объединяет:
    - модели Hugging Face;
    - собственный правиловый модуль;
    - итоговую интерпретируемую формулу fake_score.
    """

    def __init__(self):
        self.models = HuggingFaceModels()

    def analyze(
        self,
        text: str,
        headline: str | None = None,
        article_body: str | None = None,
    ) -> dict[str, Any]:
        """
        Анализирует текст.

        Если переданы headline и article_body, дополнительно оценивается
        противоречие между заголовком и основным текстом.

        Если передан только text, используется обычная оценка цельного текста.
        """

        if headline and article_body:
            model_result = self.models.analyze_pair(
                headline=headline,
                article_body=article_body,
            )
            rule_text = f"{headline}. {article_body}"
        else:
            model_result = self.models.analyze_text(text)
            rule_text = text

        rule_result = calculate_rule_score(rule_text)

        semantic_fake_risk = model_result["semantic_fake_risk"]
        emotion_risk = model_result["emotion_risk"]
        toxicity_risk = model_result["toxicity_risk"]
        rule_score = rule_result["rule_score"]
        contradiction_risk = model_result.get("contradiction_risk")

        fake_score = calculate_fake_score(
            semantic_fake_risk=semantic_fake_risk,
            emotion_risk=emotion_risk,
            toxicity_risk=toxicity_risk,
            rule_score=rule_score,
            contradiction_risk=contradiction_risk,
        )

        risk_level = get_risk_level(fake_score)
        recommendation = get_recommendation(risk_level)

        model_scores = {
            "semantic_fake_risk": semantic_fake_risk,
            "emotion_risk": emotion_risk,
            "toxicity_risk": toxicity_risk,
        }

        if contradiction_risk is not None:
            model_scores["contradiction_risk"] = contradiction_risk

        return {
            "fake_score": fake_score,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "model_scores": model_scores,
            "rule_score": rule_score,
            "triggered_rules": rule_result["triggered_rules"],
            "details": {
                "semantic_scores": model_result.get("semantic_scores", {}),
                "nli_pair_scores": model_result.get("nli_pair_scores", {}),
                "emotion_scores": model_result.get("emotion_scores", {}),
                "toxicity_scores": model_result.get("toxicity_scores", {}),
                "rule_details": rule_result["details"],
            },
        }


if __name__ == "__main__":
    test_headline = "Банк России повысил ключевую ставку."

    test_article_body = (
        "Банк России сообщил о решении снизить ключевую ставку. "
        "Информация опубликована на официальном сайте регулятора."
    )

    test_text = f"{test_headline}. {test_article_body}"

    scorer = FakeRiskScorer()
    result = scorer.analyze(
        text=test_text,
        headline=test_headline,
        article_body=test_article_body,
    )

    print()
    print("Итоговый fake_score:", round(result["fake_score"], 4))
    print("Уровень риска:", result["risk_level"])
    print("Рекомендация:", result["recommendation"])

    print()
    print("Model scores:")
    for key, value in result["model_scores"].items():
        print(f"- {key}: {value:.4f}")

    print()
    print("Rule score:", result["rule_score"])
    print("Triggered rules:")
    for rule in result["triggered_rules"]:
        print(f"- {rule}")

    print()
    print("NLI pair scores:")
    for label, score in result["details"]["nli_pair_scores"].items():
        print(f"- {label}: {score:.4f}")