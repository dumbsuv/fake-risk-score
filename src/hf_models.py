from typing import Any

from transformers import pipeline

from config import MODEL_EMOTION, MODEL_TOXICITY, MODEL_NLI


class HuggingFaceModels:
    """
    Класс для загрузки и запуска моделей Hugging Face.

    Используются три модели:
    1. NLI-модель для zero-shot оценки текста;
    2. NLI-модель для проверки противоречия между заголовком и текстом;
    3. модель эмоций;
    4. модель токсичности.

    Технически NLI-модель одна и та же, но используется в двух режимах:
    - zero-shot-classification;
    - text-classification для пары premise/hypothesis.
    """

    def __init__(self):
        self.emotion_pipeline = None
        self.toxicity_pipeline = None
        self.nli_zero_shot_pipeline = None
        self.nli_pair_pipeline = None

    def load_emotion_model(self):
        """
        Загружает модель определения эмоций.
        """

        if self.emotion_pipeline is None:
            print("Загружаю модель эмоций...")
            self.emotion_pipeline = pipeline(
                task="text-classification",
                model=MODEL_EMOTION,
                top_k=None,
            )
            print("Модель эмоций загружена.")

        return self.emotion_pipeline

    def load_toxicity_model(self):
        """
        Загружает модель определения токсичности.
        """

        if self.toxicity_pipeline is None:
            print("Загружаю модель токсичности...")
            self.toxicity_pipeline = pipeline(
                task="text-classification",
                model=MODEL_TOXICITY,
                top_k=None,
            )
            print("Модель токсичности загружена.")

        return self.toxicity_pipeline

    def load_nli_zero_shot_model(self):
        """
        Загружает NLI-модель для zero-shot классификации.
        """

        if self.nli_zero_shot_pipeline is None:
            print("Загружаю NLI-модель для zero-shot анализа...")
            self.nli_zero_shot_pipeline = pipeline(
                task="zero-shot-classification",
                model=MODEL_NLI,
            )
            print("NLI-модель для zero-shot анализа загружена.")

        return self.nli_zero_shot_pipeline

    def load_nli_pair_model(self):
        """
        Загружает NLI-модель для анализа пары:
        основной текст + заголовок.
        """

        if self.nli_pair_pipeline is None:
            print("Загружаю NLI-модель для анализа пары заголовок/текст...")
            self.nli_pair_pipeline = pipeline(
                task="text-classification",
                model=MODEL_NLI,
                top_k=None,
            )
            print("NLI-модель для анализа пары загружена.")

        return self.nli_pair_pipeline

    def predict_emotion(self, text: str) -> dict[str, Any]:
        """
        Возвращает вероятности эмоций для текста.
        """

        emotion_model = self.load_emotion_model()

        result = emotion_model(
            text,
            truncation=True,
            max_length=512,
        )
        result = normalize_pipeline_result(result)

        scores = {
            item["label"]: float(item["score"])
            for item in result
        }

        emotion_risk = calculate_emotion_risk(scores)

        return {
            "emotion_scores": scores,
            "emotion_risk": emotion_risk,
        }

    def predict_toxicity(self, text: str) -> dict[str, Any]:
        """
        Возвращает вероятности токсичности для текста.
        """

        toxicity_model = self.load_toxicity_model()

        result = toxicity_model(
            text,
            truncation=True,
            max_length=512,
        )
        result = normalize_pipeline_result(result)

        scores = {
            item["label"]: float(item["score"])
            for item in result
        }

        toxicity_risk = calculate_toxicity_risk(scores)

        return {
            "toxicity_scores": scores,
            "toxicity_risk": toxicity_risk,
        }

    def predict_semantic_risk(self, text: str) -> dict[str, Any]:
        """
        Оценивает общую семантическую подозрительность текста через zero-shot classification.

        Итоговый semantic_fake_risk берется как максимум между:
        - фейковая новость;
        - манипулятивная публикация.
        """

        nli_model = self.load_nli_zero_shot_model()

        candidate_labels = [
            "фейковая новость",
            "манипулятивная публикация",
            "достоверная новость",
        ]

        result = nli_model(
            text[:2000],
            candidate_labels=candidate_labels,
            hypothesis_template="Этот текст является {}.",
        )

        labels = result["labels"]
        scores_raw = result["scores"]

        scores = {
            label: float(score)
            for label, score in zip(labels, scores_raw)
        }

        semantic_fake_risk = max(
            scores.get("фейковая новость", 0.0),
            scores.get("манипулятивная публикация", 0.0),
        )

        return {
            "semantic_scores": scores,
            "semantic_fake_risk": semantic_fake_risk,
        }

    def predict_contradiction_risk(self, headline: str, article_body: str) -> dict[str, Any]:
        """
        Оценивает противоречие между заголовком и основным текстом.

        article_body используется как premise — основной текст.
        headline используется как hypothesis — проверяемое утверждение.

        Если contradiction высокий, значит заголовок противоречит содержанию текста.
        """

        nli_pair_model = self.load_nli_pair_model()

        premise = str(article_body)[:1200]
        hypothesis = str(headline)[:300]

        result = nli_pair_model(
            {
                "text": premise,
                "text_pair": hypothesis,
            },
            truncation=True,
            max_length=512,
        )

        result = normalize_pipeline_result(result)

        scores = {
            item["label"]: float(item["score"])
            for item in result
        }

        contradiction_risk = scores.get("contradiction", 0.0)
        entailment_score = scores.get("entailment", 0.0)
        neutral_score = scores.get("neutral", 0.0)

        return {
            "nli_pair_scores": scores,
            "contradiction_risk": contradiction_risk,
            "entailment_score": entailment_score,
            "neutral_score": neutral_score,
        }

    def analyze_text(self, text: str) -> dict[str, Any]:
        """
        Запускает модели для одного цельного текста.
        Используется для CLI, когда пользователь вводит произвольную публикацию.
        """

        semantic_result = self.predict_semantic_risk(text)
        emotion_result = self.predict_emotion(text)
        toxicity_result = self.predict_toxicity(text)

        return {
            **semantic_result,
            **emotion_result,
            **toxicity_result,
        }

    def analyze_pair(self, headline: str, article_body: str) -> dict[str, Any]:
        """
        Запускает анализ для пары:
        заголовок + основной текст.

        Используется для датасета, где есть отдельно headline и article_body.
        """

        combined_text = f"{headline}. {article_body}"

        contradiction_result = self.predict_contradiction_risk(
            headline=headline,
            article_body=article_body,
        )

        semantic_result = self.predict_semantic_risk(combined_text)
        emotion_result = self.predict_emotion(combined_text)
        toxicity_result = self.predict_toxicity(combined_text)

        return {
            **contradiction_result,
            **semantic_result,
            **emotion_result,
            **toxicity_result,
        }


def normalize_pipeline_result(result: Any) -> list[dict[str, Any]]:
    """
    Приводит результат pipeline к единому виду.
    """

    if isinstance(result, list) and result and isinstance(result[0], list):
        return result[0]

    return result


def calculate_emotion_risk(scores: dict[str, float]) -> float:
    """
    Рассчитывает риск эмоционального давления.
    """

    risk_labels = [
        "fear",
        "anger",
        "surprise",
        "sadness",
    ]

    found_scores = []

    for label, score in scores.items():
        label_lower = label.lower()

        for risk_label in risk_labels:
            if risk_label in label_lower:
                found_scores.append(score)

    if not found_scores:
        return 0.0

    return max(found_scores)


def calculate_toxicity_risk(scores: dict[str, float]) -> float:
    """
    Рассчитывает риск токсичной или опасной подачи.

    Для модели cointegrated/rubert-tiny-toxicity используются классы:
    - non-toxic
    - dangerous
    - insult
    - threat
    - obscenity
    """

    risky_labels = [
        "dangerous",
        "insult",
        "threat",
        "obscenity",
    ]

    risky_scores = []

    for label, score in scores.items():
        label_lower = label.lower()

        if label_lower in risky_labels:
            risky_scores.append(score)

    if not risky_scores:
        return 0.0

    return max(risky_scores)


if __name__ == "__main__":
    test_headline = "Банк России повысил ключевую ставку."

    test_article_body = (
        "Банк России сообщил о решении снизить ключевую ставку. "
        "Информация опубликована на официальном сайте регулятора."
    )

    models = HuggingFaceModels()
    result = models.analyze_pair(
        headline=test_headline,
        article_body=test_article_body,
    )

    print()
    print("Contradiction risk:", result["contradiction_risk"])
    print("NLI pair scores:")
    for label, score in result["nli_pair_scores"].items():
        print(f"{label}: {score:.4f}")

    print()
    print("Semantic fake risk:", result["semantic_fake_risk"])
    print("Emotion risk:", result["emotion_risk"])
    print("Toxicity risk:", result["toxicity_risk"])