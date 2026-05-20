import argparse
from pathlib import Path

from scoring import FakeRiskScorer


def read_text_from_file(file_path: str) -> str:
    """
    Читает текст из файла.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    return path.read_text(encoding="utf-8")


def print_analysis_result(result: dict) -> None:
    """
    Красиво выводит результат анализа в консоль.
    """

    print()
    print("=" * 80)
    print("РЕЗУЛЬТАТ АНАЛИЗА")
    print("=" * 80)

    print(f"Итоговый fake_score: {result['fake_score']:.4f}")
    print(f"Уровень риска: {result['risk_level']}")
    print()

    print("Рекомендация:")
    print(result["recommendation"])

    print()
    print("-" * 80)
    print("Оценки моделей")
    print("-" * 80)

    model_scores = result["model_scores"]

    print(f"Семантический риск: {model_scores['semantic_fake_risk']:.4f}")
    print(f"Эмоциональный риск: {model_scores['emotion_risk']:.4f}")
    print(f"Риск токсичной/опасной подачи: {model_scores['toxicity_risk']:.4f}")

    if "contradiction_risk" in model_scores:
        print(f"Риск противоречия заголовка и текста: {model_scores['contradiction_risk']:.4f}")

    print()
    print("-" * 80)
    print("Правиловый модуль")
    print("-" * 80)

    print(f"Rule score: {result['rule_score']:.4f}")

    if result["triggered_rules"]:
        print("Сработавшие правила:")
        for rule in result["triggered_rules"]:
            print(f"- {rule}")
    else:
        print("Сработавших правил нет.")

    semantic_scores = result["details"].get("semantic_scores", {})
    if semantic_scores:
        print()
        print("-" * 80)
        print("Подробности zero-shot/NLI оценки текста")
        print("-" * 80)

        for label, score in semantic_scores.items():
            print(f"{label}: {score:.4f}")

    nli_pair_scores = result["details"].get("nli_pair_scores", {})
    if nli_pair_scores:
        print()
        print("-" * 80)
        print("Проверка согласованности заголовка и текста")
        print("-" * 80)

        for label, score in nli_pair_scores.items():
            print(f"{label}: {score:.4f}")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Скоринговый анализ риска фейковости русскоязычного текста."
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Текст публикации или новости для анализа.",
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Путь к текстовому файлу для анализа.",
    )

    parser.add_argument(
        "--headline",
        type=str,
        help="Заголовок новости для анализа пары заголовок/текст.",
    )

    parser.add_argument(
        "--body",
        type=str,
        help="Основной текст новости для анализа пары заголовок/текст.",
    )

    args = parser.parse_args()

    text_mode = bool(args.text or args.file)
    pair_mode = bool(args.headline and args.body)

    if not text_mode and not pair_mode:
        print("Ошибка: нужно передать либо --text, либо --file, либо пару --headline и --body.")
        print()
        print("Примеры:")
        print('python src/cli.py --text "Срочно! СМИ скрывают правду..."')
        print('python src/cli.py --headline "Банк России повысил ставку" --body "Банк России сообщил о снижении ставки."')
        return

    if text_mode and pair_mode:
        print("Ошибка: используйте либо режим одного текста, либо режим пары заголовок/текст.")
        return

    scorer = FakeRiskScorer()

    if pair_mode:
        headline = args.headline
        article_body = args.body
        text = f"{headline}. {article_body}"

        result = scorer.analyze(
            text=text,
            headline=headline,
            article_body=article_body,
        )

    else:
        if args.file:
            text = read_text_from_file(args.file)
        else:
            text = args.text

        result = scorer.analyze(text=text)

    print_analysis_result(result)


if __name__ == "__main__":
    main()