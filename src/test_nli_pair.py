from transformers import pipeline

from config import MODEL_NLI


def main():
    print("Загружаю NLI-модель для проверки пары текстов...")

    nli = pipeline(
        task="text-classification",
        model=MODEL_NLI,
        top_k=None,
    )

    premise = (
        "Банк России сообщил о решении снизить ключевую ставку. "
        "Информация опубликована на официальном сайте регулятора."
    )

    hypothesis_agree = "Банк России снизил ключевую ставку."

    hypothesis_disagree = "Банк России повысил ключевую ставку."

    print()
    print("ПРИМЕР 1: согласованный заголовок")
    result_agree = nli(
        {
            "text": premise,
            "text_pair": hypothesis_agree,
        }
    )
    print(result_agree)

    print()
    print("ПРИМЕР 2: противоречащий заголовок")
    result_disagree = nli(
        {
            "text": premise,
            "text_pair": hypothesis_disagree,
        }
    )
    print(result_disagree)


if __name__ == "__main__":
    main()