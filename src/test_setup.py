from config import (
    BASE_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUTS_DIR,
    MODEL_NLI,
    MODEL_EMOTION,
    MODEL_TOXICITY,
)


def main():
    print("Проект запущен успешно.")
    print(f"Корневая папка проекта: {BASE_DIR}")
    print(f"Папка data: {DATA_DIR}")
    print(f"Папка raw: {RAW_DATA_DIR}")
    print(f"Папка processed: {PROCESSED_DATA_DIR}")
    print(f"Папка outputs: {OUTPUTS_DIR}")
    print()
    print("Выбранные модели Hugging Face:")
    print(f"1. Семантическая модель: {MODEL_NLI}")
    print(f"2. Модель эмоций: {MODEL_EMOTION}")
    print(f"3. Модель токсичности: {MODEL_TOXICITY}")


if __name__ == "__main__":
    main()