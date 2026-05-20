import pandas as pd

from config import RAW_DATA_DIR, PROCESSED_DATA_PATH


def load_raw_dataset() -> pd.DataFrame:
    """
    Загружает исходные файлы датасета и объединяет заголовки с текстами новостей.

    train_stances.csv содержит:
    - Body ID
    - Headline
    - Stance
    - Headline1
    - Headline2

    train_bodies.csv содержит:
    - Body ID
    - articleBody
    - articleBody1
    - articleBody2
    """

    bodies_path = RAW_DATA_DIR / "train_bodies.csv"
    stances_path = RAW_DATA_DIR / "train_stances.csv"

    bodies_df = pd.read_csv(bodies_path)
    stances_df = pd.read_csv(stances_path)

    merged_df = stances_df.merge(
        bodies_df,
        on="Body ID",
        how="inner"
    )

    return merged_df


def map_stance_to_label(stance: str) -> int:
    """
    Переводит исходную метку Stance в бинарную метку.

    agree:
    Заголовок и текст новости согласованы.
    В рамках прототипа это считается менее подозрительным случаем.
    label = 0.

    disagree:
    Заголовок и текст новости противоречат друг другу.
    В рамках прототипа это считается потенциально подозрительным случаем.
    label = 1.

    Важно:
    label = 1 не означает, что текст обязательно является фейком.
    Это означает наличие признака несогласованности, который используется
    для учебной апробации скорингового инструмента.
    """

    stance_normalized = stance.strip().lower()

    if stance_normalized == "disagree":
        return 1

    if stance_normalized == "agree":
        return 0

    raise ValueError(f"Неизвестное значение Stance: {stance}")


def prepare_dataset() -> pd.DataFrame:
    """
    Готовит датасет для дальнейшей обработки.

    В итоговой таблице будут колонки:
    - body_id
    - headline
    - article_body
    - text
    - stance
    - label

    Для текста используются колонки Headline1 и articleBody1,
    потому что они представлены как обычный очищенный русский текст.
    """

    df = load_raw_dataset()

    prepared_df = pd.DataFrame()

    prepared_df["body_id"] = df["Body ID"]

    # Используем очищенные текстовые колонки, а не исходные списки токенов.
    prepared_df["headline"] = df["Headline1"].astype(str)
    prepared_df["article_body"] = df["articleBody1"].astype(str)

    prepared_df["text"] = (
        prepared_df["headline"]
        + ". "
        + prepared_df["article_body"]
    )

    prepared_df["stance"] = df["Stance"].astype(str)
    prepared_df["label"] = prepared_df["stance"].apply(map_stance_to_label)

    prepared_df = prepared_df.dropna(subset=["headline", "article_body", "text", "label"])
    prepared_df = prepared_df.drop_duplicates(subset=["text"])

    return prepared_df


def save_prepared_dataset() -> pd.DataFrame:
    prepared_df = prepare_dataset()

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared_df.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")

    print("Подготовленный датасет сохранен.")
    print(f"Путь: {PROCESSED_DATA_PATH}")
    print(f"Размер: {prepared_df.shape[0]} строк, {prepared_df.shape[1]} колонок")
    print()

    print("Колонки:")
    for column in prepared_df.columns:
        print(f"- {column}")

    print()
    print("Распределение исходных Stance:")
    print(prepared_df["stance"].value_counts())

    print()
    print("Распределение бинарных label:")
    print(prepared_df["label"].value_counts())

    print()
    print("Пример подготовленного текста:")
    example = prepared_df.iloc[0]
    print(f"Headline: {example['headline']}")
    print()
    print(f"Article body начало: {example['article_body'][:500]}...")
    print()
    print(f"Label: {example['label']}")
    print(f"Stance: {example['stance']}")

    return prepared_df


if __name__ == "__main__":
    save_prepared_dataset()