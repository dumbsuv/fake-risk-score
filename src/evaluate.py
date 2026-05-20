import argparse
import json
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from config import PROCESSED_DATA_PATH, PREDICTIONS_PATH, METRICS_PATH
from scoring import FakeRiskScorer


def load_dataset(limit: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """
    Загружает подготовленный датасет.

    Если указан limit, берется сбалансированная выборка:
    половина label=0 и половина label=1.
    """

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Файл {PROCESSED_DATA_PATH} не найден. "
            f"Сначала запусти: python src/data_loader.py"
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"В датасете должны быть колонки {required_columns}. "
            f"Найдены колонки: {df.columns.tolist()}"
        )

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    if limit is not None and limit > 0 and limit < len(df):
        per_class = max(limit // 2, 1)

        sampled_parts = []

        for label_value in sorted(df["label"].unique()):
            part = df[df["label"] == label_value]

            sample_size = min(per_class, len(part))

            sampled_parts.append(
                part.sample(
                    n=sample_size,
                    random_state=random_state
                )
            )

        df = pd.concat(sampled_parts)
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df.reset_index(drop=True)


def predict_dataset(
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Прогоняет тексты через скоринговый инструмент.

    threshold:
    если fake_score >= threshold, считаем prediction = 1;
    иначе prediction = 0.
    """

    scorer = FakeRiskScorer()

    rows: list[dict[str, Any]] = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Оценка текстов"):
        text = str(row["text"])
        true_label = int(row["label"])

        headline = str(row.get("headline", ""))
        article_body = str(row.get("article_body", ""))

        result = scorer.analyze(
            text=text,
            headline=headline,
            article_body=article_body,
        )

        fake_score = float(result["fake_score"])
        predicted_label = 1 if fake_score >= threshold else 0

        rows.append(
            {
                "index": index,
                "body_id": row.get("body_id", ""),
                "headline": row.get("headline", ""),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "fake_score": fake_score,
                "risk_level": result["risk_level"],
                "semantic_fake_risk": result["model_scores"]["semantic_fake_risk"],
                "emotion_risk": result["model_scores"]["emotion_risk"],
                "toxicity_risk": result["model_scores"]["toxicity_risk"],
                "contradiction_risk": result["model_scores"].get("contradiction_risk", ""),
                "rule_score": result["rule_score"],
                "triggered_rules": "; ".join(result["triggered_rules"]),
                "recommendation": result["recommendation"],
            }
        )

    predictions_df = pd.DataFrame(rows)

    return predictions_df


def calculate_metrics(predictions_df: pd.DataFrame) -> dict[str, Any]:
    """
    Рассчитывает метрики качества.
    """

    y_true = predictions_df["true_label"].astype(int)
    y_pred = predictions_df["predicted_label"].astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    matrix = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix,
        "total_items": int(len(predictions_df)),
        "positive_true": int(y_true.sum()),
        "positive_predicted": int(y_pred.sum()),
    }

    return metrics


def save_results(predictions_df: pd.DataFrame, metrics: dict[str, Any]) -> None:
    """
    Сохраняет предсказания и метрики в папку outputs.
    """

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)

    print()
    print("Результаты сохранены:")
    print(f"- Предсказания: {PREDICTIONS_PATH}")
    print(f"- Метрики: {METRICS_PATH}")


def print_metrics(metrics: dict[str, Any]) -> None:
    """
    Печатает метрики в консоль.
    """

    print()
    print("=" * 80)
    print("МЕТРИКИ КАЧЕСТВА")
    print("=" * 80)

    print(f"Количество текстов: {metrics['total_items']}")
    print(f"Истинных label=1: {metrics['positive_true']}")
    print(f"Предсказанных label=1: {metrics['positive_predicted']}")
    print()

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")

    print()
    print("Confusion matrix:")
    print("[[TN, FP],")
    print(" [FN, TP]]")
    print(metrics["confusion_matrix"])

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Оценка качества скорингового прототипа на подготовленном датасете."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Сколько текстов взять для оценки. По умолчанию 20.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Порог fake_score для перевода в label=1. По умолчанию 0.5.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Фиксация случайной выборки.",
    )

    args = parser.parse_args()

    print("Загружаю датасет...")
    df = load_dataset(limit=args.limit, random_state=args.random_state)

    print(f"Размер выборки: {len(df)}")
    print("Распределение label:")
    print(df["label"].value_counts())

    print()
    print(f"Порог классификации: fake_score >= {args.threshold} → label=1")

    predictions_df = predict_dataset(
        df=df,
        threshold=args.threshold,
    )

    metrics = calculate_metrics(predictions_df)

    save_results(predictions_df, metrics)
    print_metrics(metrics)


if __name__ == "__main__":
    main()