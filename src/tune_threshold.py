import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import PREDICTIONS_PATH


def calculate_metrics_for_threshold(df: pd.DataFrame, threshold: float) -> dict:
    y_true = df["true_label"].astype(int)
    y_pred = (df["fake_score"] >= threshold).astype(int)

    return {
        "threshold": threshold,
        "predicted_positive": int(y_pred.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    if not PREDICTIONS_PATH.exists():
        print(f"Файл {PREDICTIONS_PATH} не найден.")
        print("Сначала запусти: python src/evaluate.py --limit 20")
        return

    df = pd.read_csv(PREDICTIONS_PATH)

    print("Файл с предсказаниями загружен.")
    print(f"Количество строк: {len(df)}")
    print()

    print("Статистика fake_score:")
    print(df["fake_score"].describe())
    print()

    print("Средний fake_score по истинным классам:")
    print(df.groupby("true_label")["fake_score"].mean())
    print()

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]

    rows = []

    for threshold in thresholds:
        rows.append(calculate_metrics_for_threshold(df, threshold))

    result_df = pd.DataFrame(rows)

    print("Метрики при разных порогах:")
    print(result_df.to_string(index=False))

    best_row = result_df.sort_values("f1_score", ascending=False).iloc[0]

    print()
    print("Лучший порог по F1-score:")
    print(best_row.to_string())


if __name__ == "__main__":
    main()