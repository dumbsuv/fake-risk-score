from pathlib import Path

import pandas as pd

from config import RAW_DATA_DIR


def read_csv_safely(file_path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1251"]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as error:
            last_error = error

    raise RuntimeError(f"Не удалось прочитать файл {file_path.name}. Ошибка: {last_error}")


def main():
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        print("CSV-файлы не найдены.")
        print(f"Папка: {RAW_DATA_DIR}")
        return

    print(f"Найдено CSV-файлов: {len(csv_files)}")
    print()

    for file_path in csv_files:
        print("=" * 100)
        print(f"Файл: {file_path.name}")
        print("=" * 100)

        df = read_csv_safely(file_path)

        print(f"Размер: {df.shape[0]} строк, {df.shape[1]} колонок")
        print()

        print("Колонки:")
        for column in df.columns:
            print(f"- {column}")

        print()
        print("Первые 3 строки:")
        print(df.head(3))

        print()
        print("Пропуски по колонкам:")
        print(df.isna().sum())

        print()
        print()


if __name__ == "__main__":
    main()