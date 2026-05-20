from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"

RAW_DATA_PATH = RAW_DATA_DIR / "news.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "news_clean.csv"

PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"

MODEL_NLI = "cointegrated/rubert-base-cased-nli-threeway"
MODEL_EMOTION = "cointegrated/rubert-tiny2-cedr-emotion-detection"
MODEL_TOXICITY = "cointegrated/rubert-tiny-toxicity"

SCORING_WEIGHTS = {
    "semantic_fake_risk": 0.45,
    "emotion_risk": 0.20,
    "toxicity_risk": 0.15,
    "rule_score": 0.20,
}

LOW_RISK_THRESHOLD = 0.40
HIGH_RISK_THRESHOLD = 0.70

PAIR_SCORING_WEIGHTS = {
    "contradiction_risk": 0.50,
    "semantic_fake_risk": 0.20,
    "emotion_risk": 0.10,
    "toxicity_risk": 0.10,
    "rule_score": 0.10,
}