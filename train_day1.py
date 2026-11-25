from pathlib import Path
import re
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import text as sk_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "smartphone_reviews.csv"
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")
MAX_ROWS = 50000  # Set to None to use every row in a large CSV
TEST_SIZE = 0.20
RANDOM_STATE = 42

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEGATIONS = {
    "not",
    "no",
    "nor",
    "neither",
    "never",
    "hardly",
    "barely",
    "scarcely",
}
STOP_WORDS = sorted(set(sk_text.ENGLISH_STOP_WORDS) - NEGATIONS)

SAMPLE_REVIEWS = [
    (
        "Amazing AMOLED display and super fast charging, easily lasts a full"
        " day.",
        5,
    ),
    (
        "Phone heats up during BGMI gaming and battery drains fast on 5G.",
        2,
    ),
    (
        "Camera clicks sharp daylight photos, low light is average for the price.",
        3,
    ),
    ("Not worth the money at all, lags after 20 minutes of use.", 1),
    (
        "Great value for money! Smooth 120Hz screen and solid build quality.",
        5,
    ),
    ("Not bad overall, speakers are loud but night camera is noisy.", 3),
    ("Fingerprint sensor is slow and UI has too many bloatware ads.", 2),
    ("Battery backup is not good when using hotspot all afternoon.", 2),
    ("Super crisp photos, clean software experience, zero heating issues.", 5),
    ("Stopped charging after three weeks, worst experience ever.", 1),
    (
        "Decent everyday phone for calls and social media, nothing"
        " extraordinary.",
        3,
    ),
    ("Fast processor and bright screen, totally happy with this upgrade.", 4),
    ("Display colors pop nicely and stereo speakers sound clear.", 4),
    ("Battery drains quickly on mobile data and charger gets warm.", 2),
    ("Not impressed with the selfie camera, indoor shots look soft.", 2),
    ("Solid performance for daily multitasking and smooth scrolling.", 4),
    ("UI feels heavy and occasionally stutters when switching apps.", 2),
    ("Good budget phone overall, camera is okay in bright sunlight.", 3),
    ("No heating during normal use and battery easily goes past one day.", 5),
    ("Screen brightness outdoors is a bit low under harsh sun.", 3),
]


# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
  rename_map = {}
  text_candidates = {
      "review",
      "review_text",
      "reviewtext",
      "reviews",
      "text",
      "body",
      "reviewbody",
      "comment",
      "content",
      "customer_review",
  }
  rating_candidates = {
      "rating",
      "ratings",
      "stars",
      "star_rating",
      "score",
      "rate",
      "overall",
  }

  for col in df.columns:
    key = re.sub(r"[^a-z0-9_]", "", col.lower().strip().replace(" ", "_"))
    if (
        key in text_candidates
        and "review_text" not in rename_map.values()
        and not pd.api.types.is_numeric_dtype(df[col])
    ):
      rename_map[col] = "review_text"
    elif key in rating_candidates and "rating" not in rename_map.values():
      rename_map[col] = "rating"

  return df.rename(columns=rename_map)


def map_rating_to_sentiment(rating: float) -> str:
  if rating >= 4.0:
    return "Positive"
  if rating < 3.0:
    return "Negative"
  return "Neutral"


def clean_review_text(text: str) -> str:
  text = str(text).lower()
  text = re.sub(r"http\S+|www\.\S+|@\S+", " ", text)
  text = re.sub(r"[^a-z0-9'\s]", " ", text)
  text = re.sub(r"\s+", " ", text).strip()
  return text


def load_and_clean_data(path: Path) -> pd.DataFrame:
  if path.exists():
    print(f"[INFO] Loading dataset from {path}")
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)
  else:
    print(
        f"[WARN] {path} not found. Using built-in sample smartphone reviews"
        " so you can test immediately."
    )
    df = pd.DataFrame(SAMPLE_REVIEWS, columns=["review_text", "rating"])

  if "review_text" not in df.columns or "rating" not in df.columns:
    raise ValueError(
        "Could not find review text and rating columns automatically. "
        f"Available columns: {df.columns.tolist()}"
    )

  df = df[["review_text", "rating"]].dropna(subset=["review_text", "rating"])
  df = df.copy()

  # Extract numeric star rating if strings like "5 out of 5 stars" appear
  if not pd.api.types.is_numeric_dtype(df["rating"]):
    extracted = df["rating"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0]
    df["rating"] = pd.to_numeric(extracted, errors="coerce")
  else:
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

  df = df.dropna(subset=["rating"])
  df["review_text"] = df["review_text"].astype(str).str.strip()
  df = df[df["review_text"].str.len() > 0].drop_duplicates(
      subset=["review_text"]
  )

  if MAX_ROWS is not None and len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE)

  df["sentiment"] = df["rating"].apply(map_rating_to_sentiment)
  df["clean_text"] = df["review_text"].apply(clean_review_text)
  df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

  print(f"[INFO] Final cleaned dataset size: {len(df)} reviews")
  print("[INFO] Sentiment distribution:\n", df["sentiment"].value_counts())
  return df


# ==========================================
# 3. MODEL TRAINING & BENCHMARKING
# ==========================================
def train_and_evaluate(df: pd.DataFrame):
  min_count = df["sentiment"].value_counts().min()
  stratify_col = df["sentiment"] if min_count >= 2 else None

  X_train, X_test, y_train, y_test = train_test_split(
      df["clean_text"],
      df["sentiment"],
      test_size=TEST_SIZE,
      random_state=RANDOM_STATE,
      stratify=stratify_col,
  )

  models = {
      "Logistic Regression": LogisticRegression(
          max_iter=1000, C=2.0, random_state=RANDOM_STATE
      ),
      "Linear SVM": LinearSVC(C=1.0, random_state=RANDOM_STATE),
      "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
  }

  labels = sorted(df["sentiment"].unique())
  metrics_rows = []
  pipelines = {}

  for name, estimator in models.items():
    pipe = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words=STOP_WORDS,
                ngram_range=(1, 2),
                max_features=15000,
                sublinear_tf=True,
                min_df=1 if len(X_train) < 50 else 2,
            ),
        ),
        ("classifier", estimator),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    metrics_rows.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Weighted_F1": round(weighted_f1, 4),
        "Macro_F1": round(macro_f1, 4),
    })
    pipelines[name] = (pipe, y_pred)

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, zero_division=0))

  metrics_df = pd.DataFrame(metrics_rows).sort_values(
      by="Weighted_F1", ascending=False
  )
  metrics_path = OUT_DIR / "day1_model_metrics.csv"
  metrics_df.to_csv(metrics_path, index=False)
  print(f"[INFO] Saved model comparison metrics to {metrics_path}")

  return X_test, y_test, labels, metrics_df, pipelines


# ==========================================
# 4. ARTIFACT EXPORT & VISUALIZATION
# ==========================================
def save_outputs(X_test, y_test, labels, metrics_df, pipelines):
  best_name = metrics_df.iloc[0]["Model"]
  best_pipe, best_preds = pipelines[best_name]

  model_path = MODEL_DIR / "best_sentiment_pipeline.joblib"
  labels_path = MODEL_DIR / "sentiment_labels.joblib"
  joblib.dump(best_pipe, model_path)
  joblib.dump(labels, labels_path)
  print(f"[INFO] Best model ({best_name}) saved to {model_path}")

  # Confusion Matrix Plot
  cm = confusion_matrix(y_test, best_preds, labels=labels)
  plt.figure(figsize=(6.5, 5))
  sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap="Blues",
      xticklabels=labels,
      yticklabels=labels,
  )
  plt.title(f"Confusion Matrix - {best_name}")
  plt.xlabel("Predicted Sentiment")
  plt.ylabel("Actual Sentiment")
  plt.tight_layout()

  cm_path = OUT_DIR / "confusion_matrix.png"
  plt.savefig(cm_path, dpi=200)
  plt.close()
  print(f"[INFO] Saved confusion matrix chart to {cm_path}")

  # Misclassified Examples for Error Analysis & Viva
  errors_df = pd.DataFrame({
      "review_text": X_test.values,
      "actual_sentiment": y_test.values,
      "predicted_sentiment": best_preds,
  })
  errors_df = errors_df[
      errors_df["actual_sentiment"] != errors_df["predicted_sentiment"]
  ]
  errors_path = OUT_DIR / "sample_misclassifications.csv"
  errors_df.head(20).to_csv(errors_path, index=False)
  print(
      f"[INFO] Saved {min(len(errors_df), 20)} misclassification samples to"
      f" {errors_path}"
  )


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
  df = load_and_clean_data(DATA_PATH)
  X_test, y_test, labels, metrics_df, pipelines = train_and_evaluate(df)
  save_outputs(X_test, y_test, labels, metrics_df, pipelines)

  print("\n=== FINAL LEADERBOARD ===")
  print(metrics_df.to_string(index=False))


if __name__ == "__main__":
  main()