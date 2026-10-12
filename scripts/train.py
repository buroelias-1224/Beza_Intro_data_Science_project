from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "features.csv"
MODEL_FOLDER = ROOT / "models"
MODEL_FILE = MODEL_FOLDER / "book_classifier.joblib"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["category", "target"])
y = df["target"]


title_steps = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=3000,
    ))
])

number_columns = [
    "price_gbp",
    "rating",
    "title_word_count",
    "title_char_count",
    "has_colon",
    "has_question_mark",
    "cover_r_mean",
    "cover_g_mean",
    "cover_b_mean",
    "cover_r_std",
    "cover_g_std",
    "cover_b_std",
]

number_steps = Pipeline([
    ("fill_missing", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

all_features = ColumnTransformer([
    ("title", title_steps, "title"),
    ("numbers", number_steps, number_columns),
])


model = Pipeline([
    ("features", all_features),
    ("classifier", LogisticRegression(
        max_iter=3000,
        random_state=42,
    )),
])


print("Training model...")
model.fit(X, y)

joblib.dump(model, MODEL_FILE)

print("Model trained on", len(df), "rows.")
print("Saved model to:", MODEL_FILE)