from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "books_clean.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "features.csv"


def get_image_features(image_path):
    if not image_path:
        return [0.0] * 6

    full_path = ROOT / image_path

    try:
        image = Image.open(full_path).convert("RGB")
        image = image.resize((64, 64))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        means = pixels.mean(axis=(0, 1))
        stds = pixels.std(axis=(0, 1))

        return [
            float(means[0]), float(means[1]), float(means[2]),
            float(stds[0]), float(stds[1]), float(stds[2])
        ]
    except:
        return [0.0] * 6


df = pd.read_csv(INPUT_FILE)
df["title"] = df["title"].fillna("").astype(str)
df["image_path"] = df["image_path"].fillna("").astype(str)
df["title_word_count"] = df["title"].apply(lambda x: len(x.split()))
df["title_char_count"] = df["title"].str.len()
df["has_colon"] = df["title"].str.contains(":", regex=False).astype(int)
df["has_question_mark"] = df["title"].str.contains("?", regex=False).astype(int)

image_features = []
for image_path in df["image_path"]:
    image_features.append(get_image_features(image_path))

image_columns = [
    "cover_r_mean", "cover_g_mean", "cover_b_mean",
    "cover_r_std", "cover_g_std", "cover_b_std",
]

img_df = pd.DataFrame(image_features, columns=image_columns)
for col in image_columns:
    df[col] = img_df[col]


df.to_csv(OUTPUT_FILE, index=False)

print("Features saved successfully to:", OUTPUT_FILE)
print(df.head())