from pathlib import Path
import pandas as pd

# Keep the paths so it still works
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "books_raw.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "books_clean.csv"

# Load the dataset
df = pd.read_csv(RAW_FILE)

# Only keep Fiction and Nonfiction
df = df[df["category"].isin(["Fiction", "Nonfiction"])]

# Clean up text columns
df["title"] = df["title"].fillna("").astype(str).str.strip()
df["detail_url"] = df["detail_url"].fillna("").astype(str).str.strip()
df["image_url"] = df["image_url"].fillna("").astype(str).str.strip()
df["image_path"] = df["image_path"].fillna("").astype(str).str.strip()

# Convert price and rating to numbers
df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Students usually just drop missing values to keep things simple
df = df.dropna(subset=["price_gbp", "rating"])

# Remove short titles and duplicates
df = df[df["title"].str.len() > 1]
df = df.drop_duplicates(subset=["title", "category"])

# Create target column using a basic lambda function
df["target"] = df["category"].apply(lambda x: 1 if x == "Fiction" else 0)

# Save the clean file
df.to_csv(OUTPUT_FILE, index=False)

print("Clean dataset rows:", len(df))
print(df["category"].value_counts())
print("File saved successfully!")