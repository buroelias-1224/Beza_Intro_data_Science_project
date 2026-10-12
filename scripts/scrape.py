from pathlib import Path
from urllib.parse import urljoin
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
CATEGORIES = {
    "Fiction": urljoin(BASE_URL, "catalogue/category/books/fiction_10/index.html"),
    "Nonfiction": urljoin(BASE_URL, "catalogue/category/books/nonfiction_13/index.html"),
}

ROOT = Path(__file__).resolve().parents[1]
RAW_FOLDER = ROOT / "data" / "raw"
IMAGE_FOLDER = ROOT / "data" / "images"
OUTPUT_FILE = RAW_FOLDER / "books_raw.csv"

RAW_FOLDER.mkdir(parents=True, exist_ok=True)
IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)


def get_page(url):
    response = requests.get(url)
    return BeautifulSoup(response.text, "html.parser")


def get_rating(rating_tag):
    if rating_tag is None:
        return None

    rating_numbers = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    classes = rating_tag.get("class", [])

    for name, val in rating_numbers.items():
        if name in classes:
            return val
    return None


def get_price(price_text):
    if not price_text:
        return None

    # Simple string cleanups for the pound sign
    price_text = price_text.replace("GBP", "").replace("£", "").replace("Â£", "").strip()
    return float(price_text)



all_books = []

for category, start_url in CATEGORIES.items():
    next_url = start_url

    while next_url:
        print("Scraping category:", category, "->", next_url)
        soup = get_page(next_url)
        products = soup.select("article.product_pod")

        for product in products:
            link = product.select_one("h3 a")
            image = product.select_one("img")
            price = product.select_one(".price_color")
            rating = product.select_one("p.star-rating")

            if link is None or image is None:
                continue

            title = link.get("title") or link.get_text(strip=True)

            book = {
                "title": title,
                "category": category,
                "detail_url": urljoin(next_url, link.get("href", "")),
                "image_url": urljoin(next_url, image.get("src", "")),
                "price_gbp": get_price(price.get_text(strip=True)) if price else None,
                "rating": get_rating(rating),
            }
            all_books.append(book)

        # Check for next page
        next_button = soup.select_one("li.next a")
        if next_button:
            next_url = urljoin(next_url, next_button.get("href", ""))
        else:
            next_url = None

        time.sleep(0.1)


df = pd.DataFrame(all_books)
df = df.drop_duplicates(subset=["title", "category"]).reset_index(drop=True)


df["image_path"] = ""

for index, row in df.iterrows():
    filename = f"{index:04d}.jpg"
    file_path = IMAGE_FOLDER / filename

    if file_path.exists():
        df.at[index, "image_path"] = str(file_path.relative_to(ROOT))
        continue

    try:
        res = requests.get(row["image_url"])
        file_path.write_bytes(res.content)
        df.at[index, "image_path"] = str(file_path.relative_to(ROOT))
    except Exception as e:
        print("Failed to download image:", e)
        df.at[index, "image_path"] = ""


df.to_csv(OUTPUT_FILE, index=False)

print("Scraping finished! Total books:", len(df))
print(df["category"].value_counts())
print("Saved to:", OUTPUT_FILE)