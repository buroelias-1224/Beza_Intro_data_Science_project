# Book Detective

This is a small machine learning project about books.

The main goal is to predict if a book is **Fiction** or **Nonfiction** using
information scraped from [Books to Scrape](https://books.toscrape.com/).

## Note

This project is meant for learning. If you use it for coursework, make sure you
understand the code and write your own explanation in the final report.

## What the project does

The project follows these steps:

1. Scrape Fiction and Nonfiction books.
2. Save the book data and cover images.
3. Clean the data.
4. Create extra features from the title and cover image.
5. Train a machine learning model.
6. Evaluate the model.
7. Use a Streamlit app to test predictions.

## Project folders

```text
book_detective_project/
|-- app/
|   |-- streamlit_app.py
|-- data/
|   |-- raw/
|   |-- images/
|   |-- processed/
|-- models/
|-- notebooks/
|-- report/
|-- scripts/
|   |-- scrape.py
|   |-- clean.py
|   |-- feature_engineering.py
|   |-- train.py
|   |-- evaluate.py
|-- requirements.txt
|-- README.md
```

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

## How to run the project

Run the scripts in this order:

```bash
python scripts/scrape.py
python scripts/clean.py
python scripts/feature_engineering.py
python scripts/train.py
python scripts/evaluate.py
```

Then start the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## What each script does

`scrape.py`

Gets book titles, prices, ratings, URLs, and cover images from the website.
The raw data is saved in `data/raw/books_raw.csv`.

`clean.py`

Cleans the raw data, removes duplicate rows, fills missing numbers, and creates
the target column.

`feature_engineering.py`

Adds extra columns such as title word count, title length, punctuation clues,
and simple colour values from the book covers.

`train.py`

Trains a Logistic Regression model using title text and numeric features. The
trained model is saved in the `models` folder.

`evaluate.py`

Checks the model performance using a train/test split and cross validation. It
also saves a confusion matrix and metrics in the `report` folder.

## Features used by the model

The model uses:

- book title text
- price
- star rating
- title word count
- title character count
- whether the title has a colon
- whether the title has a question mark
- basic cover colour features

## Streamlit app

The app lets a user type in a book title, price, rating, and rough cover colour.
It then predicts if the book is Fiction or Nonfiction.

## Possible improvements

Some improvements that could be tried later:

- compare Logistic Regression with Random Forest
- try using only title features
- add more book categories
- use more image features from the cover
- improve the Streamlit app layout
