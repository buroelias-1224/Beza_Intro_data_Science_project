from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "book_classifier.joblib"
DATA_PATH = ROOT / "data" / "processed" / "features.csv"

st.set_page_config(
    page_title="Book Detective",
    page_icon="📚",
    layout="wide",
)


st.title("📚 Book Detective Classifier")
st.write("Can we uncover whether a book is **Fiction** or **Nonfiction** using machine learning?")
st.markdown("---")

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("Model or dataset not found! Please run your python scripts first to generate them.")
    st.stop()


model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

st.sidebar.header("Prediction Controls")
title = st.text_input("Book Title", placeholder="Example: The Secret Garden")

col1, col2, col3 = st.columns(3)

with col1:
    price = st.number_input("Price (£)", min_value=0.0, value=20.0, step=0.5)

with col2:
    rating = st.slider("Star Rating", 1, 5, 3)

with col3:
    cover_color = st.selectbox(
        "Dominant Cover Color",
        ["Neutral", "Warm", "Cool"],
    )

color_values = {
    "Neutral": (0.50, 0.50, 0.50),
    "Warm": (0.75, 0.45, 0.25),
    "Cool": (0.30, 0.45, 0.75),
}

r, g, b = color_values[cover_color]

if st.button("Predict Category", type="primary"):
    if not title.strip():
        st.warning("Please enter a book title first.")
    else:
        word_count = len(title.split())
        char_count = len(title)

        record = pd.DataFrame([{
            "title": title,
            "price_gbp": price,
            "rating": rating,
            "title_word_count": word_count,
            "title_char_count": char_count,
            "has_colon": int(":" in title),
            "has_question_mark": int("?" in title),
            "cover_r_mean": r,
            "cover_g_mean": g,
            "cover_b_mean": b,
            "cover_r_std": 0.15,
            "cover_g_std": 0.15,
            "cover_b_std": 0.15,
        }])

        prediction = int(model.predict(record)[0])
        probabilities = model.predict_proba(record)[0]

        label = "Fiction" if prediction == 1 else "Nonfiction"
        confidence = float(probabilities[prediction])

        if label == "Fiction":
            st.success(f"Result: **{label}**")
        else:
            st.info(f"Result: **{label}**")

        st.metric("Confidence Score", f"{confidence:.1%}")
        st.progress(confidence)

        
        st.write("---")
        st.subheader("Input Summary")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Title Words", word_count)
        with m2:
            st.metric("Price", f"£{price:.2f}")
        with m3:
            st.metric("Rating", f"{rating}/5")

st.markdown("---")


st.subheader("Dataset Overview")

left, right = st.columns(2)

with left:
    st.write("First 10 Rows of Cleaned Data")
    st.dataframe(
        df[["title", "category", "price_gbp", "rating"]].head(10),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.write("Category Distribution")
    counts = df["category"].value_counts()
    st.bar_chart(counts)

st.caption("Dashboard built for machine learning project demonstration.")