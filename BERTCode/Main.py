from datasets import get_dataset_config_names
from datasets import load_dataset
from transformers import pipeline
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import streamlit as st
from transformers import pipeline


print(get_dataset_config_names("buruzaemon/amazon_reviews_multi"))

ds = load_dataset("buruzaemon/amazon_reviews_multi", "de", split="test")

def stars_to_sentiment(stars):
    if stars >= 4:
        return "Positiv"
    elif stars == 3:
        return "Neutral"
    else:
        return "Negativ"

sentiment_model = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)


st.set_page_config(
    page_title="Amazon Review Sentiment (BERT)",
    page_icon="⭐",
    layout="centered"
)

st.title("⭐ Amazon-Review Klassifikation (BERT)")
st.caption("Text rein → Sterne + Stimmung raus (ohne manuelles Lesen)")


@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

sentiment_model = load_model()


def stars_to_sentiment(stars: int) -> str:
    if stars >= 4:
        return "Positiv"
    elif stars == 3:
        return "Neutral"
    return "Negativ"

def sentiment_style(sentiment: str):
    # returns (emoji, streamlit color keyword)
    if sentiment == "Positiv":
        return "✅", "success"
    if sentiment == "Neutral":
        return "🟨", "warning"
    return "❌", "error"

def parse_stars(label: str) -> int:
    # label like: "4 stars" or "1 star"
    return int(label.split()[0])

def stars_bar(stars: int) -> str:
    # Visual stars: ★★★★☆
    return "★" * stars + "☆" * (5 - stars)


# UI

with st.container(border=True):
    st.subheader("📝 Rezension eingeben")
    review = st.text_area(
        "Text (z.B. eine Amazon-Bewertung)",
        placeholder="Beispiel: 'Qualität top, Lieferung schnell. Bin sehr zufrieden!'",
        height=140
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        analyze = st.button("Analysieren", type="primary", use_container_width=True)
    with col2:
        st.button("Zurücksetzen", use_container_width=True, on_click=lambda: st.session_state.update({"_clear": True}))

# Optional: Clear textarea (simple approach)
if st.session_state.get("_clear"):
    st.session_state["_clear"] = False
    st.rerun()

st.divider()


# Run inference

if analyze:
    text = (review or "").strip()
    if len(text) < 5:
        st.info("Bitte gib eine etwas längere Rezension ein (mind. ~5 Zeichen).")
    else:
        # Truncation to avoid super long input
        out = sentiment_model(text[:512], truncation=True)[0]
        pred_stars = parse_stars(out["label"])
        pred_sentiment = stars_to_sentiment(pred_stars)
        score = float(out["score"])

        emoji, level = sentiment_style(pred_sentiment)

        # Nice result card
        st.subheader("📌 Ergebnis")
        with st.container(border=True):
            # Top row metrics
            c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
            c1.metric("Geschätzte Sterne", f"{pred_stars}/5")
            c2.metric("Stimmung", f"{emoji} {pred_sentiment}")
            c3.metric("Sicherheit", f"{score:.2f}")

            st.write("**Sternanzeige:**", stars_bar(pred_stars))
            st.progress(min(max(score, 0.0), 1.0))

            # Color-coded message
            if level == "success":
                st.success(f"Das klingt insgesamt **positiv** (≈ {pred_stars} Sterne).")
            elif level == "warning":
                st.warning(f"Das klingt eher **neutral / gemischt** (≈ {pred_stars} Sterne).")
            else:
                st.error(f"Das klingt insgesamt **negativ** (≈ {pred_stars} Sterne).")



# Footer
st.caption("Modell: nlptown/bert-base-multilingual-uncased-sentiment (BERT).")