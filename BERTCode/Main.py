from datasets import get_dataset_config_names
from datasets import load_dataset
from transformers import pipeline
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import streamlit as st
from transformers import pipeline

#Dataset wird geladen
print(get_dataset_config_names("buruzaemon/amazon_reviews_multi"))

ds = load_dataset("buruzaemon/amazon_reviews_multi", "de", split="test") # deutscher Split



#initialisierung von BERT
sentiment_model = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

#Streamlit Einstellungen Kopfteil
st.set_page_config(
    page_title="Amazon Review Sentiment (BERT)",
    page_icon="⭐",
    layout="centered"
)

st.title("⭐ Amazon-Review Klassifikation (BERT)")
st.caption("Text rein → Sterne + Stimmung raus (ohne manuelles Lesen)")

# Modell Caching in Streamlit für die Performance (damit BERT nicht jedesmal neu geladen wird)
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
# Ich überschreibe sentiment_model absichtlich mit der gecachten Variante
# damit das Programm nicht bei jeder Interaktion neu initialisiert.
sentiment_model = load_model()

#Ab wie vielen Sternen welche Stimmung ist
def stars_to_sentiment(stars: int) -> str:
    if stars >= 4:
        return "Positiv"
    elif stars == 3:
        return "Neutral"
    return "Negativ"

#Emojis für Ergebnis
def sentiment_style(sentiment: str):
    # returns (emoji, streamlit color keyword)
    if sentiment == "Positiv":
        return "✅", "success"
    if sentiment == "Neutral":
        return "🟨", "warning"
    return "❌", "error"

def parse_stars(label: str) -> int: #Extrahiert Sternezahl aus dem Modell
    return int(label.split()[0])

def stars_bar(stars: int) -> str: #visuelle Sterneanzeige
    return "★" * stars + "☆" * (5 - stars)


# UI LEET CODE

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
#LEETCODE ENDE

# Text -> Sterne + Stimmung + Score

if analyze:
    text = (review or "").strip()
    if len(text) < 5:
        st.info("Bitte gib eine etwas längere Rezension ein (mind. 5 Zeichen).")
    else:
        # Das Modell arbeitet stabiler wenn sehr lange Texte gekürzt werden
        # truncation=True sorgt zusätzlich dafür, dass Token-Limits nicht überschritten werden.
        #Output des Modells
        out = sentiment_model(text[:512], truncation=True)[0]
        pred_stars = parse_stars(out["label"])
        pred_sentiment = stars_to_sentiment(pred_stars)
        score = float(out["score"])

        # Darstellungselemente für UI

        emoji, level = sentiment_style(pred_sentiment)

        # Ergebnisanzeige
        st.subheader("📌 Ergebnis")
        with st.container(border=True):
            # Ergebnis Anzeige Metriken
            c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
            c1.metric("Geschätzte Sterne", f"{pred_stars}/5")
            c2.metric("Stimmung", f"{emoji} {pred_sentiment}")
            c3.metric("Sicherheit", f"{score:.2f}")

            st.write("**Sternanzeige:**", stars_bar(pred_stars))
            st.progress(min(max(score, 0.0), 1.0))

            # Farben
            if level == "success":
                st.success(f"Das klingt insgesamt **positiv** (≈ {pred_stars} Sterne).")
            elif level == "warning":
                st.warning(f"Das klingt eher **neutral / gemischt** (≈ {pred_stars} Sterne).")
            else:
                st.error(f"Das klingt insgesamt **negativ** (≈ {pred_stars} Sterne).")



