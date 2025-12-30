from datasets import get_dataset_config_names
from datasets import load_dataset
from transformers import pipeline
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import streamlit as st
from transformers import pipeline

ds = load_dataset("buruzaemon/amazon_reviews_multi", "de", split="test")

# Indizes nach Klassen sammeln
class_indices = defaultdict(list)

#BERT
sentiment_model = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment")
def stars_to_sentiment(stars):
    if stars >= 4:
        return "Positiv"
    elif stars == 3:
        return "Neutral"
    else:
        return "Negativ"

for i in range(len(ds)):
    label = stars_to_sentiment(ds[i]["stars"])
    class_indices[label].append(i)

# gleiche Anzahl an Daten pro Klasse
k = 600
indices = (
    class_indices["Negativ"][:k] +
    class_indices["Neutral"][:k] +
    class_indices["Positiv"][:k]
)

# Daten bauen
texts = [ds[i]["review_body"] for i in indices]
true_labels = [stars_to_sentiment(ds[i]["stars"]) for i in indices]

pred_labels = []
for text in texts:
    out = sentiment_model(text[:512])[0]
    predicted_stars = int(out["label"].split()[0])
    pred_labels.append(stars_to_sentiment(predicted_stars))

# Accuracy wird angezeigt mit Confusion Matrix / evaluierung
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(true_labels, pred_labels)
print("Balanced Accuracy:", accuracy)

cm = confusion_matrix(
    true_labels,
    pred_labels,
    labels=["Negativ", "Neutral", "Positiv"]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negativ", "Neutral", "Positiv"]
)

disp.plot(cmap="Blues")
plt.show()