# BERT-Sentimental-Analysis

Mein Projekt demonstriert die Anwendung eines vortrainierten BERT-basierten Sprachmodells zur automatischen Klassifikation von Amazon Produktbewertungen (oder generell Bewertungen) in positiv, neutral oder negativ.

Das Modell analysiert ausschließlich den Textinhalt einer Rezension und sagt eine entsprechende Sternebewertung (1–5) voraus, welche anschließend in eine Sentimentklasse überführt wird.

Projektüberblick:

-Verwendung eines vortrainierten Transformer-Modells
-Textklassifikation mit BERT (Encoder only Architektur)
-Evaluation von BERT mit balancierten Klassen
-Interaktive Web Anwendung mit Streamlit

Modell:
nlptown/bert-base-multilingual-uncased-sentiment von Huggingface
