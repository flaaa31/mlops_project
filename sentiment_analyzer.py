from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

# --- MODIFICA CHIAVE ---
# Ora carichiamo il modello direttamente dal tuo repository su Hugging Face.
# Sostituisci "flaaa31" con il tuo username se è diverso.
MODEL_PATH = "flaaa31/sentiment_model_for_hf"

class SentimentAnalyzer:
    def __init__(self, model_name: str = MODEL_PATH):
        try:
            print(f"Caricamento del modello da Hugging Face Hub: '{model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            print("Modello caricato con successo.")
        except Exception as e:
            print(f"Errore durante il caricamento del modello: {e}")
            print("Assicurati che il modello esista su Hugging Face Hub e che il nome sia corretto.")
            raise

    def preprocess(self, text: str) -> str:
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def analyze(self, text: str) -> dict:
        if not text:
            return {"label": "Error", "score": 0.0, "detail": "Input text cannot be empty."}

        processed_text = self.preprocess(text)
        encoded_input = self.tokenizer(processed_text, return_tensors='pt')

        try:
            output = self.model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
        except Exception as e:
            return {"label": "Error", "score": 0.0, "detail": f"Model inference failed: {e}"}

        ranking = np.argsort(scores)[::-1]
        top_sentiment_index = ranking[0]
        label = self.config.id2label[top_sentiment_index]
        score = scores[top_sentiment_index]

        return {"label": label, "score": float(score)}

# Crea un'istanza unica da usare nell'API.
analyzer = SentimentAnalyzer()

