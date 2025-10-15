# This file contains the core logic for sentiment analysis.
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

# Recuperiamo le variabili d'ambiente. Se non esistono, usiamo valori di default.
HF_USERNAME = os.getenv("HF_USERNAME", "cardiffnlp")
REPO_NAME = os.getenv("HF_USERNAME", "twitter-roberta-base-sentiment-latest")

class SentimentAnalyzer:
    """
    A class to load a sentiment analysis model and perform predictions.
    The model can be loaded from a local path or the Hugging Face Hub.
    """
    def __init__(self, model_name: str):
        """
        Initializes the tokenizer and model from the given model_name.
        """
        self.model_name = model_name
        print(f"Attempting to load model: '{self.model_name}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            print(f"Model '{self.model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model '{self.model_name}': {e}")
            print("Please ensure the model exists and you have the correct access rights.")
            raise

    def preprocess(self, text: str) -> str:
        """
        Preprocesses the text by replacing user mentions and URLs.
        """
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def analyze(self, text: str) -> dict:
        """
        Analyzes the sentiment of a given text.

        Args:
            text (str): The input text to analyze.

        Returns:
            dict: A dictionary containing the label and score of the most likely sentiment.
        """
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

# NOTA: Abbiamo rimosso l'istanziazione globale 'analyzer = SentimentAnalyzer()'
# da questo file. La creazione dell'istanza ora avviene in main.py.
