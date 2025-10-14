
# This file contains the core logic for sentiment analysis.

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

# Use a class to encapsulate model loading and prediction
class SentimentAnalyzer:
    """
    A class to load the sentiment analysis model and perform predictions.
    """
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initializes the tokenizer and model.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            print(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
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

        # Preprocess the text
        processed_text = self.preprocess(text)

        # Tokenize the input
        encoded_input = self.tokenizer(processed_text, return_tensors='pt')

        # Get model output
        try:
            output = self.model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
        except Exception as e:
            return {"label": "Error", "score": 0.0, "detail": f"Model inference failed: {e}"}


        # Rank scores and get the top sentiment
        ranking = np.argsort(scores)
        ranking = ranking[::-1] # Reverse to get highest score first
        
        top_sentiment_index = ranking[0]
        label = self.config.id2label[top_sentiment_index]
        score = scores[top_sentiment_index]

        return {"label": label, "score": float(score)}

# Create a single instance to be used by the API
# This ensures the model is loaded only once when the application starts.
analyzer = SentimentAnalyzer()
