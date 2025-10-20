# This file contains the core logic for the sentiment analysis service.
# It defines a reusable class that encapsulates all the necessary steps:
# loading the model, preprocessing text, and performing predictions.

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

class SentimentAnalyzer:
    """A class to load the sentiment analysis model and perform predictions."""

    def __init__(self, model_name: str = "flaaa31/sentiment_model_for_hf"):

        """
        Initializes the tokenizer and the model.

        This constructor is responsible for downloading and loading the pre-trained
        RoBERTa model and its corresponding tokenizer from the Hugging Face Hub.

        Args:
            model_name (str): The identifier of the model on the Hugging Face Hub.
        """

        print(f"Loading model from Hugging Face Hub: '{model_name}'...")
        try:
            # loading tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # loading model configuration file
            self.config = AutoConfig.from_pretrained(model_name)
            # loading the pre-trained model weights for sequence classification
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            print(f"Model '{model_name}' loaded.")
        except Exception as e:
            print(f"Error in model loading: {e}")
            print("Check model name on Hugging Face Hub.")
            raise

    def preprocess(self, text: str) -> str:

        """
        Preprocesses the input text to clean it for the model.

        The RoBERTa model was trained on text where user mentions and URLs were replaced
        with generic placeholders. This function replicates that cleaning process to
        ensure that input data matches the data the model was trained on,
        improving prediction accuracy.

        Args:
            text (str): The raw input text.

        Returns:
            str: The cleaned text with placeholders.
        """

        new_text = []
        #splitting text
        for t in text.split(" "):
            # If a word starts with '@' and has more than one character, replace it with a generic '@user'.
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            
            # If a word starts with 'http', replace it with a generic 'http'.
            t = 'http' if t.startswith('http') else t
            new_text.append(t)

        # Join the words back into a single string.
        return " ".join(new_text)

    def analyze(self, text: str) -> dict:

        """
        Analyzes the sentiment of a given text string.

        It takes a raw string, preprocesses it,
        tokenizes it, feeds it to the model, and processes the output to return a
        human-readable sentiment label and confidence score.

        Args:
            text (str): The input text to analyze.

        Returns:
            dict: A dictionary containing the predicted 'label' (e.g., 'positive')
                  and the confidence 'score' (a float between 0 and 1).
        """
        # Case where the input text is empty.
        if not text:
            return {"label": "Error", "score": 0.0, "detail": "Input text cannot be empty."}

        # Cleaning text using preprocess method
        processed_text = self.preprocess(text)

        # Tokenize the cleaned text, converting it into numerical IDs and creating tensors suitable for pytorch
        encoded_input = self.tokenizer(processed_text, return_tensors='pt')

        # Perform model inference to get the raw output (logits)
        try:
            output = self.model(**encoded_input)
            # Extracting logits
            scores = output[0][0].detach().numpy()
            # Applying softmax function to convert logits into probabilities
            scores = softmax(scores)
        except Exception as e:
            return {"label": "Error", "score": 0.0, "detail": f"Model inference failed: {e}"}

        # Ranking scores and get the top sentiment
        ranking = np.argsort(scores)
        # Reverse to get highest score first
        ranking = ranking[::-1]

        #index of the highest probability     
        top_sentiment_index = ranking[0]
        # mapping index back to its string label
        label = self.config.id2label[top_sentiment_index]
        # Get the corresponding highest probability score.
        score = scores[top_sentiment_index]

        return {"label": label, "score": float(score)}

