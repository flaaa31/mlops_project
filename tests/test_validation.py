import sys, os
import pytest
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import torch

# Python Path Modification
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Validating the fine tuned model created by train.py
MODEL_TO_VALIDATE = "./sentiment_model_local"

# Performance treshold of 70% accuracy on test set
MINIMUM_ACCURACY_THRESHOLD = 0.70

@pytest.fixture(scope="module")
def validation_pipeline():
    """
    A pytest "fixture" to to load the fine tuned model once.
    """
    print(f"\n[Validation Fixture] Loading model {MODEL_TO_VALIDATE} from Hub...")
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-classification",
        model=MODEL_TO_VALIDATE,
        tokenizer=MODEL_TO_VALIDATE,
        device=device
    )
    print("Model loaded.")
    return pipe

@pytest.fixture(scope="module")
def test_dataset():
    """
    Fixture to load "test set" of tweet_eval.
    """
    print("[Validation Fixture] Loading 'tweet_eval' test set...")
    dataset = load_dataset("tweet_eval", "sentiment", split="test")
    
    print(f"Test set loaded with {len(dataset)} examples.")
    return dataset


def test_model_performance(validation_pipeline, test_dataset):
    """
    Testing validation accuracy
    """
    pipe = validation_pipeline
    
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    # Data preparation
    texts = list(test_dataset["text"]) 
    true_labels = [label_map[label] for label in test_dataset["label"]]

    # Predictions
    print("Running predictions on test set...")
    
    predictions_output = pipe(texts, batch_size=32)
    predicted_labels = [pred['label'].lower() for pred in predictions_output]
    print("Predictions complete.")

    # Accuracy check
    accuracy_on_test_set = accuracy_score(true_labels, predicted_labels)
    
    print(f"\n--- Model Validation Results ---")
    print(f"Accuracy on Test Set: {accuracy_on_test_set:.4f}")
    print(f"Minimum Threshold:    {MINIMUM_ACCURACY_THRESHOLD:.4f}")
    print("---------------------------------")

    # Asserting that accuracy reaches the minimum treshold
    assert accuracy_on_test_set > MINIMUM_ACCURACY_THRESHOLD, f"Model Accuracy ({accuracy_on_test_set:.4f}) is below the threshold ({MINIMUM_ACCURACY_THRESHOLD:.4f})!"