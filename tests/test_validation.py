import sys, os
import pytest
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline # Aggiungi AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import torch

# --- Python Path Modification ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Test Configuration ---
MODEL_TO_VALIDATE = "./sentiment_model_local" 
MINIMUM_ACCURACY_THRESHOLD = 0.70

@pytest.fixture(scope="module")
def validation_pipeline():
    """
    Fixture to load the fine-tuned model once.
    We load model and tokenizer from local path.
    """
    model_path = MODEL_TO_VALIDATE
    print(f"\n[Validation Fixture] Loading model and tokenizer explicitly from local path: {model_path}...")
    
    try:
        # Loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Loading model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        print("Model and tokenizer loaded successfully.")
        
        # device selection
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'cuda:0' if device == 0 else 'cpu'}")

        # Pipeline 
        pipe = pipeline(
            "text-classification",
            model=model, 
            tokenizer=tokenizer,
            device=device
        )
        return pipe
        
    except Exception as e:
        print(f"Error loading model/tokenizer from {model_path}: {e}")
        # Adding some debug if error occurs
        print(f"Contents of {model_path}:")
        try:
            print(os.listdir(model_path))
        except FileNotFoundError:
            print("Directory not found!")
        pytest.fail(f"Failed to load model from {model_path}: {e}")


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
    Validation test
    """
    pipe = validation_pipeline
    
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    texts = list(test_dataset["text"])
    true_labels = [label_map[label] for label in test_dataset["label"]]

    print("Running predictions on test set...")
    predictions_output = pipe(texts, batch_size=32)
    predicted_labels = [pred['label'].lower() for pred in predictions_output]
    print("Predictions complete.")

    accuracy_on_test_set = accuracy_score(true_labels, predicted_labels)
    
    print(f"\n--- Model Validation Results ---")
    print(f"Accuracy on Test Set: {accuracy_on_test_set:.4f}")
    print(f"Minimum Threshold:    {MINIMUM_ACCURACY_THRESHOLD:.4f}")
    print("---------------------------------")

    assert accuracy_on_test_set > MINIMUM_ACCURACY_THRESHOLD, f"Model Accuracy ({accuracy_on_test_set:.4f}) is below the threshold ({MINIMUM_ACCURACY_THRESHOLD:.4f})!"