import sys, os
import pytest
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import torch

# --- Python Path Modification ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Test Configuration ---

# 1. IL MODELLO DA VALIDARE:
# Non usiamo il modello base, ma il TUO modello fine-tuned
MODEL_TO_VALIDATE = "flaaa31/sentiment_model_for_hf"

# 2. SOGLIA DI PERFORMANCE:
# Stabiliamo che il modello, per essere "accettato", deve avere
# almeno il 70% di accuratezza sul test set.
# I tuoi risultati precedenti erano ~74%, quindi 70% è una rete di sicurezza ragionevole.
MINIMUM_ACCURACY_THRESHOLD = 0.70

@pytest.fixture(scope="module")
def validation_pipeline():
    """
    Fixture per caricare il modello fine-tuned UNA SOLA VOLTA.
    Carichiamo il modello che il job 'train-and-deploy' ha appena caricato su HF.
    """
    print(f"\n[Validation Fixture] Loading model {MODEL_TO_VALIDATE} from Hub...")
    # Usiamo 'pipeline' per semplificare l'inferenza in batch
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
    Fixture per caricare il "test set" di tweet_eval.
    Questo set non è MAI stato usato in training o validazione,
    quindi è perfetto per un giudizio imparziale.
    """
    print("[Validation Fixture] Loading 'tweet_eval' test set...")
    dataset = load_dataset("tweet_eval", "sentiment", split="test")
    # Per velocità, potremmo usare solo un subset, ma usiamo tutto il test set
    # dataset = dataset.shuffle(seed=42).select(range(1000)) # Opzionale
    print(f"Test set loaded with {len(dataset)} examples.")
    return dataset


def test_model_performance(validation_pipeline, test_dataset):
    """
    Il test di validazione vero e proprio.
    Verifica che l'accuratezza del modello sul test set sia sopra la soglia.
    """
    pipe = validation_pipeline
    
    # Mappiamo le label numeriche (0, 1, 2) ai nomi (negative, neutral, positive)
    # per confrontarle con l'output della pipeline.
    # NOTA: L'ordine DEVE corrispondere a quello del modello (controlla config.json su HF)
    # Per cardiffnlp/twitter-roberta-base-sentiment-latest:
    # 0: negative, 1: neutral, 2: positive
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    # Prepariamo i dati
    texts = list(test_dataset["text"]) 
    true_labels = [label_map[label] for label in test_dataset["label"]]

    # Eseguiamo le predizioni (in batch per velocità)
    print("Running predictions on test set...")
    # `pipeline` restituisce dizionari, es: {'label': 'positive', 'score': 0.99}
    predictions_output = pipe(texts, batch_size=32)
    predicted_labels = [pred['label'].lower() for pred in predictions_output]
    print("Predictions complete.")

    # Calcoliamo l'accuratezza
    accuracy_on_test_set = accuracy_score(true_labels, predicted_labels)
    
    print(f"\n--- Model Validation Results ---")
    print(f"Accuracy on Test Set: {accuracy_on_test_set:.4f}")
    print(f"Minimum Threshold:    {MINIMUM_ACCURACY_THRESHOLD:.4f}")
    print("---------------------------------")

    # L'ASSERT FONDAMENTALE:
    # Se l'accuratezza è inferiore alla soglia, il test fallisce
    # e l'intera pipeline viene bloccata.
    assert accuracy_on_test_set > MINIMUM_ACCURACY_THRESHOLD, f"Model Accuracy ({accuracy_on_test_set:.4f}) is below the threshold ({MINIMUM_ACCURACY_THRESHOLD:.4f})!"