import sys, os
import pytest

# Aggiunge la root del progetto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importiamo la CLASSE, non l'istanza globale pre-configurata
from sentiment_analyzer import SentimentAnalyzer

# Definiamo il modello di base originale, che è sempre disponibile
BASE_MODEL_FOR_TESTING = "cardiffnlp/twitter-roberta-base-sentiment-latest"

@pytest.fixture(scope="module")
def analyzer_for_testing():
    """
    Questa è una "fixture" di pytest. Crea un'istanza dell'analizzatore
    SOLO per l'esecuzione dei test.
    Usa il modello di base per garantire che i test siano indipendenti
    dal deploy e sempre eseguibili. Il modello viene caricato una sola volta.
    """
    return SentimentAnalyzer(model_name=BASE_MODEL_FOR_TESTING)

def test_positive_sentiment(analyzer_for_testing):
    """Testa un sentimento positivo usando l'analizzatore di test."""
    result = analyzer_for_testing.analyze("I love this product! It's fantastic.")
    assert result["label"].lower() == "positive"

def test_negative_sentiment(analyzer_for_testing):
    """Testa un sentimento negativo usando l'analizzatore di test."""
    result = analyzer_for_testing.analyze("I hate this service.")
    assert result["label"].lower() == "negative"

def test_neutral_sentiment(analyzer_for_testing):
    """Testa un sentimento neutro usando l'analizzatore di test."""
    result = analyzer_for_testing.analyze("Tomorrow the sun will rise at 6:00 AM")
    assert result["label"].lower() == "neutral"
