import sys, os
# aggiunge la root del progetto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentiment_analyzer import analyzer

def test_positive_sentiment():
    result = analyzer.analyze("I love this product! It's fantastic.")
    assert result["label"].lower() == "positive"

def test_negative_sentiment():
    result = analyzer.analyze("I hate this service.")
    assert result["label"].lower() == "negative"
