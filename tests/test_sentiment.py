import sys, os
import pytest

# Adding project root on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing SentimentAnalyzer class
from sentiment_analyzer import SentimentAnalyzer

# Defining original base model, for tests
BASE_MODEL_FOR_TESTING = "cardiffnlp/twitter-roberta-base-sentiment-latest"

@pytest.fixture(scope="module")
def analyzer_for_testing():
    """
    pytest "fixture": it creates an analyzer instance only for tests execution
    the base model makes tests and deploy independent between each other
    The model is loaded every time.
    """
    return SentimentAnalyzer(model_name=BASE_MODEL_FOR_TESTING)

def test_positive_sentiment(analyzer_for_testing):
    """Using test analyzer to test a positive sentiment."""
    result = analyzer_for_testing.analyze("I love this product! It's fantastic.")
    assert result["label"].lower() == "positive"

def test_negative_sentiment(analyzer_for_testing):
    """Using test analyzer to test a negative sentiment."""
    result = analyzer_for_testing.analyze("I hate this service.")
    assert result["label"].lower() == "negative"

def test_neutral_sentiment(analyzer_for_testing):
    """Using test analyzer to test a neutral sentiment."""
    result = analyzer_for_testing.analyze("Tomorrow the sun will rise at 6:00 AM")
    assert result["label"].lower() == "neutral"
