import sys, os
import pytest

# Add the root directory to PYTHONPATH to resolve some import problems
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the class to be tested
from sentiment_analyzer import SentimentAnalyzer

# Testing the "logic" of the SentimentAnalyzer class (loading, preprocessing, inference) using the original base model
BASE_MODEL_FOR_TESTING = "cardiffnlp/twitter-roberta-base-sentiment-latest"

@pytest.fixture(scope="module")
def analyzer_for_testing():
    """
    A pytest "fixture" to create a single instance of the SentimentAnalyzer.
    
    Args:
        scope="module": load model only once, in order to save time
      
    Returns:
        SentimentAnalyzer: An initialized instance of the analyzer.
    """
    print(f"\n[Fixture] Loading model {BASE_MODEL_FOR_TESTING} for testing...")
    return SentimentAnalyzer(model_name=BASE_MODEL_FOR_TESTING)

def test_positive_sentiment(analyzer_for_testing):
    """
    Tests if the analyzer correctly identifies a clearly positive sentiment.
    
    Args:
        analyzer_for_testing (SentimentAnalyzer): The instance provided by the fixture.
    """
    result = analyzer_for_testing.analyze("I love this product! It's fantastic.")
    assert result["label"].lower() == "positive"

def test_negative_sentiment(analyzer_for_testing):
    """
    Tests if the analyzer correctly identifies a clearly negative sentiment.
    
    Args:
        analyzer_for_testing (SentimentAnalyzer): The instance provided by the fixture.
    """
    result = analyzer_for_testing.analyze("I hate this service. It's the worst.")
    assert result["label"].lower() == "negative"

def test_neutral_sentiment(analyzer_for_testing):
    """
    Tests if the analyzer correctly identifies a clearly neutral (objective) statement.
    
    Args:
        analyzer_for_testing (SentimentAnalyzer): The instance provided by the fixture.
    """
    result = analyzer_for_testing.analyze("This service is normal")
    assert result["label"].lower() == "neutral"
    
def test_preprocessing_logic(analyzer_for_testing):
    """Tests if the preprocess method correctly replaces @mentions and URLs with generic ones."""
    analyzer = analyzer_for_testing
    
    raw_text = "Hello @user1, this is my site http://example.com"
    expected = "Hello @user this is my site http"
    processed = analyzer.preprocess(raw_text)
    assert processed == expected

def test_empty_input(analyzer_for_testing):
    """Tests how the analyzer handles an empty string."""
    result = analyzer_for_testing.analyze("")
    assert result["label"].lower() == "error"
    assert "empty" in result["detail"].lower()

def test_emoji_sentiment(analyzer_for_testing):
    """Tests if the model can handle emojis (since it's trained on tweets) without breaking."""
    result_pos = analyzer_for_testing.analyze("I love this! 😍")
    result_neg = analyzer_for_testing.analyze("This is terrible 😡")
    assert result_pos["label"].lower() == "positive"
    assert result_neg["label"].lower() == "negative"