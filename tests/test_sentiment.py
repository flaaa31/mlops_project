import sys, os
import pytest

# --- Python Path Modification ---
# This line adds the project's root directory (the parent of 'tests/') to
# the list of paths Python searches for modules.
# This is necessary so that Python can find and import 'sentiment_analyzer'
# from the parent directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the class to be tested
from sentiment_analyzer import SentimentAnalyzer

# --- Test Configuration ---
# Define the model to use for testing.
# We use the *original base model* from Hugging Face, not the fine-tuned one.
# This makes the tests stable and independent of the training job.
# We are testing the *logic* of the SentimentAnalyzer class (loading, preprocessing,
# inference), not the *quality* of the fine-tuning.
BASE_MODEL_FOR_TESTING = "cardiffnlp/twitter-roberta-base-sentiment-latest"

@pytest.fixture(scope="module")
def analyzer_for_testing():
    """
    A pytest "fixture" to create a single instance of the SentimentAnalyzer.
    
    - `@pytest.fixture`: Marks this function as a fixture.
    - `scope="module"`: This fixture will run only ONCE for all tests in this
      file. The same `analyzer` instance will be shared among all test functions.
      This is critical because model loading (`__init__`) is very slow, and
      we don't want to reload it for every single test.
      
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
    result = analyzer_for_testing.analyze("Tomorrow the sun will rise at 6:00 AM")
    assert result["label"].lower() == "neutral"
    
def test_preprocessing_logic(analyzer_for_testing):
    """Tests if the preprocess method correctly replaces @mentions and URLs."""
    analyzer = analyzer_for_testing
    
    raw_text = "Hello @user1, this is my site http://example.com"
    expected = "Hello @user, this is my site http"
    processed = analyzer.preprocess(raw_text)
    assert processed == expected

def test_empty_input(analyzer_for_testing):
    """Tests how the analyzer handles an empty string."""
    result = analyzer_for_testing.analyze("")
    assert result["label"].lower() == "error"
    assert "empty" in result["detail"].lower()

def test_emoji_sentiment(analyzer_for_testing):
    """Tests if the model can handle emojis (since it's trained on tweets)."""
    result_pos = analyzer_for_testing.analyze("I love this! 😍")
    result_neg = analyzer_for_testing.analyze("This is terrible 😡")
    assert result_pos["label"].lower() == "positive"
    assert result_neg["label"].lower() == "negative"