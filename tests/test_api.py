import sys, os
from fastapi.testclient import TestClient

# Add the root directory to PYTHONPATH to resolve some import problems
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the FastAPI app instance from main.py
from main import app 

# Create a test client that can make "fake" HTTP requests to your app
client = TestClient(app)

def test_api_root_endpoint():
    """Tests if the root endpoint '/' returns the HTML interface."""
    response = client.get("/")
    assert response.status_code == 200 # Response == OK
    assert "text/html" in response.headers['content-type'] # Response type has to be a web page in html
    assert "Leave a Review" in response.text # Check for a keyword that I'm sure it's in my HTML

def test_api_analyze_endpoint_positive():
    """Tests the /analyze endpoint with a positive sentiment."""
    response = client.post(
        "/analyze",
        json={"text": "This is a wonderful product, I am so happy!"}
    )
    data = response.json()
    assert response.status_code == 200 # Response == OK
    assert data["label"].lower() == "positive" # predicted sentiment has to be positive, bit we're checking the API here

def test_api_analyze_endpoint_empty():
    """Tests how the API handles an empty request."""
    response = client.post(
        "/analyze",
        json={"text": ""} 
    )
    data = response.json()
    assert response.status_code == 200 # Response == OK
    assert data["label"].lower() == "error" # expected behaviour for empty inputs

def test_api_analyze_bad_request():
    """Tests what happens if we send malformed JSON."""
    response = client.post(
        "/analyze",
        json={"not_text": "hello"} # missing "text" key
    )
    
    assert response.status_code == 422 # 422 Unprocessable Entity