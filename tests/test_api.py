import sys, os
import pytest
from fastapi.testclient import TestClient

# Add the root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the FastAPI app instance from main.py
from main import app 

# Create a test client that can make "fake" HTTP requests to your app
client = TestClient(app)

def test_api_root_endpoint():
    """Tests if the root endpoint '/' returns the HTML interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers['content-type']
    assert "Leave a Review" in response.text # Check for a keyword in your HTML

def test_api_analyze_endpoint_positive():
    """Tests the /analyze endpoint with a positive sentiment."""
    response = client.post(
        "/analyze",
        json={"text": "This is a wonderful product, I am so happy!"}
    )
    data = response.json()
    assert response.status_code == 200
    assert data["label"].lower() == "positive"

def test_api_analyze_endpoint_empty():
    """Tests how the API handles an empty request."""
    response = client.post(
        "/analyze",
        json={"text": ""} 
    )
    data = response.json()
    assert response.status_code == 200
    assert data["label"].lower() == "error"

def test_api_analyze_bad_request():
    """Tests what happens if we send malformed JSON."""
    response = client.post(
        "/analyze",
        json={"not_text": "hello"} # missing "text" key
    )
    
    assert response.status_code == 422 # 422 Unprocessable Entity