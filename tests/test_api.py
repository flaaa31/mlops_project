import sys, os
import pytest
from fastapi.testclient import TestClient

# Add the root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the FastAPI app instance from your main.py file
# NOTA: questo potrebbe richiedere di modificare main.py per evitare
# di caricare il modello se il file è importato (ma per ora proviamo così)
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
    assert data["score"] > 0.8 # Possiamo anche testare la confidenza

def test_api_analyze_endpoint_empty():
    """Tests how the API handles an empty request (via Pydantic)."""
    response = client.post(
        "/analyze",
        json={"text": ""} # Questo è gestito dalla tua classe Analyzer
    )
    data = response.json()
    assert response.status_code == 200
    assert data["label"].lower() == "error"

def test_api_analyze_bad_request():
    """Tests what happens if we send malformed JSON."""
    response = client.post(
        "/analyze",
        json={"not_text": "hello"} # Chiave "text" mancante
    )
    # FastAPI's Pydantic validation should catch this
    assert response.status_code == 422 # 422 Unprocessable Entity