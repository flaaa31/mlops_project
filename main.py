from fastapi import FastAPI
from pydantic import BaseModel

# We import the analyzer instance from our sentiment_analyzer module
from sentiment_analyzer import analyzer

# Create a FastAPI app instance
app = FastAPI(
    title="Sentiment Analysis API",
    description="An API to analyze the sentiment of text using a RoBERTa model.",
    version="1.0.0"
)

# Define the request body model for input validation
class SentimentRequest(BaseModel):
    text: str

# Define the response body model for clarity
class SentimentResponse(BaseModel):
    label: str
    score: float

@app.get("/", tags=["Health Check"])
def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"status": "ok", "message": "Welcome to the Sentiment Analysis API!"}

@app.post("/analyze", response_model=SentimentResponse, tags=["Analysis"])
def analyze_sentiment(request: SentimentRequest):
    """
    Analyzes the sentiment of the text provided in the request.
    """
    result = analyzer.analyze(request.text)
    return result
