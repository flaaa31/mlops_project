from fastapi import FastAPI
from pydantic import BaseModel

# Importing the class created
from sentiment_analyzer import SentimentAnalyzer

# Prometheus tools
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

# CONFIGURATION

# FastAPI app creation
app = FastAPI(
    title="Sentiment Analysis API",
    description="An API to analyze the sentiment of text using a RoBERTa model.",
    version="1.0.0"
)

# Analyzer instance creation when application starts
analyzer = SentimentAnalyzer()

# ROMETHEUS MONITORING

# Applying the instrumentator to Applichiamo FastAPI app, in order to expose standard metrics (latency, requests ecc...)
Instrumentator().instrument(app).expose(app)

# Creating the counter
sentiment_counter = Counter(
    "sentiment_analysis_predictions_total",
    "Counts the number of predictions for each sentiment label.",
    ["label"]  # Le etichette vanno definite come lista di stringhe
)


# DATA MODELS (Pydantic)

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    score: float


# API ENDPOINT

@app.get("/", tags=["Health Check"])
def read_root():
    """Main endpoint to check API status."""
    return {"status": "ok", "message": "Welcome to the Sentiment Analysis API!"}

@app.post("/analyze", response_model=SentimentResponse, tags=["Analysis"])
def analyze_sentiment(request: SentimentRequest):
    """
    Analzyzes sentiment of the given text and updates Prometheus metrics.
    """
    result = analyzer.analyze(request.text)
    
    response_data = SentimentResponse(label=result.get("label"), score=result.get("score"))
    
    # Increasing our personalized counter 
    if response_data.label in ["positive", "negative", "neutral"]:
        
        sentiment_counter.labels(label=response_data.label).inc()
    
    return response_data



