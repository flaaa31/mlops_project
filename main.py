from fastapi import FastAPI
from pydantic import BaseModel

# Importiamo la classe, non l'istanza
from sentiment_analyzer import SentimentAnalyzer

# Importiamo gli strumenti per Prometheus
from prometheus_fastapi_instrumentator import Instrumentator
# CORREZIONE 1: La classe Counter va importata da prometheus_client
from prometheus_client import Counter

# --- CONFIGURAZIONE E AVVIO ---

# Creiamo l'app FastAPI
app = FastAPI(
    title="Sentiment Analysis API",
    description="An API to analyze the sentiment of text using a RoBERTa model.",
    version="1.0.0"
)

# Creiamo l'istanza dell'analizzatore all'avvio dell'applicazione
analyzer = SentimentAnalyzer()

# --- MONITORAGGIO CON PROMETHEUS ---

# Applichiamo l'instrumentator all'app FastAPI
# Questo esporrà automaticamente metriche standard (latenza, richieste, etc.)
Instrumentator().instrument(app).expose(app)

# CORREZIONE 2: Creiamo il contatore con la sintassi corretta per le etichette
sentiment_counter = Counter(
    "sentiment_analysis_predictions_total",
    "Counts the number of predictions for each sentiment label.",
    ["label"]  # Le etichette vanno definite come lista di stringhe
)


# --- MODELLI DI DATI (Pydantic) ---

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    score: float


# --- ENDPOINTS DELL'API ---

@app.get("/", tags=["Health Check"])
def read_root():
    """Endpoint principale per verificare lo stato dell'API."""
    return {"status": "ok", "message": "Welcome to the Sentiment Analysis API!"}

@app.post("/analyze", response_model=SentimentResponse, tags=["Analysis"])
def analyze_sentiment(request: SentimentRequest):
    """
    Analizza il sentiment del testo fornito e aggiorna le metriche di Prometheus.
    """
    result = analyzer.analyze(request.text)
    
    response_data = SentimentResponse(label=result.get("label"), score=result.get("score"))
    
    # Incrementiamo il nostro contatore personalizzato con l'etichetta predetta
    if response_data.label in ["positive", "negative", "neutral"]:
        # CORREZIONE 3: Passare l'etichetta come keyword argument
        sentiment_counter.labels(label=response_data.label).inc()
    
    return response_data



