from fastapi import FastAPI
from pydantic import BaseModel
import os

# Importiamo la CLASSE SentimentAnalyzer, non un'istanza
from sentiment_analyzer import SentimentAnalyzer

# --- Logica di Configurazione ---
# Recuperiamo lo username di Hugging Face dalle variabili d'ambiente.
# Se non è impostato, il sistema di test userà il modello di base.
HF_USERNAME = os.getenv("HF_USERNAME")

# Determiniamo quale modello caricare.
# Se HF_USERNAME è disponibile, puntiamo al nostro modello fine-tunato.
# Altrimenti, per sicurezza o test locali, potremmo puntare a un modello di base.
if HF_USERNAME:
    MODEL_TO_LOAD = f"{HF_USERNAME}/sentiment_model_for_hf"
else:
    # Modello di fallback nel caso in cui l'app venga eseguita senza context.
    MODEL_TO_LOAD = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# --- Creazione dell'istanza dell'Analizzatore ---
# L'istanza viene creata qui, una sola volta all'avvio dell'applicazione.
try:
    analyzer = SentimentAnalyzer(model_name=MODEL_TO_LOAD)
except Exception as e:
    # Se il modello non può essere caricato all'avvio, l'app non dovrebbe partire.
    print(f"CRITICAL ERROR: Could not load the sentiment model. Shutting down. Error: {e}")
    # In un'app reale, qui si potrebbe uscire o gestire l'errore in modo più elegante.
    analyzer = None # Imposta a None per gestire l'errore negli endpoint.


# --- Applicazione FastAPI ---
app = FastAPI(
    title="Sentiment Analysis API",
    description="An API to analyze the sentiment of text using a RoBERTa model.",
    version="1.0.0"
)

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    score: float

@app.get("/", tags=["Health Check"])
def read_root():
    """Root endpoint to check if the API is running."""
    status = "ok" if analyzer is not None else "error"
    message = "Welcome to the Sentiment Analysis API!" if status == "ok" else "Sentiment analyzer model could not be loaded."
    return {"status": status, "message": message}

@app.post("/analyze", response_model=SentimentResponse, tags=["Analysis"])
def analyze_sentiment(request: SentimentRequest):
    """Analyzes the sentiment of the text provided in the request."""
    if analyzer is None:
        return {"label": "Error", "score": 0.0, "detail": "Model is not available."}
    
    result = analyzer.analyze(request.text)
    return result
