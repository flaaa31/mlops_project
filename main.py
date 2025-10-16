from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

# Importing the class created
from sentiment_analyzer import SentimentAnalyzer

# Prometheus tools
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

# --- HTML/CSS/JS per l'Interfaccia Utente ---
# Abbiamo inserito tutto qui per mantenere il progetto in un unico file.
# Usiamo Tailwind CSS per uno stile moderno senza file CSS esterni.
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="it" class="h-full">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis API</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
        @import url('https://rsms.me/inter/inter.css');
        .sentiment-positive { color: #22c55e; } /* Verde */
        .sentiment-negative { color: #ef4444; } /* Rosso */
        .sentiment-neutral { color: #6b7280; }  /* Grigio */
    </style>
</head>
<body class="bg-gray-900 text-white h-full flex items-center justify-center">

    <div class="bg-gray-800 p-8 rounded-2xl shadow-2xl w-full max-w-2xl text-center">
        <h1 class="text-4xl font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-teal-300">Sentiment Analysis API</h1>
        <p class="text-gray-400 mb-6">Analizza il sentiment di un testo in tempo reale usando un modello RoBERTa.</p>
        
        <div>
            <textarea id="textInput" 
                      rows="4"
                      class="w-full p-4 bg-gray-700 border-2 border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition resize-none" 
                      placeholder="Scrivi una frase qui... (es. 'I love this product, it's amazing!')"></textarea>
            
            <button type="button" 
                    id="analyzeButton"
                    class="w-full mt-4 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition-transform transform hover:scale-105">
                Analizza Sentiment
            </button>
        </div>

        <div id="result" class="mt-6 text-2xl font-semibold min-h-[32px]">
            <!-- Il risultato apparirà qui -->
        </div>
    </div>

    <script>
        const textInput = document.getElementById('textInput');
        const resultDiv = document.getElementById('result');
        const analyzeButton = document.getElementById('analyzeButton');

        analyzeButton.addEventListener('click', async () => {
            const text = textInput.value;
            if (!text.trim()) {
                resultDiv.innerHTML = '<span class="text-yellow-400">Per favore, inserisci del testo.</span>';
                return;
            }

            analyzeButton.disabled = true;
            analyzeButton.textContent = 'Analizzando...';
            resultDiv.innerHTML = '';

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                if (!response.ok) {
                    throw new Error(`Errore HTTP: ${response.status}`);
                }

                const data = await response.json();
                
                const scorePercentage = (data.score * 100).toFixed(1);
                const sentimentClass = `sentiment-${data.label.toLowerCase()}`;
                
                resultDiv.innerHTML = `Sentiment: <span class="${sentimentClass}">${data.label}</span> (${scorePercentage}%)`;

            } catch (error) {
                // CORREZIONE QUI: Usiamo una stringa semplice senza caratteri speciali.
                console.error('Analysis error:', error);
                resultDiv.innerHTML = '<span class="text-red-500">An error occurred. Please try again.</span>';
            } finally {
                analyzeButton.disabled = false;
                analyzeButton.textContent = 'Analizza Sentiment';
            }
        });

        textInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                analyzeButton.click();
            }
        });
    </script>
</body>
</html>
"""

# CONFIGURATION
app = FastAPI(
    title="Sentiment Analysis API",
    description="An API to analyze the sentiment of text using a RoBERTa model.",
    version="1.0.0",
    docs_url="/docs", # Manteniamo /docs per la documentazione tecnica
    redoc_url=None # Disabilitiamo l'altra interfaccia di documentazione per pulizia
)

# Analyzer instance creation when application starts
analyzer = SentimentAnalyzer()

# PROMETHEUS MONITORING
Instrumentator().instrument(app).expose(app)
sentiment_counter = Counter(
    "sentiment_analysis_predictions_total",
    "Counts the number of predictions for each sentiment label.",
    ["label"]
)

# DATA MODELS (Pydantic)
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    score: float

# API ENDPOINTS
@app.get("/", response_class=HTMLResponse, tags=["User Interface"])
async def read_root():
    """Serve la pagina HTML interattiva come interfaccia utente principale."""
    return HTML_CONTENT

@app.post("/analyze", response_model=SentimentResponse, tags=["Analysis"])
def analyze_sentiment(request: SentimentRequest):
    """
    Analizza il sentiment del testo fornito e aggiorna le metriche di Prometheus.
    """
    result = analyzer.analyze(request.text)
    response_data = SentimentResponse(label=result.get("label"), score=result.get("score"))
    
    if response_data.label in ["positive", "negative", "neutral"]:
        sentiment_counter.labels(label=response_data.label).inc()
    
    return response_data



