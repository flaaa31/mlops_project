from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

# Importing the class created
from sentiment_analyzer import SentimentAnalyzer

# Prometheus tools
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter

# HTML/CSS/JS for the Product Review Interface [vibe coded because I'm not a developer :')]
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Product Review</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
        @import url('https://rsms.me/inter/inter.css');
        .response-positive { color: #22c55e; } /* Green */
        .response-negative { color: #ef4444; } /* Red */
        .response-neutral { color: #9ca3af; }  /* Gray */
    </style>
</head>
<body class="bg-gray-900 text-white h-full flex items-center justify-center p-4">

    <div class="bg-gray-800 p-8 rounded-2xl shadow-2xl w-full max-w-2xl text-center">
        <h1 class="text-4xl font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-500">Leave a Review</h1>
        <p class="text-gray-400 mb-6">Your feedback is important to us. Let us know how we did!</p>
        
        <div>
            <textarea id="textInput" 
                      rows="5"
                      class="w-full p-4 bg-gray-700 border-2 border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none transition resize-none" 
                      placeholder="Write your review here... (e.g., 'I love this product, it's amazing!')"></textarea>
            
            <button type="button" 
                    id="submitButton"
                    class="w-full mt-4 bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-4 rounded-lg transition-transform transform hover:scale-105">
                Submit Review
            </button>
        </div>

        <div id="response" class="mt-6 text-xl font-medium min-h-[56px] flex items-center justify-center">
            <!-- The contextual response will appear here -->
        </div>
    </div>

    <script>
        const textInput = document.getElementById('textInput');
        const responseDiv = document.getElementById('response');
        const submitButton = document.getElementById('submitButton');

        submitButton.addEventListener('click', async () => {
            const text = textInput.value;
            if (!text.trim()) {
                responseDiv.innerHTML = '<span class="text-yellow-400">Please enter your review before submitting.</span>';
                return;
            }

            // Show loading state
            submitButton.disabled = true;
            submitButton.textContent = 'Submitting...';
            responseDiv.innerHTML = '';

            try {
                const apiResponse = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                if (!apiResponse.ok) {
                    throw new Error(`HTTP Error: ${apiResponse.status}`);
                }

                const data = await apiResponse.json();
                
                let responseMessage = '';
                let responseClass = '';

                // Generate a contextual response based on the sentiment
                switch (data.label.toLowerCase()) {
                    case 'positive':
                        responseMessage = "Thank you so much for your positive feedback! We're thrilled you enjoyed our product.";
                        responseClass = 'response-positive';
                        break;
                    case 'negative':
                        responseMessage = "We are very sorry to hear you had a negative experience. We appreciate your feedback and will use it to improve.";
                        responseClass = 'response-negative';
                        break;
                    case 'neutral':
                        responseMessage = "Thank you for your time. Your review has been recorded.";
                        responseClass = 'response-neutral';
                        break;
                    default: // Should not happen, but as a fallback
                        responseMessage = "Thank you for your feedback.";
                        responseClass = 'text-gray-400';
                }
                
                responseDiv.innerHTML = `<p class="${responseClass}">${responseMessage}</p>`;
                textInput.value = ''; // Clear the textarea after submission

            } catch (error) {
                console.error('Analysis error:', error);
                responseDiv.innerHTML = '<span class="text-red-500">An error occurred. Please try again.</span>';
            } finally {
                // Restore the button
                submitButton.disabled = false;
                submitButton.textContent = 'Submit Review';
            }
        });

        textInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                submitButton.click();
            }
        });
    </script>
</body>
</html>
"""

# BACKEND CONFIGURATION

# CONFIGURATION
app = FastAPI(
    title="Sentiment Analysis API",
    description="An API to analyze the sentiment of text using a RoBERTa model.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
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
    """Serves the interactive HTML user interface."""
    return HTML_CONTENT

@app.post("/analyze", response_model=SentimentResponse, tags=["Analysis"])
def analyze_sentiment(request: SentimentRequest):
    """
    Analyzes the sentiment of the provided text and updates Prometheus metrics.
    This endpoint is used by the frontend UI.
    """
    result = analyzer.analyze(request.text)
    response_data = SentimentResponse(label=result.get("label"), score=result.get("score"))
    
    if response_data.label in ["positive", "negative", "neutral"]:
        sentiment_counter.labels(label=response_data.label).inc()
    
    return response_data

