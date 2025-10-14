import gradio as gr
from sentiment_analyzer import analyzer  # import dal tuo file già esistente

def predict_sentiment(text: str) -> str:
    """
    Funzione chiamata da Gradio per analizzare il sentiment.
    
    Args:
        text (str): Testo da analizzare
    
    Returns:
        str: Sentiment e punteggio in formato leggibile
    """
    if not text.strip():
        return "⚠️ Inserisci del testo da analizzare."

    result = analyzer.analyze(text)
    
    if result["label"] == "Error":
        return f"Errore: {result.get('detail', 'Unknown error')}"
    
    return f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})"

# Creazione dell'interfaccia Gradio
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Scrivi qui il testo da analizzare...",
        label="Testo"
    ),
    outputs=gr.Textbox(label="Risultato"),
    title="Sentiment Analyzer",
    description="Analizza il sentiment del testo usando RoBERTa pre-addestrato."
)

# Avvio dell'app
if __name__ == "__main__":
    iface.launch()
