import os
import argparse
from datasets import DatasetDict, Dataset, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

# --- Configurazione (Parametri fissi) ---
MODEL_CHECKPOINT = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET_PATH = "data/processed/tokenized_dataset"
OUTPUT_DIR = "models/sentiment_retrained" # Dove verrà salvato il modello
LOGGING_DIR = "logs" # Cartella per i log del training

def run_training():
    print("--- 1. Caricamento Dati e Tokenizer ---")
    
    # Carica il dataset tokenizzato salvato nella Fase 1
    full_dataset = load_from_disk(DATASET_PATH)

    # Suddivisione in train e test (80/20)
    # È meglio fare lo split prima di salvare, ma lo facciamo qui per semplicità iniziale
    split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
    
    # 2. Caricamento Modello e Tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # 3. Impostazioni del Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,                     # Esempio: 3 epoche
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,                       # Warm-up
        weight_decay=0.01,
        logging_dir=LOGGING_DIR,
        logging_steps=500,
        eval_strategy="epoch",            # Valuta alla fine di ogni epoca
        save_strategy="epoch",
        load_best_model_at_end=True,            # Carica il modello migliore
        metric_for_best_model="accuracy",       # Metric per il modello migliore
        push_to_hub=False                       # Non pushare ancora su HF
    )

    # 4. Creazione e Avvio del Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        # Nota: Qui andrebbe una funzione compute_metrics per l'accuratezza
    )
    
    print("--- 4. Avvio Training ---")
    trainer.train()
    
    # 5. Salvataggio finale del modello migliore (se load_best_model_at_end=True)
    trainer.save_model(OUTPUT_DIR)
    
    # Esecuzione del test finale e salvataggio delle metriche
    metrics = trainer.evaluate(split_dataset["test"])
    print("Metriche di Valutazione Finali:", metrics)
    
    # Salvataggio delle metriche (per i test di integrazione futuri)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        import json
        json.dump(metrics, f)
        
    print(f"Modello addestrato e metriche salvate in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_training()