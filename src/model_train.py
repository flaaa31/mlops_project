import os
import json
import numpy as np
import argparse # Mantenuto per futura espansione se necessario

from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

# ====================================================================
# --- 1. CONFIGURAZIONE DEI PERCORSI E DEL MODELLO ---
# I percorsi sono relativi alla root del progetto, come richiesto dalla pipeline CI/CD
# ====================================================================

MODEL_CHECKPOINT = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET_PATH = "data/processed/tokenized_dataset" 
OUTPUT_DIR = "models/sentiment_retrained" 
LOGGING_DIR = "logs" 

# ====================================================================
# --- 2. FUNZIONE PER IL CALCOLO DELLE METRICHE (CORREZIONE KEYERROR) ---
# ====================================================================

def compute_metrics(eval_pred):
    """
    Calcola metriche di classificazione (accuratezza e F1) per il Trainer.
    È essenziale per usare 'metric_for_best_model' con l'accuratezza.
    """
    logits, labels = eval_pred
    
    # Prende la classe con il punteggio più alto
    predictions = np.argmax(logits, axis=-1)
    
    # Calcola l'accuratezza e la F1-score (media macro per dataset sbilanciati)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        'eval_accuracy': accuracy,
        'eval_f1_score': f1,
    }

# ====================================================================
# --- 3. FUNZIONE PRINCIPALE DI TRAINING ---
# ====================================================================

def run_training():
    print("--- 1. Caricamento Dati e Tokenizer ---")
    
    # Crea la directory di output se non esiste
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carica il dataset tokenizzato salvato nella Fase 1
    # Nota: Assumi che il dataset sia già splittato in train/test o lo splitti qui
    try:
        full_dataset = load_from_disk(DATASET_PATH)
    except Exception as e:
        print(f"ERRORE: Impossibile caricare il dataset da {DATASET_PATH}. Assicurati che esista. Errore: {e}")
        return

    # Suddivisione in train e test (se non è già un DatasetDict)
    if not isinstance(full_dataset, dict):
        # Splitta il dataset se è un singolo oggetto Dataset
        split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
    else:
        # Se è già un DatasetDict (con chiavi 'train', 'test', ecc.)
        split_dataset = full_dataset
    
    # 2. Caricamento Modello e Tokenizer
    # num_labels=3 perché cerchi positivo, neutro e negativo.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # 3. Impostazioni del Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,                     
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,                       
        weight_decay=0.01,
        logging_dir=LOGGING_DIR,
        logging_steps=500,
        eval_strategy="epoch",                  
        save_strategy="epoch",
        load_best_model_at_end=True,            
        metric_for_best_model="eval_accuracy",  
        push_to_hub=False                      
    )

    # 4. Creazione e Avvio del Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,        # PASSA LA FUNZIONE DI CALCOLO METRICHE
    )
    
    print("--- 4. Avvio Training ---")
    trainer.train()
    
    # 5. Salvataggio finale del modello migliore
    final_model_path = os.path.join(OUTPUT_DIR, "final_best_model")
    trainer.save_model(final_model_path)
    print(f"Modello addestrato salvato in: {final_model_path}")

    # 6. Esecuzione del test finale e salvataggio delle metriche
    metrics = trainer.evaluate(split_dataset["test"])
    print("Metriche di Valutazione Finali:")
    print(json.dumps(metrics, indent=4))
    
    # Salvataggio delle metriche per il 'Test di Integrazione' della pipeline CI/CD
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f)
        
    print(f"Metriche salvate in {os.path.join(OUTPUT_DIR, 'metrics.json')}")


if __name__ == "__main__":
    run_training()