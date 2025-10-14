import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
import evaluate

# --- 1. CONFIGURAZIONE ---
# Definiamo i parametri principali
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET_NAME = "sentiment140"
OUTPUT_DIR = "./sentiment_model"
LOGGING_DIR = "./logs"
NUM_TRAIN_EPOCHS = 1 # Per un'esecuzione veloce, altrimenti aumentare a 3-5
BATCH_SIZE = 16 # Aumentare se si ha una GPU con più memoria

def preprocess_function(examples, tokenizer):
    """Funzione per tokenizzare il testo."""
    return tokenizer(examples['text'], truncation=True, padding=True)

def compute_metrics(eval_pred):
    """Funzione per calcolare le metriche di valutazione."""
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    
    return {"accuracy": accuracy, "f1": f1}

def main():
    """
    Funzione principale che orchestra il download del dataset,
    il fine-tuning del modello e il salvataggio dei risultati.
    """
    # --- 2. CARICAMENTO E PREPARAZIONE DEL DATASET ---
    print(f"Caricamento del dataset '{DATASET_NAME}'...")
    # Usiamo un dataset pubblico da Hugging Face.
    # 'sentiment140' contiene tweet etichettati come negativi (0) o positivi (4).
    dataset = load_dataset(DATASET_NAME, split='train')

    # Rinominiamo le colonne per coerenza e mappiamo le etichette
    dataset = dataset.rename_column("sentiment", "label")
    dataset = dataset.rename_column("text", "text")
    
    # RoBERTa si aspetta etichette 0 (negativo), 1 (neutro), 2 (positivo).
    # Il modello originale è stato addestrato su {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    # Il dataset sentiment140 ha {0: 'Negative', 4: 'Positive'}.
    # Dobbiamo mappare 4 a 2. Non abbiamo etichette neutrali qui.
    def map_labels(example):
        if example['label'] == 4:
            example['label'] = 2 # Mappiamo 'Positive'
        # L'etichetta 0 per 'Negative' è già corretta
        return example

    dataset = dataset.map(map_labels)
    
    # Per rendere il processo più veloce, usiamo un sottoinsieme del dataset.
    # Rimuovi o aumenta il valore di 'select' per un training completo.
    print("Creazione di un sottoinsieme del dataset per un training più rapido...")
    subset = dataset.shuffle(seed=42).select(range(20000)) # 10k per train, 10k per test

    # Divisione in training e test set
    train_test_split_dataset = subset.train_test_split(test_size=0.3)
    train_dataset = train_test_split_dataset['train']
    eval_dataset = train_test_split_dataset['test']
    
    print(f"Dataset suddiviso in {len(train_dataset)} esempi di training e {len(eval_dataset)} di valutazione.")

    # --- 3. TOKENIZZAZIONE ---
    print(f"Caricamento del tokenizer per il modello '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizzazione dei dataset...")
    tokenized_train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    tokenized_eval_dataset = eval_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 4. CARICAMENTO DEL MODELLO ---
    print(f"Caricamento del modello pre-addestrato '{MODEL_NAME}'...")
    # Specifichiamo il numero di etichette attese. Il modello originale ne ha 3.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # --- 5. TRAINING ---
    print("Definizione degli argomenti di training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=LOGGING_DIR,
        push_to_hub=False, # Impostare a True per caricare su Hugging Face Hub
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Inizio del processo di fine-tuning...")
    trainer.train()

    # --- 6. SALVATAGGIO DEL MODELLO ---
    print(f"Salvataggio del modello fine-tuned in '{OUTPUT_DIR}'...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training completato con successo!")

if __name__ == "__main__":
    main()
