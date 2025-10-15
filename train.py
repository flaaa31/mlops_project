import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- FASE 1: CONFIGURAZIONE ---
# Modello di base da cui partire
BASE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Nome del dataset da Hugging Face
DATASET_NAME = "tweet_eval"
# Nome della configurazione specifica del dataset
DATASET_CONFIG = "sentiment"
# Nome del repository dove verrà salvato il modello su Hugging Face
REPO_NAME_ON_HUB = "sentiment_model_for_hf"

# Leggiamo le credenziali dai secrets di GitHub
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")

push_to_hub = bool(HF_TOKEN and HF_USERNAME)

if push_to_hub:
    repo_id = f"{HF_USERNAME}/{REPO_NAME_ON_HUB}"
    print(f"Credenziali trovate. Il modello verrà caricato su Hugging Face Hub in '{repo_id}'.")
else:
    print("Variabili d'ambiente HF_TOKEN o HF_USERNAME non impostate. Salto il push sull'Hub.")


# --- FASE 2: CARICAMENTO E PREPARAZIONE DEL DATASET ---
def main():
    print(f"Caricamento del dataset '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    # Utilizziamo un sottoinsieme per un training più veloce
    train_subset = dataset['train'].shuffle(seed=42).select(range(7000))
    validation_subset = dataset['validation'].shuffle(seed=42).select(range(3000))
    print(f"Dataset suddiviso in {len(train_subset)} esempi di training e {len(validation_subset)} di valutazione.")


    # --- FASE 3: TOKENIZZAZIONE ---
    print(f"Caricamento del tokenizer per '{BASE_MODEL}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_train_dataset = train_subset.map(tokenize_function, batched=True)
    tokenized_validation_dataset = validation_subset.map(tokenize_function, batched=True)


    # --- FASE 4: CARICAMENTO DEL MODELLO E METRICHE ---
    print(f"Caricamento del modello pre-addestrato '{BASE_MODEL}'...")
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


    # --- FASE 5: DEFINIZIONE DEGLI ARGOMENTI DI TRAINING ---
    print("Definizione degli argomenti di training...")
    training_args = TrainingArguments(
        output_dir="logs",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",  # <-- CORREZIONE APPLICATA QUI
        save_strategy="no",     # Non salviamo checkpoint in locale per risparmiare spazio
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=push_to_hub,
        hub_token=HF_TOKEN,
        hub_model_id=repo_id if push_to_hub else None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )


    # --- FASE 6: TRAINING E DEPLOY ---
    print("Inizio del processo di fine-tuning...")
    trainer.train()

    if push_to_hub:
        print(f"Upload del modello su '{repo_id}' in corso...")
        trainer.push_to_hub()
        print("Upload completato con successo.")
    else:
        print("Salvataggio del modello in locale nella cartella 'sentiment_model_for_hf'...")
        trainer.save_model("sentiment_model_for_hf")
        print("Modello salvato correttamente.")


if __name__ == "__main__":
    main()

