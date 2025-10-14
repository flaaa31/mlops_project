import os
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# --- 1. CONFIGURAZIONE ---
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET_NAME = "tweet_eval"
DATASET_CONFIG = "sentiment"
OUTPUT_DIR = "sentiment_model_for_hf" # Nome del repository su Hugging Face
LOGGING_DIR = "./logs"
NUM_TRAIN_EPOCHS = 1
BATCH_SIZE = 16

def preprocess_function(examples, tokenizer):
    """Tokenizza il testo di input."""
    return tokenizer(examples['text'], truncation=True, padding=True)

def compute_metrics(eval_pred):
    """Calcola le metriche di accuratezza e F1."""
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": accuracy, "f1": f1}

def main():
    # --- INTEGRAZIONE CON HUGGING FACE HUB ---
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    repo_id = f"{hf_username}/{OUTPUT_DIR}" if hf_username else None

    if not hf_token or not repo_id:
        print("Variabili d'ambiente HF_TOKEN o HF_USERNAME non impostate. Salto il push sull'Hub.")
        push_to_hub_enabled = False
    else:
        print(f"Credenziali trovate. Il modello verrà caricato su Hugging Face Hub in '{repo_id}'.")
        push_to_hub_enabled = True

    # --- 2. CARICAMENTO E PREPARAZIONE DEL DATASET ---
    print(f"Caricamento del dataset '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split='train')
    subset = dataset.shuffle(seed=42).select(range(10000))
    train_test_split_dataset = subset.train_test_split(test_size=0.3)
    train_dataset = train_test_split_dataset['train']
    eval_dataset = train_test_split_dataset['test']
    print(f"Dataset suddiviso in {len(train_dataset)} esempi di training e {len(eval_dataset)} di valutazione.")

    # --- 3. TOKENIZZAZIONE ---
    print(f"Caricamento del tokenizer per '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train_dataset = train_dataset.map(lambda ex: preprocess_function(ex, tokenizer), batched=True)
    tokenized_eval_dataset = eval_dataset.map(lambda ex: preprocess_function(ex, tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 4. CARICAMENTO DEL MODELLO ---
    print(f"Caricamento del modello pre-addestrato '{MODEL_NAME}'...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # --- 5. TRAINING E DEPLOY ---
    print("Definizione degli argomenti di training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no", # Non salva checkpoint locali per risparmiare spazio
        load_best_model_at_end=True,
        logging_dir=LOGGING_DIR,
        push_to_hub=push_to_hub_enabled,
        hub_model_id=repo_id,
        hub_token=hf_token,
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

    # --- 6. PUSH SULL'HUB ---
    if push_to_hub_enabled:
        print(f"Push del modello migliore su Hugging Face Hub: '{repo_id}'...")
        trainer.push_to_hub(commit_message="End of training from CI/CD pipeline")
        print("Push completato!")
    else:
        print("Salvataggio locale del modello (push sull'Hub disabilitato).")
        trainer.save_model(OUTPUT_DIR)

    print("Processo completato con successo!")

if __name__ == "__main__":
    main()

