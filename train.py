import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# CONFIGURATION
BASE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET_NAME = "tweet_eval"
DATASET_CONFIG = "sentiment"
HF_REPO = "sentiment_model_for_hf"

# HF Credentials
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")

push_to_hub_flag = bool(HF_TOKEN and HF_USERNAME)
repo_id = None
if push_to_hub_flag:
    repo_id = f"{HF_USERNAME}/{HF_REPO}"
    print(f"Credentials confirmed. The model will be loaded on Hugging Face Hub in '{repo_id}'.")
else:
    print("HF_TOKEN or HF_USERNAME not found. There will be no HF Hub push.")


def main():
    print(f"Loading dataset '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)


    # Subset of 20.000 examples for train
    train_subset = dataset['train'].shuffle(seed=42).select(range(20000))
    
    # entire validation dataset
    validation_subset = dataset['validation'] 
    print(f"Dataset divided in {len(train_subset)} training examples and {len(validation_subset)} validation examples.")

    print(f"Loading '{BASE_MODEL}' tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_train_dataset = train_subset.map(tokenize_function, batched=True)
    tokenized_validation_dataset = validation_subset.map(tokenize_function, batched=True)

    print(f"Loading pre-trained model '{BASE_MODEL}'...")
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

    training_args = TrainingArguments(
        output_dir="logs",

        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch", 
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="accuracy",
        push_to_hub=push_to_hub_flag,
        hub_token=HF_TOKEN,
        hub_model_id=repo_id if push_to_hub_flag else None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()

    if push_to_hub_flag:
        print(f"Uploading model on '{repo_id}'...")
        trainer.push_to_hub()
        print("Uploading completed")
    else:
        print("Saving model on 'sentiment_model_for_hf' folder...")
        trainer.save_model("sentiment_model_for_hf")
        print("Model saved.")


if __name__ == "__main__":
    main()

