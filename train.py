import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# CONFIGURATION
# Base model from Hugging Face Hub to fine-tune
BASE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Dataset from Hugging Face Hub
DATASET_NAME = "tweet_eval"
# Specific configuration of the dataset (sentiment analysis task)
DATASET_CONFIG = "sentiment"
# Name of the trained model in local
LOCAL_MODEL_OUTPUT_DIR = "sentiment_model_local"

print(f"The model will be saved locally in: '{LOCAL_MODEL_OUTPUT_DIR}'.")


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation during training.
    
    This function is passed to the `Trainer` and called at each evaluation step.
    
    Args:
        eval_pred (tuple): A tuple containing model predictions (logits) and true labels.
        
    Returns:
        dict: A dictionary of computed metrics (accuracy, f1, precision, recall).
    """
    logits, labels = eval_pred
    # Get the predicted class by finding the index with the highest logit
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    """
    Main function to run the model fine-tuning and deployment pipeline.
    """
    
    # Loading Dataset
    print(f"Loading dataset '{DATASET_NAME}' with config '{DATASET_CONFIG}'...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    # Preparing Data Subsets
    # Shuffle and select a subset for training (20000 examples), a good balance between speed and quality.
    train_subset = dataset['train'].shuffle(seed=42).select(range(20000))
    
    # Use the entire validation dataset for robust evaluation
    validation_subset = dataset['validation'] 
    print(f"Dataset prepared: {len(train_subset)} training examples, {len(validation_subset)} validation examples.")

    # Tokenization
    print(f"Loading '{BASE_MODEL}' tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize_function(examples):
        """Tokenization function."""

        # `truncation=True` 
        return tokenizer(examples['text'], 
                         padding="max_length", # pads all sentences to the same length.
                         truncation=True)      # truncates sentences that are too long.

    # Apply tokenization to the datasets using .map()
    print("Tokenizing datasets...")
    tokenized_train_dataset = train_subset.map(tokenize_function, batched=True)
    tokenized_validation_dataset = validation_subset.map(tokenize_function, batched=True)

    # Loading Model
    print(f"Loading pre-trained model '{BASE_MODEL}' for fine-tuning...")

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, 
                                                               num_labels=3) # (positive, negative, neutral)

    # Configure Training
    training_args = TrainingArguments(
        output_dir=LOCAL_MODEL_OUTPUT_DIR,  # Directory to save logs and checkpoints

        # Training Hyperparameters
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,       # Number of steps to warm up the learning rate
        weight_decay=0.01,      # Strength of weight decay regularization

        # Logging and Saving
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",      # Run evaluation at the end of each epoch
        save_strategy="epoch",      # Save a checkpoint at the end of each epoch
        load_best_model_at_end=True, # Load the best model (based on metric) at the end
        save_total_limit=1,         # Only keep the single best checkpoint
        metric_for_best_model="accuracy", # Metric to determine the "best" model
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        compute_metrics=compute_metrics, # Pass the metrics function
        tokenizer=tokenizer
    )

    # Start Training
    print("Starting model training...")
    trainer.train()
    print("Training finished.")
    
    # Saving final model
    print(f"Saving final model to {LOCAL_MODEL_OUTPUT_DIR}...")
    trainer.save_model(LOCAL_MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(LOCAL_MODEL_OUTPUT_DIR)
    print("Model and tokenizer saved successfully.")


if __name__ == "__main__":
    main()