import os
import json
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any, Union

# Third-party libraries
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import evaluate


# ==============================================================================
# 🚀 Configuration Constants
# ==============================================================================

# Define paths for the dataset
TRAIN_FOLDER: str = "/mnt/c/Users/migue/Desktop/citilink_summarization/dataset_split/train"
VAL_FOLDER: str = "/mnt/c/Users/migue/Desktop/citilink_summarization/dataset_split/val"

# Model Checkpoint
CHECKPOINT: str = "facebook/bart-base"

# Tokenization and Model-specific parameters
# BART max input length is 1024.
CHUNK_MAX_LENGTH: int = 1024
# Stride for overlapping chunks, typically half the max length for context retention.
CHUNK_STRIDE: int = 512
# Maximum length for the generated summary (target label)
TARGET_MAX_LENGTH: int = 128


# ==============================================================================
# 💾 Data Loading and Preparation Functions
# ==============================================================================

def load_segments_from_folder(folder_path: str) -> pd.DataFrame:
    """
    Loads text and summary segments from all JSON files in a specified folder.

    The function iterates through all JSON files, extracts 'text' and 'resumo' 
    (summary) from each segment, and compiles them into a DataFrame.

    Args:
        folder_path: The path to the directory containing the JSON dataset files.

    Returns:
        A pandas DataFrame where each row represents a training/validation segment
        with columns: 'document_id', 'text', 'resumo', and 'tema'.
    """
    all_segments: List[Dict[str, str]] = []
    
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            path = os.path.join(folder_path, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc: Dict[str, Any] = json.load(f)
                    
                # Ensure 'segments' key exists and is iterable
                if "segments" in doc and isinstance(doc["segments"], list):
                    for i, seg in enumerate(doc["segments"]):
                        # Data Validation: Ensure the required 'resumo' key is present
                        if "resumo" not in seg or not seg.get("text"):
                            print(f"Warning: Skipping segment {i} in '{filename}' due to missing 'resumo' or 'text'.")
                            continue
                        
                        all_segments.append({
                            "document_id": doc.get("document_id", "N/A"),
                            "text": seg["text"],
                            "resumo": seg["resumo"],
                            "tema": seg.get("tema", "") # 'tema' is optional
                        })
                else:
                    print(f"Warning: '{filename}' does not contain a 'segments' list.")

            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON in file: {filename}")
            except Exception as e:
                print(f"An unexpected error occurred while processing '{filename}': {e}")

    return pd.DataFrame(all_segments)


def chunk_text(text: str, tokenizer: AutoTokenizer, max_length: int = CHUNK_MAX_LENGTH, stride: int = CHUNK_STRIDE) -> List[str]:
    """
    Splits a long text into overlapping chunks based on token limits.
    This uses a sliding window approach for better context preservation.

    Args:
        text: The input text to be chunked.
        tokenizer: The Hugging Face tokenizer instance.
        max_length: The maximum number of tokens per chunk (includes special tokens).
        stride: The number of tokens to advance the window for the next chunk (overlap is max_length - stride).

    Returns:
        A list of text strings, where each string is a chunk of the original text.
    """
    # Tokenize the whole text without truncation to get raw token IDs
    tokens: List[int] = tokenizer.encode(text, truncation=False)
    chunks: List[str] = []
    start: int = 0
    
    # Iterate through the tokens using a sliding window (stride)
    while start < len(tokens):
        end: int = min(start + max_length, len(tokens))
        
        # Decode the token IDs back into a text string chunk
        chunk: str = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        chunks.append(chunk)
        
        # If the end of the text is reached, stop
        if end == len(tokens):
            break
            
        # Move the window forward by the stride
        start += stride
        
    return chunks


def prepare_dataset(df: pd.DataFrame) -> Dataset:
    """
    Processes the raw DataFrame by chunking long texts and combining chunks 
    into a single string, preparing it for tokenization.

    Args:
        df: The input DataFrame containing 'text' and 'resumo' columns.

    Returns:
        A Hugging Face Dataset object with 'texto' (concatenated chunks) and 
        'sumario' (summary/label).
    """
    processed_rows: List[Dict[str, str]] = []
    
    # Iterate over the DataFrame rows (segments)
    for _, row in df.iterrows():
        # 1. Chunk the input text
        text_chunks: List[str] = chunk_text(str(row["text"]), tokenizer)
        
        # 2. Concatenate all chunks back into one input text for the model
        # This is a common approach for handling long documents in summarization, 
        # though it means the model only sees the concatenated text, not the 
        # structure of individual chunks directly in the input.
        full_input: str = " ".join(text_chunks)
        
        processed_rows.append({
            "texto": full_input,
            "sumario": str(row["resumo"])
        })
        
    # Convert the list of dicts into a Hugging Face Dataset
    return Dataset.from_pandas(pd.DataFrame(processed_rows))


def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
    """
    Tokenizes the input text ('texto') and the target summary ('sumario').

    Args:
        examples: A batch of examples from the dataset, typically with keys 
                  'texto' (input) and 'sumario' (target).

    Returns:
        A dictionary containing tokenized inputs ('input_ids', 'attention_mask')
        and tokenized labels ('labels').
    """
    inputs: List[str] = [str(x) for x in examples["texto"]]
    targets: List[str] = [str(x) for x in examples["sumario"]]

    # Tokenize the input text, truncating to CHUNK_MAX_LENGTH (1024)
    model_inputs = tokenizer(
        inputs,
        max_length=CHUNK_MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Tokenize the target summaries (labels), truncating to TARGET_MAX_LENGTH (128)
    # The 'with tokenizer.as_target_tokenizer()' context manager is now deprecated;
    # it's recommended to tokenize labels separately and apply label masking.
    labels = tokenizer(
        targets,
        max_length=TARGET_MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Replace padding tokens in the labels with -100 so they are ignored by the loss function
    # Note: DataCollatorForSeq2Seq handles this for the padding it introduces, 
    # but it's good practice for pre-tokenized padded labels as well.
    labels_input_ids: List[List[int]] = labels["input_ids"]
    
    # Standard practice for labels in seq2seq models: assign the tokenized labels
    model_inputs["labels"] = labels_input_ids
    return model_inputs


# ==============================================================================
# 📊 Metrics and Evaluation
# ==============================================================================

# Load the ROUGE metric from the Hugging Face 'evaluate' library
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds: tuple) -> Dict[str, float]:
    """
    Computes the ROUGE scores for a batch of predictions.

    Args:
        eval_preds: A tuple containing (predictions, labels).

    Returns:
        A dictionary of ROUGE-1, ROUGE-2, and ROUGE-L scores (in percent).
    """
    # Unpack predictions and labels
    preds, labels = eval_preds

    # Handle case where Trainer returns a nested tuple (e.g., when logging losses)
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Convert predictions to numpy array
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    
    # Get the token IDs by selecting the index of the maximum logit (greedy decoding)
    # Note: For generation, this is typically done using model.generate() in the Trainer callback,
    # but this handles the standard metric calculation on raw logits.
    pred_ids: np.ndarray = np.argmax(preds, axis=-1)

    # Handle labels
    if isinstance(labels, tuple):
        labels = labels[0]
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    
    # Replace the special label -100 (ignored loss) with the tokenizer's padding ID
    # This is necessary before decoding, as -100 is not a valid token ID.
    labels: np.ndarray = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels
    decoded_preds: List[str] = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels: List[str] = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scores
    result: Dict[str, float] = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Format the results to percentages and round to 2 decimal places
    return {k: round(v * 100, 2) for k, v in result.items()}


# ==============================================================================
# 🧠 Main Execution: Load, Preprocess, and Train
# ==============================================================================

if __name__ == "__main__":
    print("✨ Starting Summarization Model Setup and Training...")

    # --- 1. Data Loading ---
    print(f"Loading training data from: {TRAIN_FOLDER}")
    train_df: pd.DataFrame = load_segments_from_folder(TRAIN_FOLDER)
    print(f"Loaded {len(train_df)} training segments.")

    print(f"Loading validation data from: {VAL_FOLDER}")
    val_df: pd.DataFrame = load_segments_from_folder(VAL_FOLDER)
    print(f"Loaded {len(val_df)} validation segments.")

    # --- 2. Model and Tokenizer Initialization ---
    print(f"Initializing Tokenizer and Model from checkpoint: {CHECKPOINT}")
    try:
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
    except Exception as e:
        print(f"Failed to load model or tokenizer: {e}")
        exit()

    # --- 3. Dataset Preparation (Chunking and Concatenation) ---
    print("Preparing datasets (chunking and concatenating text)...")
    train_dataset: Dataset = prepare_dataset(train_df)
    val_dataset: Dataset = prepare_dataset(val_df)
    print(f"Training dataset size (after preparation): {len(train_dataset)}")
    print(f"Validation dataset size (after preparation): {len(val_dataset)}")

    # --- 4. Tokenization (Mapping) ---
    print("Tokenizing datasets...")
    # Apply the preprocessing function (tokenization and padding) to the datasets
    tokenized_train: Dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val: Dataset = val_dataset.map(preprocess_function, batched=True)
    print("Tokenization complete.")

    # --- 5. Training Setup ---
    print("Configuring Training Arguments and Trainer...")
    
    # Configure GPU usage if available
    use_fp16: bool = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir="./results_bart_segments", # Directory for checkpoints and logs
        eval_strategy="steps",              # Evaluation is performed every 'eval_steps'
        eval_steps=100,                     # Run evaluation every 100 steps
        learning_rate=2e-5,                 # Standard learning rate for fine-tuning
        per_device_train_batch_size=4,      # Batch size per GPU/CPU
        num_train_epochs=3,                 # Number of training epochs
        weight_decay=0.01,                  # L2 regularization
        save_total_limit=1,                 # Only save the best checkpoint (based on evaluation metric)
        logging_steps=10,                   # Log training loss every 10 steps
        fp16=use_fp16                       # Enable half-precision training if CUDA is available
    )

    # A special data collator that correctly prepares batches for Sequence-to-Sequence models
    data_collator: DataCollatorForSeq2Seq = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # --- 6. Training ---
    print("--- Starting Model Fine-Tuning ---")
    trainer.train()
    print("--- Training Complete ---")

    # --- 7. Save Final Model and Tokenizer ---
    FINAL_SAVE_PATH: str = "./trained_bart_segments"
    print(f"Saving final model to: {FINAL_SAVE_PATH}")
    trainer.save_model(FINAL_SAVE_PATH)
    tokenizer.save_pretrained(FINAL_SAVE_PATH)
    print("Model and tokenizer saved successfully.")

    print("✅ Deployment script execution finished.")