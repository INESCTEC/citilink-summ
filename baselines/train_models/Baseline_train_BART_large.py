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

# --- 🎯 CHANGED CHECKPOINT TO BART-LARGE ---
# BART-Large is the larger version (12 encoder layers, 12 decoder layers) of the model, 
# typically offering better performance but requiring more memory and computation time.
CHECKPOINT: str = "facebook/bart-large"

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
    
    Args:
        folder_path: The path to the directory containing the JSON dataset files.

    Returns:
        A pandas DataFrame where each row represents a training/validation segment
        with columns: 'document_id', 'text', 'resumo', and 'tema'.
    """
    all_segments: List[Dict[str, str]] = []
    
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            path = os.path.join(folder_path, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc: Dict[str, Any] = json.load(f)
                    
                if "segments" in doc and isinstance(doc["segments"], list):
                    for i, seg in enumerate(doc["segments"]):
                        if "resumo" not in seg or not seg.get("text"):
                            print(f"Warning: Skipping segment {i} in '{filename}' due to missing 'resumo' or 'text'.")
                            continue
                        
                        all_segments.append({
                            "document_id": doc.get("document_id", "N/A"),
                            "text": seg["text"],
                            "resumo": seg["resumo"],
                            "tema": seg.get("tema", "")
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
    Splits a long text into overlapping chunks based on token limits using a sliding window.
    
    Args:
        text: The input text to be chunked.
        tokenizer: The Hugging Face tokenizer instance.
        max_length: The maximum number of tokens per chunk.
        stride: The number of tokens to advance the window.

    Returns:
        A list of text strings, where each string is a chunk of the original text.
    """
    tokens: List[int] = tokenizer.encode(text, truncation=False)
    chunks: List[str] = []
    start: int = 0
    
    while start < len(tokens):
        end: int = min(start + max_length, len(tokens))
        chunk: str = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        chunks.append(chunk)
        
        if end == len(tokens):
            break
            
        start += stride
        
    return chunks


def prepare_dataset(df: pd.DataFrame) -> Dataset:
    """
    Processes the raw DataFrame by chunking long texts and combining chunks 
    into a single string for model input.

    Args:
        df: The input DataFrame containing 'text' and 'resumo' columns.

    Returns:
        A Hugging Face Dataset object with 'texto' (concatenated chunks) and 
        'sumario' (summary/label).
    """
    processed_rows: List[Dict[str, str]] = []
    
    for _, row in df.iterrows():
        # Chunk the text
        text_chunks: List[str] = chunk_text(str(row["text"]), tokenizer)
        
        # Concatenate all chunks back into one input text
        full_input: str = " ".join(text_chunks)
        
        processed_rows.append({
            "texto": full_input,
            "sumario": str(row["resumo"])
        })
        
    return Dataset.from_pandas(pd.DataFrame(processed_rows))


def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
    """
    Tokenizes the input text ('texto') and the target summary ('sumario')
    with appropriate truncation and padding.

    Args:
        examples: A batch of examples from the dataset.

    Returns:
        A dictionary containing tokenized inputs ('input_ids', 'attention_mask')
        and tokenized labels ('labels').
    """
    inputs: List[str] = [str(x) for x in examples["texto"]]
    targets: List[str] = [str(x) for x in examples["sumario"]]

    # Tokenize input text (source)
    model_inputs = tokenizer(
        inputs,
        max_length=CHUNK_MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Tokenize target summaries (labels)
    labels = tokenizer(
        targets,
        max_length=TARGET_MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Assign tokenized labels
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ==============================================================================
# 📊 Metrics and Evaluation
# ==============================================================================

# Load the ROUGE metric
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds: tuple) -> Dict[str, float]:
    """
    Computes the ROUGE scores for a batch of predictions against ground truth labels.

    Args:
        eval_preds: A tuple containing (predictions, labels) from the Trainer.

    Returns:
        A dictionary of ROUGE-1, ROUGE-2, and ROUGE-L scores (in percent).
    """
    preds, labels = eval_preds

    # Convert predictions (logits) to token IDs
    if isinstance(preds, tuple):
        preds = preds[0]
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    pred_ids: np.ndarray = np.argmax(preds, axis=-1)

    # Prepare labels for decoding (replace -100 with pad token ID)
    if isinstance(labels, tuple):
        labels = labels[0]
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
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
    print(f"✨ Starting BART-Large Summarization Model Setup and Training with checkpoint: {CHECKPOINT}")

    # --- 1. Data Loading ---
    train_df: pd.DataFrame = load_segments_from_folder(TRAIN_FOLDER)
    val_df: pd.DataFrame = load_segments_from_folder(VAL_FOLDER)
    print(f"Loaded {len(train_df)} training segments and {len(val_df)} validation segments.")

    # --- 2. Model and Tokenizer Initialization ---
    try:
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        # Note: BART-Large has significantly more parameters (approx 400M) than BART-Base (approx 140M),
        # so training will be slower and more memory-intensive.
        model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
    except Exception as e:
        print(f"Failed to load model or tokenizer: {e}")
        exit()

    # --- 3. Dataset Preparation ---
    train_dataset: Dataset = prepare_dataset(train_df)
    val_dataset: Dataset = prepare_dataset(val_df)

    # --- 4. Tokenization (Mapping) ---
    tokenized_train: Dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val: Dataset = val_dataset.map(preprocess_function, batched=True)
    print("Tokenization complete.")

    # --- 5. Training Setup ---
    # Adjust batch size for BART-Large if GPU memory is constrained (often reduced to 1 or 2)
    use_fp16: bool = torch.cuda.is_available()
    print(f"FP16 training enabled: {use_fp16}")

    training_args = TrainingArguments(
        output_dir="./results_bart_large_segments", # New directory name for BART-Large results
        eval_strategy="steps",
        eval_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=2 if use_fp16 else 1, # Potentially reduced batch size for BART-Large
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        logging_steps=10,
        fp16=use_fp16
    )

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
    FINAL_SAVE_PATH: str = "./trained_bart_large_segments"
    print(f"Saving final model to: {FINAL_SAVE_PATH}")
    trainer.save_model(FINAL_SAVE_PATH)
    tokenizer.save_pretrained(FINAL_SAVE_PATH)
    print("Model and tokenizer saved successfully.")

    print("✅ Deployment script execution finished.")