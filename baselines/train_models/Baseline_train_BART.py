# baseline_train_BART.py
# --------------------
# This script fine‑tunes a BART model on the CitiLink-Summ dataset.


import os
import json
import argparse
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

# The training data is expected in a single JSON file containing citilink
# records.  During execution the file is randomly partitioned 60/20/20 into
# training, validation and test subsets.

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
# Data Loading and Preparation Functions
# ==============================================================================

def load_citilink(filepath: str) -> pd.DataFrame:
    """Read a consolidated citilink JSON and return a flat DataFrame.

    The dataset file is expected to use the CitiLink-Summ format. In brief, it contains a top-level
    "municipalities" list, each entry holding a "minutes" list, and each
    minute containing an "agenda_items" list.  Only those items with both
    "text" and "summary" are loaded.

        {
            "municipalities": [
                {"municipality": ..., "minutes": [
                    {"minute_id": ..., "agenda_items": [
                        {"text": ..., "summary": ..., ...},
                        ...
                    ]},
                    ...
                ]},
                ...
            ]
        }

    We simply iterate over every agenda item and collect the ``text`` and
    ``summary`` fields.  Rows lacking either are skipped.

    Args:
        filepath: path to the single JSON dataset file.

    Returns:
        DataFrame with columns ``texto`` and ``sumario`` ready for
        further processing (chunking/tokenization) in this script.
    """
    all_rows: List[Dict[str, str]] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - basic error handling
        print(f"Unable to load {filepath}: {exc}")
        return pd.DataFrame()

    for muni in data.get("municipalities", []):
        for minute in muni.get("minutes", []):
            for item in minute.get("agenda_items", []):
                text = item.get("text")
                summary = item.get("summary")
                if not text or not summary:
                    continue
                all_rows.append({"texto": text, "sumario": summary})

    return pd.DataFrame(all_rows)


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

    The incoming ``df`` should already be produced by ``load_citilink`` so it
    contains two columns named ``texto`` and ``sumario``.  These correspond to
    the original meeting text and its human‑written summary, respectively.

    Args:
        df: The input DataFrame containing ``texto`` and ``sumario`` columns.

    Returns:
        A Hugging Face ``Dataset`` object with the same ``texto`` and ``sumario``
        fields, where ``texto`` has been chunked and concatenated for easier
        tokenization.
    """
    processed_rows: List[Dict[str, str]] = []

    # Iterate over the DataFrame rows (segments)
    for _, row in df.iterrows():
        # 1. Chunk the input text using the 'texto' column
        text_chunks: List[str] = chunk_text(str(row["texto"]), tokenizer)

        # 2. Concatenate all chunks back into one input text for the model
        # This is a common approach for handling long documents in summarization,
        # though it means the model only sees the concatenated text, not the
        # structure of individual chunks directly in the input.
        full_input: str = " ".join(text_chunks)

        processed_rows.append({
            "texto": full_input,
            "sumario": str(row["sumario"])
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
#  Metrics and Evaluation
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
    # parse command-line arguments so the script can be run directly or via
    # ``python baseline_train_BART.py /path/to/citilink.json``.  the defaults
    # are chosen to work reasonably well for a small sample but users can
    # override them when running on the full dataset.
    parser = argparse.ArgumentParser(
        description="Fine-tune BART on the citilink summarization dataset"
    )
    parser.add_argument("input", help="path to citilink JSON file")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed used for splitting (60/20/20)")
    parser.add_argument("--output-dir", default="./results_bart_segments",
                        help="directory where model checkpoints/logs are written")
    parser.add_argument("--epochs", type=int, default=3,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="per-device train batch size")
    args = parser.parse_args()

    print("✨ Starting Summarization Model Setup and Training...")
    print(f"training configuration: epochs={args.epochs}, batch_size={args.batch_size}, seed={args.seed}")

    # -------------------------------------------------------------
    # 1. Data Loading & Random Split (60/20/20)
    # -------------------------------------------------------------
    df: pd.DataFrame = load_citilink(args.input)
    print(f"loaded {len(df)} agenda items from {args.input}")

    # random shuffle with optional seed for reproducibility
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_total = len(df)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    print(f"split into {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

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
    test_dataset: Dataset = prepare_dataset(test_df)
    print(f"Training dataset size (after preparation): {len(train_dataset)}")
    print(f"Validation dataset size (after preparation): {len(val_dataset)}")
    print(f"Test dataset size (after preparation): {len(test_dataset)}")

    # --- 4. Tokenization (Mapping) ---
    print("Tokenizing datasets...")
    # Apply the preprocessing function (tokenization and padding) to the datasets
    tokenized_train: Dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val: Dataset = val_dataset.map(preprocess_function, batched=True)
    tokenized_test: Dataset = test_dataset.map(preprocess_function, batched=True)
    print("Tokenization complete.")

    # --- 5. Training Setup ---
    print("Configuring Training Arguments and Trainer...")
    
    # Configure GPU usage if available
    use_fp16: bool = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=args.output_dir,              # Directory for checkpoints and logs (from CLI)
        eval_strategy="steps",              # Evaluation is performed every 'eval_steps'
        eval_steps=100,                     # Run evaluation every 100 steps
        learning_rate=2e-5,                 # Standard learning rate for fine-tuning
        per_device_train_batch_size=args.batch_size,      # Batch size per GPU/CPU
        num_train_epochs=args.epochs,                 # Number of training epochs
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
    # place the final model/tokenizer inside the same output directory
    FINAL_SAVE_PATH: str = os.path.join(args.output_dir, "final")
    print(f"Saving final model to: {FINAL_SAVE_PATH}")
    trainer.save_model(FINAL_SAVE_PATH)
    tokenizer.save_pretrained(FINAL_SAVE_PATH)
    print("Model and tokenizer saved successfully.")

    print("✅ Deployment script execution finished.")