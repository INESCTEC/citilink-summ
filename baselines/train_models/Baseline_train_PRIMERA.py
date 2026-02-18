import os
import json
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any, Union
import argparse

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

# PRIMERA script will read a single citilink JSON file supplied via CLI

# Model Checkpoint (PRIMERA is based on LongT5)
CHECKPOINT: str = "allenai/PRIMERA"

# Tokenization and Model-specific parameters
CHUNK_MAX_LENGTH: int = 1024
CHUNK_STRIDE: int = 512
TARGET_MAX_LENGTH: int = 128
# PRIMERA requires a task prefix for summarization
# The prompt is generally "summarize: " but your local variable is "resume: " (Portuguese)
# We will use the common English prefix for better compatibility, but you can revert it if needed.
PRIMERA_PREFIX: str = "summarize: " 
# Note: If your model was trained with 'resume: ', stick to that, 
# but 'summarize: ' is the standard for the English checkpoint.

# ==============================================================================
# 💾 Data Loading and Preparation Functions
# ==============================================================================

# dataset loader reused from the other scripts

def load_citilink(filepath: str) -> pd.DataFrame:
    """
    Read a consolidated citilink JSON and return a flat DataFrame.

    The dataset file is expected to use the CitiLink-Summ format. In brief, it
    contains a top-level ``municipalities`` list, each entry holding a
    ``minutes`` list, and each minute containing an ``agenda_items`` list.
    Only those items with both ``text`` and ``summary`` are loaded.

    Args:
        filepath: path to the single JSON dataset file.

    Returns:
        DataFrame with columns ``texto`` and ``sumario`` ready for further
        processing (chunking/tokenization) in this script.
    """
    all_rows: List[Dict[str, str]] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover
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
    Processes the raw DataFrame by chunking long texts. Each chunk is paired
    with the original summary (``sumario``).

    Args:
        df: The input DataFrame containing ``texto`` and ``sumario`` columns.

    Returns:
        A Hugging Face Dataset object with ``texto`` (chunk) and ``sumario``
        (summary/label).
    """
    processed_rows: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        text_chunks: List[str] = chunk_text(str(row["texto"]), tokenizer)
        for chunk in text_chunks:
            processed_rows.append({
                "texto": chunk,
                "sumario": str(row["sumario"])
            })

    return Dataset.from_pandas(pd.DataFrame(processed_rows))


def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
    """
    Tokenizes the input text (with PRIMERA prefix) and the target summary.

    Args:
        examples: A batch of examples from the dataset.

    Returns:
        A dictionary containing tokenized inputs and labels.
    """
    # Apply the required PRIMERA prefix
    inputs: List[str] = [PRIMERA_PREFIX + str(x) for x in examples["texto"]]
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
    
    # Note: PRIMERA is an Encoder-Decoder model. We should use generate() 
    # for proper decoding during validation, but this function runs 
    # on logits. We'll use argmax for simple decoding here, but for 
    # better ROUGE, generation should be configured in TrainingArguments.
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
    parser = argparse.ArgumentParser(
        description="Fine-tune PRIMERA on the citilink summarization dataset"
    )
    parser.add_argument("input", help="path to citilink JSON file")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed used for splitting (60/20/20)")
    parser.add_argument("--output-dir", default="./results_primera_segments",
                        help="directory where model checkpoints/logs are written")
    parser.add_argument("--epochs", type=int, default=3,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="per-device train batch size")
    args = parser.parse_args()

    print(f"✨ Starting PRIMERA Summarization Model Setup and Training with checkpoint: {CHECKPOINT}")

    # --- 1. Data Loading & Split ---
    df: pd.DataFrame = load_citilink(args.input)
    print(f"loaded {len(df)} agenda items from {args.input}")

    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_total = len(df)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]
    print(f"split into {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    # --- 2. Model and Tokenizer Initialization ---
    try:
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
        model.gradient_checkpointing_enable()
    except Exception as e:
        print(f"Failed to load model or tokenizer: {e}")
        exit()

    # --- 3. Dataset Preparation ---
    train_dataset: Dataset = prepare_dataset(train_df)
    val_dataset: Dataset = prepare_dataset(val_df)
    print(f"Training dataset size (chunks): {len(train_dataset)}")
    print(f"Validation dataset size (chunks): {len(val_dataset)}")

    # --- 4. Tokenization (Mapping) ---
    tokenized_train: Dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val: Dataset = val_dataset.map(preprocess_function, batched=True)
    print("Tokenization complete.")

    # --- 5. Training Setup ---
    use_fp16: bool = torch.cuda.is_available()
    print(f"FP16 training enabled: {use_fp16}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=1,
        logging_steps=20,
        fp16=use_fp16,
        gradient_accumulation_steps=4
    )

    data_collator: DataCollatorForSeq2Seq = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # --- 6. Training ---
    print("--- Starting Model Fine-Tuning ---")
    trainer.train()
    print("--- Training Complete ---")

    # --- 7. Save Final Model and Tokenizer ---
    FINAL_SAVE_PATH: str = os.path.join(args.output_dir, "final")
    print(f"Saving final model to: {FINAL_SAVE_PATH}")
    trainer.save_model(FINAL_SAVE_PATH)
    tokenizer.save_pretrained(FINAL_SAVE_PATH)
    print("Model and tokenizer saved successfully.")

    print("✅ Deployment script execution finished.")