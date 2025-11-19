import os
import json
import numpy as np
import pandas as pd
import torch
import evaluate
from typing import List, Dict, Any, Union

# Third-party libraries
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq, # Used as a base reference
)
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


# ==============================================================================
# 🚀 Configuration Constants
# ==============================================================================

# Define paths for the dataset
TRAIN_FOLDER: str = "/mnt/c/Users/migue/Desktop/citilink_summarization/dataset_split/train"
VAL_FOLDER: str = "/mnt/c/Users/migue/Desktop/citilink_summarization/dataset_split/val" # 🎯 ADDED VALIDATION FOLDER

# Model Checkpoint
# LED model with max input length of 16384 tokens
CHECKPOINT: str = "allenai/led-base-16384"

# Tokenization and Model-specific parameters
CHUNK_MAX_LENGTH: int = 1024 # Chunk size for a single training example
CHUNK_STRIDE: int = 512
TARGET_MAX_LENGTH: int = 128
OUTPUT_DIR: str = "./results_led_segments" # Consistent output directory name


# ==============================================================================
# 💾 Data Loading and Preparation Functions
# ==============================================================================

def load_segments_from_folder(folder_path: str) -> pd.DataFrame:
    """
    Loads text and summary segments from all JSON files in a specified folder.
    
    Args:
        folder_path: The path to the directory containing the JSON dataset files.

    Returns:
        A pandas DataFrame where each row represents a segment.
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
                    for seg in doc["segments"]:
                        if "resumo" not in seg or not seg.get("text"):
                            continue
                        
                        all_segments.append({
                            "document_id": doc.get("document_id", "N/A"),
                            "text": seg["text"],
                            "resumo": seg["resumo"],
                            "tema": seg.get("tema", "")
                        })
            except Exception as e:
                print(f"Error processing file '{filename}': {e}")

    return pd.DataFrame(all_segments)


def chunk_text_with_global_attention(text: str, tokenizer: AutoTokenizer, max_length: int = CHUNK_MAX_LENGTH, stride: int = CHUNK_STRIDE) -> tuple[List[List[int]], List[List[int]]]:
    """
    Splits a long text into overlapping chunks of token IDs and generates 
    the corresponding global attention mask for LED.

    Args:
        text: The input text string.
        tokenizer: The Hugging Face tokenizer instance.
        max_length: The maximum number of tokens per chunk.
        stride: The number of tokens to advance the window.

    Returns:
        A tuple: (list of token ID lists (chunks), list of global attention mask lists).
    """
    tokens: List[int] = tokenizer.encode(text, truncation=False)
    chunks: List[List[int]] = []
    global_attention_masks: List[List[int]] = []
    start: int = 0
    
    while start < len(tokens):
        end: int = min(start + max_length, len(tokens))
        chunk_tokens: List[int] = tokens[start:end]
        
        # 1. Create the Global Attention Mask
        # Standard practice: set global attention only on the first token of the chunk [1, 0, 0, ...]
        # This forces the model to attend to the start of the segment.
        global_mask: List[int] = [0] * len(chunk_tokens)
        if len(global_mask) > 0:
            global_mask[0] = 1 # The first token receives global attention
            
        chunks.append(chunk_tokens)
        global_attention_masks.append(global_mask)
        
        if end == len(tokens):
            break
            
        start += stride
        
    return chunks, global_attention_masks


def prepare_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer, desc: str) -> Dataset:
    """
    Tokenizes the entire dataset, applies chunking and global attention logic, 
    and handles padding/truncation to create a Hugging Face Dataset.
    
    Args:
        df: The input DataFrame containing 'text' and 'resumo' columns.
        tokenizer: The initialized tokenizer.
        desc: Description for the tqdm progress bar.

    Returns:
        A Hugging Face Dataset object.
    """
    input_ids_list, attention_masks_list, global_attention_masks_list, labels_list = [], [], [], []

    # Process all segments/rows
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        # 1. Chunk the input text and generate global masks
        chunks_tokens, global_masks = chunk_text_with_global_attention(
            str(row["text"]), tokenizer, CHUNK_MAX_LENGTH, CHUNK_STRIDE
        )
        
        # 2. Tokenize the target summary (label)
        target: List[int] = tokenizer.encode(
            str(row["resumo"]), 
            truncation=True, 
            max_length=TARGET_MAX_LENGTH
        )
        
        # 3. Extend lists for each chunk
        for chunk, global_mask in zip(chunks_tokens, global_masks):
            input_ids_list.append(chunk)
            global_attention_masks_list.append(global_mask)
            attention_masks_list.append([1] * len(chunk))
            
            # Use the same target for all chunks of a single segment
            labels_list.append(target)
            
    # --- Padding and Tensor Creation ---
    
    # Pad to the length of the longest example in this batch
    max_len_input: int = max(len(x) for x in input_ids_list) if input_ids_list else 0
    max_len_label: int = max(len(x) for x in labels_list) if labels_list else 0
    
    padded_data = {
        "input_ids": [],
        "attention_mask": [],
        "global_attention_mask": [],
        "labels": []
    }

    # Apply padding to inputs
    for input_ids, attention_mask, global_mask in zip(input_ids_list, attention_masks_list, global_attention_masks_list):
        pad_len = max_len_input - len(input_ids)
        padded_data["input_ids"].append(input_ids + [tokenizer.pad_token_id] * pad_len)
        padded_data["attention_mask"].append(attention_mask + [0] * pad_len)
        padded_data["global_attention_mask"].append(global_mask + [0] * pad_len)

    # Apply padding to labels
    for labels in labels_list:
        pad_len = max_len_label - len(labels)
        # Pad labels with -100 so the loss function ignores them
        padded_labels = labels + [-100] * pad_len
        
        # Ensure label IDs are valid (not special tokens outside normal vocab range if any)
        # Note: -100 is handled by the loss function, but we ensure other tokens are correct
        processed_labels = [t if t >= 0 else -100 for t in padded_labels]
        padded_data["labels"].append(processed_labels)

    return Dataset.from_dict(padded_data)


# ==============================================================================
# 📦 Custom Data Collator
# ==============================================================================

class LEDDataCollator:
    """
    A custom data collator for LED to ensure correct conversion to PyTorch 
    tensors for input_ids, attention_mask, global_attention_mask, and labels.
    
    Since the dataset is pre-padded, this collator primarily converts lists to tensors.
    """
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        
        # Convert pre-padded lists to PyTorch tensors
        input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
        attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
        global_attention_mask = torch.tensor([b["global_attention_mask"] for b in batch], dtype=torch.long)
        labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask, 
            "labels": labels
        }


# ==============================================================================
# 📊 Metrics and Evaluation
# ==============================================================================

rouge = evaluate.load("rouge")

def compute_metrics(eval_preds: tuple) -> Dict[str, float]:
    """
    Computes the ROUGE scores for a batch of predictions.
    
    Args:
        eval_preds: A tuple containing (predictions, labels) from the Trainer.

    Returns:
        A dictionary of ROUGE-1, ROUGE-2, and ROUGE-L scores (in percent).
    """
    preds, labels = eval_preds

    # If preds is logits (default for compute_metrics), select the max logit index (token ID)
    if isinstance(preds, tuple):
        preds = preds[0]
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
        
    # Get the token IDs by selecting the index of the maximum logit
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
    print(f"✨ Starting LED Summarization Model Setup with checkpoint: {CHECKPOINT}")

    # --- 1. Data Loading ---
    train_df: pd.DataFrame = load_segments_from_folder(TRAIN_FOLDER)
    val_df: pd.DataFrame = load_segments_from_folder(VAL_FOLDER)
    print(f"Loaded {len(train_df)} training segments and {len(val_df)} validation segments.")

    # --- 2. Model and Tokenizer Initialization ---
    try:
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
        # Enable gradient checkpointing to save GPU memory, essential for large models like LED
        model.gradient_checkpointing_enable() 
    except Exception as e:
        print(f"Failed to load model or tokenizer: {e}")
        exit()

    # --- 3. Dataset Preparation (Chunking, Global Attention, and Tokenization) ---
    print("Preparing and tokenizing training dataset...")
    train_dataset: Dataset = prepare_dataset(train_df, tokenizer, desc="Preparing train dataset")
    print("Preparing and tokenizing validation dataset...")
    val_dataset: Dataset = prepare_dataset(val_df, tokenizer, desc="Preparing validation dataset")
    
    print(f"Training dataset size (chunks): {len(train_dataset)}")
    print(f"Validation dataset size (chunks): {len(val_dataset)}")

    # --- 4. Training Setup ---
    use_fp16: bool = torch.cuda.is_available()
    print(f"FP16 training enabled: {use_fp16}")

    # LED is a large model, so use small batch size and gradient accumulation.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",              # 🎯 Evaluate every 'eval_steps'
        eval_steps=100,                     # Run evaluation every 100 steps
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,        # Load the model with the best validation metric
        metric_for_best_model="rougeL",     # Use ROUGE-L to determine the best model
        
        learning_rate=2e-5,
        per_device_train_batch_size=1,      # Extremely low batch size due to model size
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        logging_steps=20,
        fp16=use_fp16,
        gradient_accumulation_steps=4       # Accumulate gradients over 4 batches
    )

    data_collator = LEDDataCollator(tokenizer)

    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,           # 🎯 ADD VALIDATION DATASET
        compute_metrics=compute_metrics,    # 🎯 ADD METRICS
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # --- 5. Training ---
    print("--- Starting Model Fine-Tuning ---")
    trainer.train()
    print("--- Training Complete ---")

    # --- 6. Save Final Model and Tokenizer ---
    print(f"Saving final model to: {OUTPUT_DIR}")
    # The best model based on validation metrics is automatically loaded and saved here
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model and tokenizer saved successfully.")

    print("✅ Deployment script execution finished.")