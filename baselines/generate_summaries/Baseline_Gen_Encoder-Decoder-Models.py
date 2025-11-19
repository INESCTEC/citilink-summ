import os
import json
from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Union, Tuple, Optional

# ===============================================================
# CONFIGURATION
# ===============================================================
# Relative path to the directory containing the original dataset's JSON files.
# Assumes the script is being run from a subfolder of the main directory.
DATASET_FOLDER: str = "../dataset"
# A base model used to determine a reasonable maximum token length.
BASE_CHECKPOINT: str = "allenai/primera"
# The size of the overlap (stride) between consecutive text chunks.
CHUNK_STRIDE: int = 256
# Fraction of the total dataset to reserve for validation.
VALIDATION_SPLIT: float = 0.1

# Dictionary mapping descriptive model names to their relative paths.
# Using relative paths facilitates deployment across different environments.
model_paths: Dict[str, str] = {
    "BART (Fine-tuned for CitiLink)": "../train_models/trained_bart_segments",
    "BART Large (Fine-tuned for citilink)": "../train_models/trained_bart_large_segments",
    "Primera (Fine-tuned for CitiLink)": "../train_models/trained_primera_segments",
    "PTT5": "../train_models/trained_ptt5_segments"
}

# ===============================================================
# UTILITIES
# ===============================================================

def chunk_text_dynamic(text: str, tokenizer: AutoTokenizer, max_length: int, stride: int = CHUNK_STRIDE) -> List[str]:
    """
    Splits a long text document into overlapping chunks, respecting the token limit.
    
    Args:
        text: The complete input text to be summarized.
        tokenizer: The model-specific tokenizer.
        max_length: The maximum context window size of the model (in tokens).
        stride: The size of the overlap (in tokens).
        
    Returns:
        A list of strings, where each string is a text chunk of the model's size.
    """
    # Tokenize the entire text without truncation
    tokens: List[int] = tokenizer.encode(text, truncation=False)
    chunks: List[str] = []
    start: int = 0
    
    while start < len(tokens):
        # Determine the end point, ensuring it doesn't exceed the max length
        end: int = min(start + max_length, len(tokens))
        
        # Decode the chunk back into a string
        chunk: str = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        chunks.append(chunk)
        
        # If the end of the token list was reached, break
        if end == len(tokens):
            break
            
        # Move the start pointer by the stride value (overlap)
        start += stride
        
    return chunks

def prepare_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer, model_max_len: int) -> List[Dict[str, Any]]:
    """
    Processes the raw DataFrame into a list of samples, including tokenized text chunks.
    
    Args:
        df: The input DataFrame containing segments to process.
        tokenizer: The model-specific tokenizer for chunking.
        model_max_len: The context window size for chunking.
        
    Returns:
        A list of dictionaries, each containing metadata and text chunks.
    """
    processed_rows: List[Dict[str, Any]] = []
    
    # Iterate over each segment in the validation/test set
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Splitting Dataset into Chunks"):
        # Split the text into overlapping chunks
        text_chunks: List[str] = chunk_text_dynamic(str(row["text"]), tokenizer, max_length=model_max_len)
        
        processed_rows.append({
            "chunks": text_chunks,
            "sumario": str(row["resumo"]), # Original summary (resumo)
            "titulo": row.get("tema", ""), # Original topic (tema)
            "segment_id": row["segment_id"],
            "text": row["text"],
            "source_file": row["source_file"],
            "municipality": row["municipality"],
            "year": row["year"]
        })
        
    return processed_rows

@torch.no_grad()
def load_model(name: str) -> Dict[str, Union[AutoModelForSeq2SeqLM, AutoTokenizer, int, torch.device]]:
    """
    Loads a Seq2Seq model and its tokenizer, determines the maximum context length,
    and moves the model to the appropriate device (CUDA or CPU).
    
    Args:
        name: The Hugging Face model ID or local path.
        
    Returns:
        A dictionary containing the model, tokenizer, max_length, and device.
    """
    # Use CUDA if available, otherwise fall back to CPU
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    # Load the model and ensure it's on the correct device
    model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.to(device)
    model.eval() # Set the model to evaluation mode
    
    # Determine the maximum context window length of the model (priority: config > tokenizer > default)
    max_length: int = getattr(model.config, 'max_position_embeddings', 
                              getattr(model.config, 'n_positions',
                                      getattr(tokenizer, 'model_max_length', 512)))
    
    return {"model": model, "tokenizer": tokenizer, "max_length": max_length, "device": device}

def safe_prompt(document: str, model_name: str) -> str:
    """
    Adds model-specific prefixes or instruction prompts if needed (e.g., T5).
    """
    if "T5" in model_name:
        if "flan" in model_name.lower():
            # Flan T5 often benefits from a direct instruction
            return "summarize the following text in Portuguese: " + document
        else:
            # Standard T5 prefix
            return "summarize: " + document
    # For BART, LED, Primera, etc., the document is passed directly
    return document

def get_generation_tokens(model_name: str) -> Tuple[int, int]:
    """
    Determines the maximum and minimum generation lengths based on the model type.
    """
    if "LED" in model_name or "Primera" in model_name:
        # Long-context models can generate longer summaries
        return 512, 40
    else:
        # Standard Seq2Seq models
        return 256, 40

# ---------------- MAIN EXECUTION ----------------
def main():
    # --- 1. Load All Data Segments ---
    all_segments: List[Dict[str, Any]] = []
    print(f"Loading data from {DATASET_FOLDER}...")
    
    # Iterate over all JSON files in the dataset folder
    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(".json"):
            path = os.path.join(DATASET_FOLDER, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    
                # Extract segment data along with metadata
                for seg in doc.get("segments", []):
                    all_segments.append({
                        "document_id": doc.get("document_id", "N/A"),
                        "segment_id": seg.get("segment_id", "N/A"),
                        "text": seg.get("text", ""),
                        "resumo": seg.get("resumo", ""),
                        "tema": seg.get("tema", ""),
                        "source_file": doc["metadata"].get("source_file", ""),
                        "municipality": doc["metadata"].get("municipality", ""),
                        "year": doc["metadata"].get("year", "")
                    })
            except Exception as e:
                print(f"Warning: Could not load or parse file {filename}: {e}")
                
    df: pd.DataFrame = pd.DataFrame(all_segments)
    print(f"Total segments loaded: {len(df)}")

    # --- 2. Create Validation Split ---
    if len(df) == 0:
        print("Dataset is empty. Exiting.")
        return

    # Use a fixed random state for reproducibility
    _, val_df = train_test_split(df, test_size=VALIDATION_SPLIT, random_state=42)
    print(f"Validation set size: {len(val_df)} segments ({VALIDATION_SPLIT * 100:.1f}% of total).")

    precomputed_data: List[Dict[str, Any]] = []

    # --- 3. Iterate over Models and Generate Summaries ---
    for model_name, model_path in model_paths.items():
        print("\n" + "="*40)
        print(f"=== Starting Generation for: {model_name} ===")
        print("="*40)
        
        # Load the model, tokenizer, and configuration
        try:
            bundle: Dict[str, Any] = load_model(model_path)
        except Exception as e:
            print(f"❌ Critical Error loading model {model_name} from {model_path}: {e}. Skipping.")
            continue
            
        model: AutoModelForSeq2SeqLM = bundle["model"]
        tokenizer: AutoTokenizer = bundle["tokenizer"]
        max_length: int = bundle["max_length"]
        device: torch.device = bundle["device"]

        # Print debug info about model-specific tokens
        try:
            cfg = model.config
            print(f"[MODEL TOKENS] {model_name} -> max_len:{max_length}, bos:{getattr(tokenizer, 'bos_token_id', None)}, eos:{getattr(tokenizer, 'eos_token_id', None)}, pad:{getattr(tokenizer, 'pad_token_id', None)}, decoder_start:{getattr(cfg, 'decoder_start_token_id', None)}")
        except Exception as e:
            print(f"[MODEL TOKENS] Could not read token IDs for {model_name}: {e}")
        
        # Get generation constraints (max/min tokens)
        max_new, min_new = get_generation_tokens(model_name)
        print(f"Generation Limits: max_new_tokens={max_new}, min_new_tokens={min_new}")

        # Prepare the dataset: chunk the validation data according to the current model's max_length
        val_data: List[Dict[str, Any]] = prepare_dataset(val_df, tokenizer, model_max_len=max_length)
        print(f"Data prepared into chunks for {model_name}.")

        # Iterate over validation data to generate summaries for each model
        for i, item in enumerate(tqdm(val_data, desc=f"Generating with {model_name}")):
            
            # Initialize or retrieve the entry for this segment in the precomputed_data list
            if len(precomputed_data) <= i:
                precomputed_data.append({
                    "segment_id": item["segment_id"],
                    "text": item["text"],
                    "resumo": item["sumario"],
                    "tema": item["titulo"],
                    "source_file": item["source_file"],
                    "municipality": item["municipality"],
                    "year": item["year"],
                    "generated_summaries": {}
                })

            summaries: List[str] = []
            
            # --- Chunk-by-Chunk Generation and Aggregation ---
            for chunk in item["chunks"]:
                document: str = safe_prompt(chunk, model_name)

                # Tokenize the current chunk
                inputs: Dict[str, torch.Tensor] = tokenizer(
                    document,
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)

                # Ensure generation IDs are correctly set for robustness
                pad_id: Optional[int] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                eos_id: Optional[int] = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

                # Use model config for decoder_start_token_id, otherwise tokenizer's BOS
                decoder_start: Optional[int] = getattr(model.config, 'decoder_start_token_id', None)
                if decoder_start is None:
                    decoder_start = getattr(tokenizer, 'bos_token_id', None)

                # Generate the summary for the current chunk
                try:
                    summary_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        min_new_tokens=min_new,
                        do_sample=False,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                        decoder_start_token_id=decoder_start,
                    )

                    out: str = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
                    
                    # Quick debug check: log short outputs to help diagnose generation issues
                    if len(out.split()) <= 2:
                        print(f"\n[DEBUG] Short generation observed for model: {model_name}, segment ID: {item['segment_id']}")
                            
                    summaries.append(out)
                    
                except Exception as gen_e:
                    print(f"\n[ERROR] Generation failed for segment {item['segment_id']} chunk: {gen_e}")
                    summaries.append("") # Add empty string on failure to maintain list integrity

            # Join all chunk summaries to form the full document summary
            full_summary: str = " ".join(s for s in summaries if s) # Join only non-empty summaries
            precomputed_data[i]["generated_summaries"][model_name] = full_summary

        # Clear GPU memory before loading the next model
        del model, tokenizer, bundle
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    # --- 4. Save Results ---
    output_path: str = "./val_precomputed_all_models_dynamic.json"
    print("\n" + "="*40)
    print("=== Saving Final Results ===")
    print("="*40)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(precomputed_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Validation summaries for all models successfully saved to {output_path}")

if __name__ == "__main__":
    main()