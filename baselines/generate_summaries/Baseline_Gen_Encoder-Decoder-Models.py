import os
import json
import argparse
from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Union, Tuple, Optional

# the loader below mirrors the one used in training scripts

# ===============================================================
# CONFIGURATION
# ===============================================================
# Path to the aggregated citilink_summ JSON file will be supplied on the command line
# A base model used to determine a reasonable maximum token length (unused)
BASE_CHECKPOINT: str = "allenai/primera"
# The size of the overlap (stride) between consecutive text chunks.
CHUNK_STRIDE: int = 256
# default fraction of the dataset to reserve for validation (overridable via CLI)
VALIDATION_SPLIT: float = 0.1

# Dictionary mapping descriptive model names to their relative paths.
# Using relative paths facilitates deployment across different environments.
model_paths: Dict[str, str] = {
    # Use the relative paths you provided (relative to this script folder)
    "BART (Fine-tuned for CitiLink)": "../train_models/results_bart_segments/final",
    "BART Large (Fine-tuned for citilink)": "../train_models/results_bart_large_segments/final",
    "LED (Fine-tuned for CitiLink)": "../train_models/results_led_segments",
    "Primera (Fine-tuned for CitiLink)": "../train_models/results_primera_segments/final",
    "PTT5": "../train_models/trained_ptt5_segments"
}

# ===============================================================
# UTILITIES
# ===============================================================

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
        muni_name = muni.get("municipality", "")
        for minute in muni.get("minutes", []):
            # minute-level metadata
            minute_source = minute.get("source_file", "") or minute.get("file", "")
            # try to extract a year if provided at minute-level
            minute_year = minute.get("year", "")
            for item in minute.get("agenda_items", []):
                text = item.get("text") or item.get("texto")
                summary = item.get("summary") or item.get("resumo")
                if not text or not summary:
                    continue

                # item-level metadata (use sensible fallbacks)
                segment_id = item.get("segment_id") or item.get("id") or ""
                tema = item.get("theme") or item.get("tema") or item.get("title") or ""
                source = item.get("source_file") or minute_source or ""
                year = item.get("year") or minute_year or ""

                all_rows.append({
                    "texto": text,
                    "sumario": summary,
                    "segment_id": segment_id,
                    "tema": tema,
                    "source_file": source,
                    "municipality": muni_name,
                    "year": year,
                })

    return pd.DataFrame(all_rows)


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
        text_chunks: List[str] = chunk_text_dynamic(str(row.get("texto", row.get("text", ""))), tokenizer, max_length=model_max_len)
        
        processed_rows.append({
            "chunks": text_chunks,
            "sumario": str(row.get("sumario", row.get("resumo", ""))),
            "titulo": row.get("tema", ""),
            "segment_id": row.get("segment_id", ""),
            "text": row.get("texto", row.get("text", "")),
            "source_file": row.get("source_file", ""),
            "municipality": row.get("municipality", ""),
            "year": row.get("year", "")
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

    # convert relative paths to absolute to avoid hub parsing issues
    if os.path.exists(name):
        name = os.path.abspath(name)
        local_flag = True
    else:
        local_flag = False

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        name, use_fast=True, local_files_only=local_flag
    )
    # Load the model and ensure it's on the correct device
    model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(
        name, local_files_only=local_flag
    )
    model.to(device)
    model.eval()  # Set the model to evaluation mode

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
    parser = argparse.ArgumentParser(description="Generate summaries for validation segments using multiple models.")
    parser.add_argument("input", help="path to citilink JSON file")
    parser.add_argument("--val-split", type=float, default=VALIDATION_SPLIT,
                        help="fraction of examples reserved for validation (deprecated; use --split to control 60/20/20)")
    parser.add_argument("--split", type=float, default=0.2,
                        help="fraction for validation and test each (default 0.2 => 60/20/20)")
    parser.add_argument("--models-root", type=str, default="../train_models",
                        help="base directory where trained model folders are located (default ../train_models)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for splitting")
    args = parser.parse_args()

    # --- 1. Load All Data Segments ---
    print(f"Loading data from {args.input}...")
    df: pd.DataFrame = load_citilink(args.input)
    print(f"Total segments loaded: {len(df)}")

    # rename columns if necessary to match expected names
    df = df.rename(columns={"texto": "text", "sumario": "resumo"})

    # --- 2. Create Train/Validation/Test Split (60/20/20 by default) ---
    if len(df) == 0:
        print("Dataset is empty. Exiting.")
        return

    # args.split indicates fraction for validation and test each (e.g., 0.2 -> 60/20/20)
    val_fraction = float(args.split)
    if not (0.0 < val_fraction < 0.5):
        print("--split must be >0 and <0.5 (fraction for validation and test each). Exiting.")
        return

    # First carve out train (1 - 2*val_fraction) vs temp (2*val_fraction)
    train_frac = 1.0 - 2.0 * val_fraction
    train_df, temp_df = train_test_split(df, test_size=1.0 - train_frac, random_state=args.seed)
    # Split the temp into validation and test equally
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)

    print(f"Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)} ({train_frac*100:.1f}%/{val_fraction*100:.1f}%/{val_fraction*100:.1f}%).")

    precomputed_data: List[Dict[str, Any]] = []

    # --- 3. Iterate over Models and Generate Summaries ---
    for model_name, model_path in model_paths.items():
        print("\n" + "="*40)
        print(f"=== Starting Generation for: {model_name} ===")
        print("="*40)
        # Resolve model paths using the provided models root. This helps when
        # the path in `model_paths` is relative or the user placed models under
        # a different base directory.
        resolved_path = model_path
        if not os.path.exists(resolved_path):
            # try common alternatives under the provided models root
            base_name = os.path.basename(model_path)
            candidates = [
                os.path.join(args.models_root, base_name),
                os.path.join(args.models_root, base_name, "final"),
                os.path.join(args.models_root, base_name.replace('trained_', 'results_')),
                os.path.join(args.models_root, base_name.replace('trained_', 'results_'), "final"),
                os.path.join(args.models_root, base_name + "_segments"),
                os.path.join(args.models_root, base_name + "_segments", "final"),
            ]

            # also try to find any directory in models_root that contains a keyword from the model name
            keywords = []
            mn = model_name.lower()
            if "bart" in mn:
                keywords.append("bart")
            if "primera" in mn or "primera" in base_name.lower():
                keywords.append("primera")
            if "ptt5" in mn or "ptt5" in base_name.lower() or "t5" in mn:
                keywords.append("ptt5")
            if "led" in mn or "long" in mn:
                keywords.append("led")

            try:
                for entry in os.listdir(args.models_root):
                    entry_l = entry.lower()
                    if any(k in entry_l for k in keywords) and os.path.isdir(os.path.join(args.models_root, entry)):
                        candidates.append(os.path.join(args.models_root, entry))
                        candidates.append(os.path.join(args.models_root, entry, "final"))
            except Exception:
                pass

            # pick the first candidate that looks like a model repository
            found = False
            for c in candidates:
                if not os.path.exists(c):
                    continue
                # prefer a 'final' subfolder if present
                final_sub = os.path.join(c, "final")
                if os.path.exists(final_sub):
                    resolved_path = final_sub
                    found = True
                    break

                # accept directories that contain common model files
                common_files = ["pytorch_model.bin", "tf_model.h5", "flax_model.msgpack", "adapter_model.bin", "config.json"]
                if any(os.path.exists(os.path.join(c, cf)) for cf in common_files):
                    resolved_path = c
                    found = True
                    break

            if not found:
                print(f"⚠️  Model path for '{model_name}' not found under '{args.models_root}'. Checked candidates: {candidates}. Skipping.")
                continue

        # Load the model, tokenizer, and configuration
        try:
            bundle: Dict[str, Any] = load_model(resolved_path)
        except Exception as e:
            print(f"❌ Critical Error loading model {model_name} from {resolved_path}: {e}. Skipping.")
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
        # Use the validation split for generation
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