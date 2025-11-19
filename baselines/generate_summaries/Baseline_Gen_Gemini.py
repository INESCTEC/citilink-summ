import os
import json
import google.generativeai as genai
from tqdm import tqdm
import pandas as pd
import argparse
import time
from typing import List, Dict, Any

# ===============================================================
# CONFIG DEFAULTS
# ===============================================================
# Default configuration, overridden by command-line arguments
DEFAULT_TEST_FOLDER: str = "/mnt/c/Users/migue/Desktop/citilink_summarization/dataset_split/test"
DEFAULT_MODEL_NAME: str = "gemini-2.5-flash"
DEFAULT_OUTPUT_FILE: str = "gemini_generated_output.jsonl"
API_KEY_ENV: str = "GEMINI_API_KEY"

# ===============================================================
# 1. ARGUMENT PARSING
# ===============================================================
def parse_args():
    """Parses command-line arguments for the script configuration."""
    parser = argparse.ArgumentParser(description="Generate summaries using the Gemini API for a test dataset.")
    parser.add_argument(
        "--test_folder",
        type=str,
        default=DEFAULT_TEST_FOLDER,
        help="Path to the directory containing the test JSON dataset files."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="The name of the Gemini model to use for generation (e.g., gemini-2.5-flash)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="The file path to save the generated JSONL output."
    )
    parser.add_argument(
        "--few_shot_count",
        type=int,
        default=5,
        help="Number of initial segments to use as few-shot examples."
    )
    return parser.parse_args()


# ===============================================================
# 2. DATA LOADING
# ===============================================================
def load_test_segments(folder: str) -> List[Dict[str, Any]]:
    """
    Loads text segments and reference summaries from all JSON files in the test folder.
    """
    rows: List[Dict[str, Any]] = []
    
    if not os.path.isdir(folder):
        print(f"Error: Test folder not found at {folder}")
        # Return an empty list to avoid crashing the script
        return rows
        
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            path = os.path.join(folder, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    
                for seg in doc.get("segments", []):
                    if seg.get("text") and seg.get("resumo"):
                        rows.append({
                            "document_id": doc.get("document_id", "N/A"),
                            "segment_id": seg.get("segment_id", None),
                            "text": seg["text"],
                            "reference": seg["resumo"],
                            "tema": seg.get("tema", "")
                        })
            except Exception as e:
                print(f"Error processing file '{filename}': {e}")
                
    return rows


# ===============================================================
# 3. GENERATION AND SAVING LOGIC
# ===============================================================
def generate_summaries(args: argparse.Namespace):
    """
    Main logic to load data, connect to Gemini, generate summaries, and save results.
    """
    # --- Data Loading ---
    test_segments: List[Dict[str, Any]] = load_test_segments(args.test_folder)
    if not test_segments:
        print("No test segments loaded. Exiting.")
        return

    print(f"Loaded {len(test_segments)} test segments from {args.test_folder}.")

    # --- API Key and Model Setup ---
    if API_KEY_ENV not in os.environ:
        raise ValueError(
            f"❌ API key not found. Please set the environment variable: {API_KEY_ENV}"
        )

    genai.configure(api_key=os.environ[API_KEY_ENV])
    model = genai.GenerativeModel(args.model_name)
    print(f"Initialized Gemini model: {args.model_name}")

    # --- Few-Shot Prompt Block ---
    few_shot: List[Dict[str, Any]] = test_segments[:args.few_shot_count]
    FEW_SHOT_PROMPT: str = ""
    for ex in few_shot:
        FEW_SHOT_PROMPT += (
            "### EXAMPLE\n"
            f"Texto:\n{ex['text']}\n"
            f"Resumo:\n{ex['reference']}\n\n"
        )
    print(f"Prepared few-shot block with {len(few_shot)} examples.")

    # --- Generation and Saving ---
    print(f"Starting generation and saving outputs to: {args.output_file}")
    
    # Open the output file
    try:
        with open(args.output_file, "w", encoding="utf-8") as fw:
            for item in tqdm(test_segments, desc="Generating Summaries"):
                
                # Construct the final prompt for the current segment
                prompt: str = (
                    FEW_SHOT_PROMPT +
                    "### NEW SEGMENT\n"
                    f"Texto:\n{item['text']}\n\n"
                    "Gere um resumo claro, objetivo e fiel ao texto acima:\n"
                )

                summary: str = ""
                MAX_RETRIES: int = 5
                
                # Implementation of Exponential Backoff
                for attempt in range(MAX_RETRIES):
                    try:
                        response = model.generate_content(prompt)
                        summary = response.text.strip()
                        break # Success, exit retry loop
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            wait_time = 2 ** attempt
                            print(f"\n[Warning] Attempt {attempt+1}/{MAX_RETRIES} failed for doc {item['document_id']}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"\n[Error] Final attempt failed for doc {item['document_id']}: {e}. Skipping segment.")
                            summary = "" # Ensure summary is empty if all retries fail
                            
                # Prepare and write the record
                record: Dict[str, Any] = {
                    "document_id": item["document_id"],
                    "segment_id": item["segment_id"],
                    "text": item["text"],
                    "reference": item["reference"],
                    "generated": summary
                }

                # Write the record as a single JSON line
                fw.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("\n✅ Generation Complete.")
        print(f"Generated outputs successfully written to: {args.output_file}")

    except Exception as file_error:
        print(f"\n[FATAL ERROR] Could not open or write to output file: {file_error}")


# ===============================================================
# 4. MAIN EXECUTION
# ===============================================================
if __name__ == "__main__":
    args = parse_args()
    generate_summaries(args)