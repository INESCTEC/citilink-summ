import os
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from typing import List, Dict, Any

# ===============================================================
# CONFIG DEFAULTS
# ===============================================================
DEFAULT_TEST_FOLDER: str = "/mnt/c/Users/migue/Desktop/citilink_summarization/dataset_split/test"
DEFAULT_MODEL_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT_FILE: str = "qwen2_1.5b_generated.jsonl"
DEFAULT_FEW_SHOT_NUM: int = 5
DEFAULT_MAX_INPUT_LENGTH: int = 1024
DEFAULT_MAX_GEN_LENGTH: int = 128
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================================================
# 1. ARGUMENT PARSING
# ===============================================================
def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the script configuration."""
    parser = argparse.ArgumentParser(description="Generate summaries using a Hugging Face CausalLM (Qwen) for a test dataset.")
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
        help="The Hugging Face model name (e.g., Qwen/Qwen2.5-1.5B-Instruct)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="The file path to save the generated JSONL output."
    )
    parser.add_argument(
        "--few_shot_num",
        type=int,
        default=DEFAULT_FEW_SHOT_NUM,
        help="Number of initial segments to use as few-shot examples."
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=DEFAULT_MAX_INPUT_LENGTH,
        help="Maximum sequence length for the input context."
    )
    parser.add_argument(
        "--max_gen_length",
        type=int,
        default=DEFAULT_MAX_GEN_LENGTH,
        help="Maximum number of new tokens to generate for the summary."
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
# 3. GENERATION LOGIC
# ===============================================================
def generate_summaries(args: argparse.Namespace):
    """
    Main logic to load data, load the Hugging Face model, generate summaries, and save results.
    """
    # --- Data Loading ---
    test_segments: List[Dict[str, Any]] = load_test_segments(args.test_folder)
    if not test_segments:
        print("No test segments loaded. Exiting.")
        return

    print(f"Loaded {len(test_segments)} test segments from {args.test_folder}.")

    # --- Model and Tokenizer Initialization ---
    print(f"Loading model '{args.model_name}' to device: {DEVICE}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model or tokenizer: {e}")
        return

    # --- Few-Shot Prompt Block ---
    few_shot: List[Dict[str, Any]] = test_segments[:args.few_shot_num]
    few_shot_prompt: str = ""
    for ex in few_shot:
        few_shot_prompt += (
            f"Texto:\n{ex['text']}\nResumo:\n{ex['reference']}\n\n"
        )
    print(f"Prepared few-shot block with {len(few_shot)} examples.")


    # --- Generation and Saving ---
    print(f"Starting generation and saving outputs to: {args.output_file}")

    try:
        with open(args.output_file, "w", encoding="utf-8") as fw:
            for item in tqdm(test_segments, desc="Generating Summaries"):
                # The final prompt includes the few-shot context and the new segment
                prompt: str = few_shot_prompt + f"Texto:\n{item['text']}\nResumo:\n"

                summary: str = ""
                try:
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=args.max_input_length,
                        truncation=True
                    ).to(DEVICE)

                    # Determine max_new_tokens based on max input and total limits
                    max_new_tokens: int = args.max_gen_length
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False, # Use greedy decoding
                        pad_token_id=tokenizer.eos_token_id
                    )

                    # Decode the generated part only
                    summary = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    ).strip()

                except Exception as e:
                    print(f"\n[Error] Generation failed for document {item['document_id']}: {e}. Skipping segment.")
                    summary = ""

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