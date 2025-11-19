import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Desativa o paralelismo do tokenizer
import json
import evaluate
import bert_score
from tqdm import tqdm
import argparse
from typing import List, Dict, Any

# ===============================================================
# CONFIG
# ===============================================================
# Tokenizador e Modelo usados para o BERTScore
BERT_SCORE_MODEL_TYPE: str = "microsoft/deberta-xlarge-mnli" 
BERT_SCORE_LANG: str = "pt" # Idioma Português

# ===============================================================
# 1. ARGUMENT PARSING
# ===============================================================
def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments, expecting the path to the generated JSONL file.
    """
    parser = argparse.ArgumentParser(description="Evaluate generated summaries from a JSONL file.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the JSONL file containing the generated summaries and references (output of the generation script)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="gemini_evaluation_results.json",
        help="File path to save the final evaluation results (JSON format)."
    )
    return parser.parse_args()


# ===============================================================
# 2. DATA LOADING
# ===============================================================
def load_generated_data(file_path: str) -> tuple[List[str], List[str]]:
    """
    Loads generated summaries and reference summaries from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file.
        
    Returns:
        A tuple (list of generated summaries, list of references).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Input file not found at: {file_path}")
        
    generated: List[str] = []
    references: List[str] = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading generated data"):
            try:
                record: Dict[str, Any] = json.loads(line)
                # Assumes the generation script saved 'generated' and 'reference' keys
                generated.append(record.get("generated", "").strip())
                references.append(record.get("reference", "").strip())
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Skipping malformed JSON line: {e}")
                
    # Filter out empty strings which can cause errors in some metrics
    filtered_data = [(g, r) for g, r in zip(generated, references) if g and r]
    
    if not filtered_data:
        print("Error: No valid summary-reference pairs loaded.")
        return [], []
        
    print(f"Loaded {len(filtered_data)} valid pairs for evaluation.")
    return [g for g, r in filtered_data], [r for g, r in filtered_data]


# ===============================================================
# 3. EVALUATION LOGIC
# ===============================================================
def compute_metrics(generated: List[str], references: List[str]) -> Dict[str, Any]:
    """
    Computes ROUGE, BLEU, METEOR, and BERTScore.
    """
    print("\nStarting evaluation...")
    
    # Initialize metric results dictionary
    results: Dict[str, Any] = {}
    
    # --- ROUGE ---
    print("-> Computing ROUGE...")
    rouge = evaluate.load("rouge")
    results["rouge"] = rouge.compute(
        predictions=generated,
        references=references
    )

    # --- BLEU ---
    print("-> Computing BLEU...")
    bleu = evaluate.load("bleu")
    results["bleu"] = bleu.compute(
        predictions=generated,
        references=references
    )

    # --- METEOR ---
    print("-> Computing METEOR...")
    meteor = evaluate.load("meteor")
    results["meteor"] = meteor.compute(
        predictions=generated,
        references=references
    )
    
    # --- BERTScore ---
    print(f"-> Computing BERTScore (Model: {BERT_SCORE_MODEL_TYPE}, Lang: {BERT_SCORE_LANG})...")
    
    # BERTScore requires bert_score.score to be imported
    try:
        bertscore_results = bert_score.score(
            cands=generated,
            refs=references,
            lang=BERT_SCORE_LANG,
            model_type=BERT_SCORE_MODEL_TYPE,
            verbose=True
        )

        results["bertscore"] = {
            "precision": float(bertscore_results[0].mean()),
            "recall": float(bertscore_results[1].mean()),
            "f1": float(bertscore_results[2].mean())
        }
    except Exception as e:
        print(f"❌ BERTScore Error: Could not compute BERTScore. Ensure required model/dependencies are installed. Error: {e}")
        results["bertscore"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return results

# ===============================================================
# 4. MAIN EXECUTION
# ===============================================================
def main():
    args = parse_args()

    # 1. Load Data
    generated, references = load_generated_data(args.input_file)
    
    if not generated:
        return
    
    # 2. Compute Metrics
    evaluation_results = compute_metrics(generated, references)

    # 3. Save Results
    output_path = args.output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*40)
    print(f"✅ EVALUATION COMPLETE. Results saved to {output_path}")
    print("="*40)
    print(json.dumps(evaluation_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()