import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from statistics import mean
from sacrebleu import corpus_bleu
import evaluate
import bert_score
from transformers import AutoTokenizer
from moverscore_v2 import get_idf_dict, word_mover_score
from tqdm import tqdm
from typing import List, Dict, Any, Union, Tuple, Optional

# --------------------- Configuration ---------------------
# Maximum token length used for chunking large texts 
MAX_TOKENS = 512
# Base directory for input/output files
BASE_PATH = "."
# Path to the input JSON file containing pre-generated summaries from multiple models
# Default to the generator's validation output filename.
INPUT_FILE = "../generate_summaries/val_precomputed_all_models_dynamic.json"
# Tokenizer checkpoint used for BERTScore and MoverScore chunking
TOKENIZER_NAME = "bert-base-multilingual-cased"

# Load metric modules from the 'evaluate' library
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

# --------------------- Funções ---------------------
def chunk_text(text: str, tokenizer: AutoTokenizer, max_len: int = MAX_TOKENS) -> List[str]:
    """
    Splits a long text into chunks of max_len tokens.
    """
    tokens: List[int] = tokenizer.encode(text, add_special_tokens=False)
    chunks: List[str] = []
    
    for i in range(0, len(tokens), max_len):
        chunk: List[int] = tokens[i:i+max_len]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
        
    return chunks if chunks else [text]

def compute_all_metrics(preds: List[str], refs: List[str], tokenizer: Optional[AutoTokenizer] = None) -> Dict[str, float]:
    """
    Computes a comprehensive set of evaluation metrics between predictions and references.
    """
    scores: Dict[str, float] = {}
    
    # ROUGE
    scores.update(rouge.compute(predictions=preds, references=refs))
    
    # BLEU
    scores["bleu"] = corpus_bleu(preds, [[r] for r in refs]).score
    
    # METEOR
    try:
        scores["meteor"] = meteor.compute(predictions=preds, references=[[r] for r in refs])["meteor"]
    except Exception as e:
        # print(f"⚠️ METEOR Error: {e}")
        scores["meteor"] = 0.0
        
    # BERTScore
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        
    preds_chunks: List[List[str]] = [[c for c in chunk_text(p, tokenizer) if c.strip()] for p in preds]
    refs_chunks: List[List[str]] = [[c for c in chunk_text(r, tokenizer) if c.strip()] for r in refs]

    P_all, R_all, F1_all = [], [], []
    for p_chunks, r_chunks in zip(preds_chunks, refs_chunks):
        min_len: int = min(len(p_chunks), len(r_chunks))
        for p, r in zip(p_chunks[:min_len], r_chunks[:min_len]):
            P, R, F1 = bert_score.score([p], [r], lang="pt", verbose=False)
            P_all.append(P.item())
            R_all.append(R.item())
            F1_all.append(F1.item())
            
    scores["bertscore_precision"] = mean(P_all) if P_all else 0.0
    scores["bertscore_recall"] = mean(R_all) if R_all else 0.0
    scores["bertscore_f1"] = mean(F1_all) if F1_all else 0.0

    # MoverScore
    try:
        idf_ref: Dict[str, float] = get_idf_dict([r for sub in refs_chunks for r in sub])
        idf_hyp: Dict[str, float] = get_idf_dict([p for sub in preds_chunks for p in sub])
        
        mover: List[float] = word_mover_score(
            [r for sub in refs_chunks for r in sub],
            [p for sub in preds_chunks for p in sub],
            idf_ref, idf_hyp,
            stop_words=[], n_gram=1,
            remove_subwords=True, batch_size=8
        )
        scores["moverscore"] = mean(mover) if mover else 0.0
    except Exception as e:
        # print(f"⚠️ MoverScore Error: {e}")
        scores["moverscore"] = 0.0

    return scores

# --------------------- Main Execution ---------------------
def main():
    # If the configured input file is missing, try to auto-discover a precomputed file
    if not os.path.exists(INPUT_FILE):
        gen_dir = os.path.dirname(INPUT_FILE) or "../generate_summaries"
        candidates = []
        try:
            for fn in os.listdir(gen_dir):
                if fn.lower().endswith('.json') and 'precomputed' in fn.lower():
                    candidates.append(os.path.join(gen_dir, fn))
        except Exception:
            candidates = []

        if candidates:
            input_path = candidates[0]
            print(f"⚠️  Input file {INPUT_FILE} not found; using discovered file: {input_path}")
        else:
            print(f"❌ ERRO CRÍTICO: O ficheiro de entrada não foi encontrado no caminho especificado: {INPUT_FILE}")
            print("Certifique-se de que está a executar o script a partir do diretório correto.")
            return
    else:
        input_path = INPUT_FILE

    # Load the precomputed summaries file
    with open(input_path, "r", encoding="utf-8") as f:
        results: List[Dict[str, Any]] = json.load(f)

    print(f"Loaded {len(results)} test segments from {input_path}")
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    segment_metrics: List[Dict[str, Any]] = []
    global_accum: Dict[str, Dict[str, List[float]]] = {}
    global_by_camara: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    # Iterate through each segment in the test set
    for i, seg in enumerate(tqdm(results, desc="Processing test segments")):
        ref: str = seg.get("resumo", seg.get("resumo_gold", "")) 
        camara: str = seg.get("municipality", "Unknown")

        for model_name, summary in seg.get("generated_summaries", {}).items():
            
            if not ref.strip() or not summary.strip():
                continue

            try:
                metrics: Dict[str, float] = compute_all_metrics([summary], [ref], tokenizer=tokenizer)
            except Exception as e:
                print(f"⚠️ Error computing metrics for segment {i}, model {model_name}: {e}")
                continue

            # 1. Save segment-level metrics
            record: Dict[str, Any] = {"id": i, "camara": camara, "model": model_name, **metrics}
            segment_metrics.append(record)

            # 2. Accumulate metrics globally 
            if model_name not in global_accum:
                global_accum[model_name] = {k: [] for k in metrics.keys()}
            for k, v in metrics.items():
                if v is not None:
                    global_accum[model_name][k].append(v)

            # 3. Accumulate metrics by municipality 
            if camara not in global_by_camara:
                global_by_camara[camara] = {}
            if model_name not in global_by_camara[camara]:
                global_by_camara[camara][model_name] = {k: [] for k in metrics.keys()}
            for k, v in metrics.items():
                if v is not None:
                    global_by_camara[camara][model_name][k].append(v)

    # --------------------- Save Results ---------------------
    
    # Save Segment metrics (JSONL format)
    output_path_segment: str = f"{BASE_PATH}/test_segment_metrics.jsonl"
    with open(output_path_segment, "w", encoding="utf-8") as f:
        for record in segment_metrics:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save Global metrics (average across all segments)
    global_metrics: List[Dict[str, Union[str, float]]] = []
    for model, scores_dict in global_accum.items():
        avg_scores: Dict[str, float] = {k: mean(v) for k, v in scores_dict.items() if v}
        avg_scores["model"] = model
        global_metrics.append(avg_scores)
        
    output_path_global: str = f"{BASE_PATH}/test_global_metrics.json"
    with open(output_path_global, "w", encoding="utf-8") as f:
        json.dump(global_metrics, f, indent=2, ensure_ascii=False)

    # Save Metrics by municipality (camara)
    global_camara_metrics: Dict[str, List[Dict[str, Union[str, float]]]] = {}
    for camara, models_dict in global_by_camara.items():
        global_camara_metrics[camara] = []
        for model, scores_dict in models_dict.items():
            avg_scores: Dict[str, float] = {k: mean(v) for k, v in scores_dict.items() if v}
            avg_scores["model"] = model
            global_camara_metrics[camara].append(avg_scores)
            
    output_path_camara: str = f"{BASE_PATH}/test_global_metrics_by_camara.json"
    with open(output_path_camara, "w", encoding="utf-8") as f:
        json.dump(global_camara_metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ Métricas do TEST SET pré-carregadas e guardadas com sucesso em três ficheiros de saída (Segment, Global, By_Camara).")


if __name__ == "__main__":
    main()