import json
import os
import argparse
import re
import numpy as np
from collections import defaultdict
from pathlib import Path

# Load MITRE knowledge base
def load_mitre_kb(kb_path="./assets/mitre_kb.json"):
    try:
        with open(kb_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: MITRE knowledge base not found at {kb_path}. Validation will be skipped.")
        return {}

mitre_kb = load_mitre_kb()

def extract_mitre_techniques(text):
    """Extract MITRE technique IDs from text."""
    mitre_pattern = r'T\d{4}(?:\.\d{3})?'
    matches = re.findall(mitre_pattern, text)
    unique_matches = list(set(matches))
    return unique_matches

def is_valid_technique(technique: str):
    """Check if a technique ID is valid according to the MITRE knowledge base."""
    if not mitre_kb:
        return True  # Skip validation if KB is not available
    return technique in mitre_kb

def load_results(results_dir="./results"):
    """Load all result files from the results directory."""
    results_files = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith("_results.json"):
            model_name = filename.split("_results.json")[0]
            file_path = os.path.join(results_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    results_files[model_name] = json.load(f)
                print(f"Loaded {len(results_files[model_name])} examples from {filename}")
            except json.JSONDecodeError:
                print(f"Error: Could not parse {filename} as JSON")
                
    return results_files

def precision_at_k(preds, trues):
    """Calculate precision at k metric."""
    correct = len(set(preds).intersection(set(trues)))
    return (correct / len(preds)) if preds else 0

def recall_at_k(preds, trues):
    """Calculate recall at k metric."""
    correct = len(set(preds).intersection(set(trues)))
    return correct / len(trues) if trues else 0

def f1_score(precision, recall):
    """Calculate F1 score from precision and recall."""
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def mrr_at_k(preds, trues):
    """Calculate Mean Reciprocal Rank at k."""
    for i, cls in enumerate(preds):
        if cls in trues:
            return 1 / (i + 1)
    return 0

def evaluate_model(data, mode="technique"):
    """Evaluate model performance on the given data."""
    precision = []
    recall = []
    mrr = []
    analysis_data = []
    
    for item in data:
        if item.get('predicted') is None:
            continue
        
        # Handle case where gold is a string instead of a list
        if isinstance(item.get('gold'), str):
            item['gold'] = extract_mitre_techniques(text=item['gold'])
        
        if mode == "technique":
            # For technique mode, remove subtechnique information
            trues = list(set([e.split(".")[0] for e in item.get('gold', [])]))
            preds = []
            for e in item.get('predicted', []):
                e = e.split(".")[0]
                if is_valid_technique(e):
                    preds.append(e)
            preds = list(set(preds))
        else:
            # For subtechnique mode, keep the full technique IDs
            trues = item.get('gold', [])
            preds = []
            for e in item.get('predicted', []):
                if is_valid_technique(e):
                    preds.append(e)
            preds = list(set(preds))

        # Save data for analysis
        if item.get('input') is not None and item.get('instruction') is not None:
            analysis_item = {
                "text": item['input'],
                "gold": trues,
                "rag_output": extract_mitre_techniques(item['instruction']),
                "llm_output": preds
            }
            analysis_data.append(analysis_item)
        
        # Calculate metrics
        precision.append(precision_at_k(preds, trues))
        recall.append(recall_at_k(preds, trues))
        mrr.append(mrr_at_k(preds, trues))
    
    # Calculate averages
    avg_precision = np.mean(precision) if precision else 0
    avg_recall = np.mean(recall) if recall else 0
    avg_f1 = f1_score(avg_precision, avg_recall)
    avg_mrr = np.mean(mrr) if mrr else 0
    
    return avg_precision, avg_recall, avg_f1, avg_mrr, analysis_data

def compare_models(results_data, mode="technique"):
    """Compare performance across all models."""
    model_results = {}
    analysis_data = defaultdict(list)
    
    for model, data in results_data.items():
        precision, recall, f1, mrr, model_analysis = evaluate_model(data, mode)
        model_results[model] = (precision, recall, f1, mrr)
        analysis_data[model].extend(model_analysis)
    
    # Save analysis data
    os.makedirs("./analysis", exist_ok=True)
    for model, data in analysis_data.items():
        with open(f"./analysis/{model}_analysis.jsonl", "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    
    return model_results

def generate_markdown_table(results):
    """Generate a markdown table of the results."""
    table = "### Model Comparison Results\n\n"
    table += "| Model | Precision | Recall | F1-score | MRR |\n"
    table += "|-------|-----------|--------|----------|-----|\n"
    
    for model, metrics in results.items():
        precision, recall, f1, mrr = metrics
        table += f"| {model} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {mrr:.4f} |\n"
    
    return table

def main():
    parser = argparse.ArgumentParser(description="Evaluate MITRE ATT&CK technique identification across models")
    parser.add_argument("--mode", type=str, default="technique", choices=["technique", "subtechnique"], 
                        help="Whether to evaluate at technique or subtechnique level")
    parser.add_argument("--results_dir", type=str, default="./results", 
                        help="Directory containing result files")
    parser.add_argument("--mitre_kb", type=str, default="./assets/mitre_kb.json", 
                        help="Path to MITRE knowledge base JSON file")
    args = parser.parse_args()

    # Load the MITRE knowledge base
    global mitre_kb
    mitre_kb = load_mitre_kb(args.mitre_kb)
    
    # Load results
    results_data = load_results(args.results_dir)
    
    if not results_data:
        print("No result files found. Please make sure your results files end with '_results.json'")
        return
    
    # Compare models
    model_results = compare_models(results_data, args.mode)
    
    # Generate and print markdown table
    markdown_table = generate_markdown_table(model_results)
    print(markdown_table)
    
    # Save results to file
    with open("model_comparison_results.md", "w") as f:
        f.write(markdown_table)
    print("Results saved to model_comparison_results.md")

if __name__ == "__main__":
    main()
