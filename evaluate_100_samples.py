import pandas as pd
import json
import re

CSV_PATH = r"C:\Users\Raja\Coriolis\Final_Distillation\Test Data\Test_Data_100.csv"

def normalize_text(text):
    text = text.strip().lower()
    # Remove punctuation for fair matching
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def normalize_type(etype):
    etype = etype.strip().upper()
    mapping = {
        "ORG": "ORGANIZATION",
        "PER": "PERSON",
        "LOC": "LOCATION",
        "GPE": "LOCATION",
        "FAC": "FACILITY",
        "ART": "WORK_OF_ART",
        "CHAR": "CHARACTER",
        "NUM": "CARDINAL",
        "TECH": "TECHNOLOGY"
    }
    return mapping.get(etype, etype)

def parse_entities(text):
    """Safely extract entities from JSON-like string with normalization."""
    if not isinstance(text, str) or not text.strip():
        return set()
    
    matches = re.findall(r'"entity"\s*:\s*"(.*?)"\s*,\s*"type"\s*:\s*"(.*?)"', text, re.IGNORECASE)
    
    entities = set()
    for e, t in matches:
        # Apply normalization to both text and type
        entities.add((normalize_text(e), normalize_type(t)))
    
    return entities

def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def evaluate():
    df = pd.read_csv(CSV_PATH)
    
    # Structure to hold metrics: mode -> model -> {"TP": 0, "FP": 0, "FN": 0}
    results = {
        "zero_shot": {"Teacher": {"TP": 0, "FP": 0, "FN": 0}, "Base_Student": {"TP": 0, "FP": 0, "FN": 0}, "Distilled": {"TP": 0, "FP": 0, "FN": 0}},
        "one_shot":  {"Teacher": {"TP": 0, "FP": 0, "FN": 0}, "Base_Student": {"TP": 0, "FP": 0, "FN": 0}, "Distilled": {"TP": 0, "FP": 0, "FN": 0}},
        "two_shot":  {"Teacher": {"TP": 0, "FP": 0, "FN": 0}, "Base_Student": {"TP": 0, "FP": 0, "FN": 0}, "Distilled": {"TP": 0, "FP": 0, "FN": 0}},
    }
    
    for _, row in df.iterrows():
        mode = row.get("mode", "unknown")
        if mode not in results:
            continue
            
        gt_set = parse_entities(row.get("Claude", ""))
        teacher_set = parse_entities(row.get("Teacher_Output", ""))
        base_set = parse_entities(row.get("Base_Student_Output", ""))
        distilled_set = parse_entities(row.get("Distilled_Student_Output", ""))
        
        # Evaluate Teacher
        t_tp = len(gt_set & teacher_set)
        t_fp = len(teacher_set - gt_set)
        t_fn = len(gt_set - teacher_set)
        
        results[mode]["Teacher"]["TP"] += t_tp
        results[mode]["Teacher"]["FP"] += t_fp
        results[mode]["Teacher"]["FN"] += t_fn
        
        # Evaluate Base Student
        b_tp = len(gt_set & base_set)
        b_fp = len(base_set - gt_set)
        b_fn = len(gt_set - base_set)
        
        results[mode]["Base_Student"]["TP"] += b_tp
        results[mode]["Base_Student"]["FP"] += b_fp
        results[mode]["Base_Student"]["FN"] += b_fn
        
        # Evaluate Distilled
        d_tp = len(gt_set & distilled_set)
        d_fp = len(distilled_set - gt_set)
        d_fn = len(gt_set - distilled_set)
        
        results[mode]["Distilled"]["TP"] += d_tp
        results[mode]["Distilled"]["FP"] += d_fp
        results[mode]["Distilled"]["FN"] += d_fn

    # Format the output table
    print("="*80)
    print(f"{'Mode':<15} | {'Model':<15} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 80)
    
    for mode in ["zero_shot", "one_shot", "two_shot"]:
        for model in ["Teacher", "Base_Student", "Distilled"]:
            metrics = results[mode][model]
            p, r, f1 = compute_metrics(metrics["TP"], metrics["FP"], metrics["FN"])
            print(f"{mode:<15} | {model:<15} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")
        print("-" * 80)

if __name__ == "__main__":
    evaluate()
