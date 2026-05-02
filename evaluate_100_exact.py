import pandas as pd
import re

CSV_PATH = r"C:\Users\Raja\Coriolis\Final_Distillation\Test Data\Test_Data_100.csv"

def parse_entities_exact(text):
    """Extract entities for exact text match and strict (text+type) match."""
    if not isinstance(text, str) or not text.strip():
        return set(), set()
    
    matches = re.findall(r'"entity"\s*:\s*"(.*?)"\s*,\s*"type"\s*:\s*"(.*?)"', text, re.IGNORECASE)
    
    text_only = set()
    strict = set()
    for e, t in matches:
        # Just strip boundary whitespace, keep case and punctuation exactly as is
        e_exact = e.strip()
        t_exact = t.strip()
        text_only.add(e_exact)
        strict.add((e_exact, t_exact))
    
    return text_only, strict

def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def evaluate():
    df = pd.read_csv(CSV_PATH)
    
    results = {
        "zero_shot": {"Teacher": {"text": {"TP": 0, "FP": 0, "FN": 0}, "strict": {"TP": 0, "FP": 0, "FN": 0}},
                      "Base_Student": {"text": {"TP": 0, "FP": 0, "FN": 0}, "strict": {"TP": 0, "FP": 0, "FN": 0}},
                      "Distilled": {"text": {"TP": 0, "FP": 0, "FN": 0}, "strict": {"TP": 0, "FP": 0, "FN": 0}}},
        "one_shot":  {"Teacher": {"text": {"TP": 0, "FP": 0, "FN": 0}, "strict": {"TP": 0, "FP": 0, "FN": 0}},
                      "Base_Student": {"text": {"TP": 0, "FP": 0, "FN": 0}, "strict": {"TP": 0, "FP": 0, "FN": 0}},
                      "Distilled": {"text": {"TP": 0, "FP": 0, "FN": 0}, "strict": {"TP": 0, "FP": 0, "FN": 0}}},
        "two_shot":  {"Teacher": {"text": {"TP": 0, "FP": 0, "FN": 0}, "strict": {"TP": 0, "FP": 0, "FN": 0}},
                      "Base_Student": {"text": {"TP": 0, "FP": 0, "FN": 0}, "strict": {"TP": 0, "FP": 0, "FN": 0}},
                      "Distilled": {"text": {"TP": 0, "FP": 0, "FN": 0}, "strict": {"TP": 0, "FP": 0, "FN": 0}}},
    }
    
    for _, row in df.iterrows():
        mode = row.get("mode", "unknown")
        if mode not in results:
            continue
            
        gt_text, gt_strict = parse_entities_exact(row.get("Claude", ""))
        teacher_text, teacher_strict = parse_entities_exact(row.get("Teacher_Output", ""))
        base_text, base_strict = parse_entities_exact(row.get("Base_Student_Output", ""))
        dist_text, dist_strict = parse_entities_exact(row.get("Distilled_Student_Output", ""))
        
        models_data = {
            "Teacher": (teacher_text, teacher_strict),
            "Base_Student": (base_text, base_strict),
            "Distilled": (dist_text, dist_strict)
        }
        
        for model, (m_text, m_strict) in models_data.items():
            # Text Only evaluation
            results[mode][model]["text"]["TP"] += len(gt_text & m_text)
            results[mode][model]["text"]["FP"] += len(m_text - gt_text)
            results[mode][model]["text"]["FN"] += len(gt_text - m_text)
            
            # Strict (Text + Type) evaluation
            results[mode][model]["strict"]["TP"] += len(gt_strict & m_strict)
            results[mode][model]["strict"]["FP"] += len(m_strict - gt_strict)
            results[mode][model]["strict"]["FN"] += len(gt_strict - m_strict)

    # Output
    print("="*110)
    print(f"{'Mode':<12} | {'Model':<15} | {'TEXT ONLY MATCH (Prec/Rec/F1)':<35} | {'STRICT MATCH (Text+Type) (Prec/Rec/F1)'}")
    print("-" * 110)
    
    for mode in ["zero_shot", "one_shot", "two_shot"]:
        for model in ["Teacher", "Base_Student", "Distilled"]:
            text_m = results[mode][model]["text"]
            t_p, t_r, t_f1 = compute_metrics(text_m["TP"], text_m["FP"], text_m["FN"])
            
            strict_m = results[mode][model]["strict"]
            s_p, s_r, s_f1 = compute_metrics(strict_m["TP"], strict_m["FP"], strict_m["FN"])
            
            text_str = f"{t_p:.4f} / {t_r:.4f} / {t_f1:.4f}"
            strict_str = f"{s_p:.4f} / {s_r:.4f} / {s_f1:.4f}"
            
            print(f"{mode:<12} | {model:<15} | {text_str:<35} | {strict_str}")
        print("-" * 110)

if __name__ == "__main__":
    evaluate()
