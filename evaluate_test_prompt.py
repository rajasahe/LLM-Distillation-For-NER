import pandas as pd
import json
import re
import difflib

def normalize_type(etype):
    etype = etype.strip().upper()
    mapping = {
        "ORG": "ORGANIZATION",
        "PER": "PERSON",
        "LOC": "LOCATION",
        "GPE": "LOCATION",  # often conflated
        "FAC": "FACILITY",
        "ART": "WORK_OF_ART",
        "CHAR": "CHARACTER",
        "NUM": "CARDINAL",  # sometimes mixed
        "TECH": "TECHNOLOGY"
    }
    return mapping.get(etype, etype)

def normalize_text(text):
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def parse_entities(text_output):
    if not isinstance(text_output, str):
        return []
    # try to find all entity and type using regex since JSON might be malformed
    matches = re.findall(r'\"entity\"\s*:\s*\"(.*?)\"\s*,\s*\"type\"\s*:\s*\"(.*?)\"', text_output, re.IGNORECASE)
    return [{"entity": e, "type": t} for e, t in matches]

def is_match(pred, gt, mode, text_only=False):
    pred_ent = normalize_text(pred["entity"])
    gt_ent = normalize_text(gt["entity"])
    pred_type = normalize_type(pred["type"])
    gt_type = normalize_type(gt["type"])
    
    type_match = (pred_type == gt_type)
    
    if mode == "zero_shot":
        if not type_match:
            ZERO_SHOT_TYPE_EQUIVALENCY = [
                {"ORGANIZATION", "EDUCATIONAL_INSTITUTION", "COMPANY", "UNIVERSITY", "BUSINESS", "INSTITUTION", "ORG"},
                {"LOCATION", "GPE", "CITY", "COUNTRY", "STATE", "FACILITY"},
                {"PERSON", "CHARACTER", "PER", "PEOPLE"},
                {"DATE", "TIME", "DURATION"},
                {"MONEY", "PRICE", "CURRENCY", "USD", "RUPEE"},
                {"CARDINAL", "ORDINAL", "QUANTITY", "NUM", "NUMBER"},
                {"PRODUCT", "SOFTWARE", "HARDWARE"},
                {"LAW", "DOCUMENT"},
                {"WORK_OF_ART", "ART", "SKILL"}
            ]
            for bucket in ZERO_SHOT_TYPE_EQUIVALENCY:
                if pred_type in bucket and gt_type in bucket:
                    type_match = True
                    break
                    
        # Relaxed matching for text
        abbr_match = False
        words_gt = gt_ent.split()
        words_pred = pred_ent.split()
        if len(words_gt) > 1 and "".join([w[0] for w in words_gt]) == pred_ent:
            abbr_match = True
        elif len(words_pred) > 1 and "".join([w[0] for w in words_pred]) == gt_ent:
            abbr_match = True
            
        # Spelling mistakes
        sim = difflib.SequenceMatcher(None, pred_ent, gt_ent).ratio()
        
        text_match = (pred_ent == gt_ent) or abbr_match or (sim >= 0.8) or (pred_ent in gt_ent) or (gt_ent in pred_ent)
    else:
        text_match = (pred_ent == gt_ent)
        
    if text_only:
        return text_match
    return text_match and type_match

def calculate_metrics(y_true_list, y_pred_list, mode, text_only=False):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # We will match predictions with ground truths
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    # Greedy matching
    for i, pred in enumerate(y_pred_list):
        for j, gt in enumerate(y_true_list):
            if j not in matched_gt_indices and is_match(pred, gt, mode, text_only):
                true_positives += 1
                matched_gt_indices.add(j)
                matched_pred_indices.add(i)
                break
                
    false_positives = len(y_pred_list) - len(matched_pred_indices)
    false_negatives = len(y_true_list) - len(matched_gt_indices)
    
    return true_positives, false_positives, false_negatives

def evaluate():
    df = pd.read_csv(r'C:\Users\Raja\Coriolis\Final_Distillation\Imp\TEST_PROMPT_100_FINAL_WITH_OUTPUTS.csv')
    
    metrics_by_mode = {}
    
    for idx, row in df.iterrows():
        gt_raw = row['Ground Truth (Claude and ChatGPT)']
        teacher_raw = row['Teacher Model Output (gemma3 12B)']
        new_raw = row['Distilled Model Output (gemma3 270M)']
        mode = row['mode']
        
        if mode not in metrics_by_mode:
            metrics_by_mode[mode] = {
                "Teacher": {"Strict_TP": 0, "Strict_FP": 0, "Strict_FN": 0, "Text_TP": 0, "Text_FP": 0, "Text_FN": 0},
                "NewModel": {"Strict_TP": 0, "Strict_FP": 0, "Strict_FN": 0, "Text_TP": 0, "Text_FP": 0, "Text_FN": 0}
            }
        
        gt_entities = parse_entities(gt_raw)
        teacher_entities = parse_entities(teacher_raw)
        new_entities = parse_entities(new_raw)
        
        # Teacher Strict
        tp, fp, fn = calculate_metrics(gt_entities, teacher_entities, mode, text_only=False)
        metrics_by_mode[mode]["Teacher"]["Strict_TP"] += tp
        metrics_by_mode[mode]["Teacher"]["Strict_FP"] += fp
        metrics_by_mode[mode]["Teacher"]["Strict_FN"] += fn
        
        # Teacher Text-Only
        tp, fp, fn = calculate_metrics(gt_entities, teacher_entities, mode, text_only=True)
        metrics_by_mode[mode]["Teacher"]["Text_TP"] += tp
        metrics_by_mode[mode]["Teacher"]["Text_FP"] += fp
        metrics_by_mode[mode]["Teacher"]["Text_FN"] += fn
        
        # New Model Strict
        tp, fp, fn = calculate_metrics(gt_entities, new_entities, mode, text_only=False)
        metrics_by_mode[mode]["NewModel"]["Strict_TP"] += tp
        metrics_by_mode[mode]["NewModel"]["Strict_FP"] += fp
        metrics_by_mode[mode]["NewModel"]["Strict_FN"] += fn
        
        # New Model Text-Only
        tp, fp, fn = calculate_metrics(gt_entities, new_entities, mode, text_only=True)
        metrics_by_mode[mode]["NewModel"]["Text_TP"] += tp
        metrics_by_mode[mode]["NewModel"]["Text_FP"] += fp
        metrics_by_mode[mode]["NewModel"]["Text_FN"] += fn

    results = []
    
    for mode, mode_metrics in metrics_by_mode.items():
        for model_name, m in mode_metrics.items():
            # Strict F1
            strict_prec = m["Strict_TP"] / (m["Strict_TP"] + m["Strict_FP"]) if (m["Strict_TP"] + m["Strict_FP"]) > 0 else 0
            strict_rec = m["Strict_TP"] / (m["Strict_TP"] + m["Strict_FN"]) if (m["Strict_TP"] + m["Strict_FN"]) > 0 else 0
            strict_f1 = 2 * strict_prec * strict_rec / (strict_prec + strict_rec) if (strict_prec + strict_rec) > 0 else 0
            
            # Text F1
            text_prec = m["Text_TP"] / (m["Text_TP"] + m["Text_FP"]) if (m["Text_TP"] + m["Text_FP"]) > 0 else 0
            text_rec = m["Text_TP"] / (m["Text_TP"] + m["Text_FN"]) if (m["Text_TP"] + m["Text_FN"]) > 0 else 0
            text_f1 = 2 * text_prec * text_rec / (text_prec + text_rec) if (text_prec + text_rec) > 0 else 0
            
            results.append({
                "Mode": mode,
                "Model": model_name,
                "Text F1": round(text_f1, 4),
                "Strict F1": round(strict_f1, 4),
                "Text Prec": round(text_prec, 4),
                "Strict Prec": round(strict_prec, 4),
                "Text Rec": round(text_rec, 4),
                "Strict Rec": round(strict_rec, 4)
            })
            
    res_df = pd.DataFrame(results)
    
    # Optional sorting logic to keep outputs intuitive
    mode_order = {"two_shot": 0, "one_shot": 1, "zero_shot": 2}
    model_order = {"Teacher": 0, "NewModel": 1}
    
    res_df['mode_rank'] = res_df['Mode'].map(mode_order)
    res_df['model_rank'] = res_df['Model'].map(model_order)
    res_df = res_df.sort_values(['mode_rank', 'model_rank']).drop(columns=['mode_rank', 'model_rank'])
    
    # Rename NewModel to Distilled for clarity
    res_df['Model'] = res_df['Model'].replace({"NewModel": "Distilled"})
    
    # ── Overall aggregate ────────────────────────────────────────
    overall_rows = []
    for model_name in ["Teacher", "Distilled"]:
        sub = res_df[res_df["Model"] == model_name]
        # Weighted by implicit count — simple macro average across modes
        overall_rows.append({
            "Mode": "overall",
            "Model": model_name,
            "Text F1":    round(sub["Text F1"].mean(), 4),
            "Strict F1":  round(sub["Strict F1"].mean(), 4),
            "Text Prec":  round(sub["Text Prec"].mean(), 4),
            "Strict Prec":round(sub["Strict Prec"].mean(), 4),
            "Text Rec":   round(sub["Text Rec"].mean(), 4),
            "Strict Rec": round(sub["Strict Rec"].mean(), 4),
        })
    res_df = pd.concat([res_df, pd.DataFrame(overall_rows)], ignore_index=True)

    # Save
    out_path = r'C:\Users\Raja\Coriolis\Final_Distillation\Imp\NER_Eval_Script_Results.csv'
    res_df.to_csv(out_path, index=False)
    print("=== Combined Comparison Metrics (F1, Precision, Recall) ===")
    print(res_df.to_string(index=False))
    print(f"\nSaved -> {out_path}")

if __name__ == '__main__':
    evaluate()
