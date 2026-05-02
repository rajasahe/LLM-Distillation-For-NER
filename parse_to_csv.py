import json
import csv
from pathlib import Path

# File paths
input_file = r"C:\Users\Raja\Coriolis\Final_Distillation\Test Data\test_ner_dataset_final_200.jsonl"
output_file = r"C:\Users\Raja\Coriolis\Final_Distillation\Test Data\test_ner_prompts_200.csv"

def build_prompt(record):
    base_prompt = """Extract named entities from the sentence. Return a JSON array only.
Each item: {"entity": "exact text from sentence", "type": "ENTITY_TYPE"}
If no entities exist, return [].

Rules:
- Copy entity text exactly. Do not change it.
- Strip leading "a", "an", "the" from entity text only.
- Skip generic nouns, job titles alone, and abstract concepts."""

    examples = record.get("examples", [])
    target_sentence = record.get("target_sentence", "")
    
    # Start assembling the prompt
    prompt = base_prompt + "\n\n"
    
    # Add examples if there are any (for one-shot and two-shot)
    if examples:
        prompt += "Examples:\n"
        for ex in examples:
            ex_sentence = ex.get("sentence", "")
            # Dump the entities strictly as a JSON string
            ex_entities_json = json.dumps(ex.get("entities", []))
            prompt += f'Sentence: """{ex_sentence}"""\n\nOutput: {ex_entities_json}\n\n---\n\n'
    
    # Add the final target sentence
    prompt += f'Sentence: """{target_sentence}"""\n\nOutput:'
    
    return prompt

def main():
    if not Path(input_file).exists():
        print(f"Error: Could not find the input file at {input_file}")
        return

    records = []
    # 1. Parse the JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                
    # 2. Build prompts and 3. Save as CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # Writing header
        writer.writerow(["mode", "target_sentence", "prompt"])
        
        for record in records:
            mode = record.get("mode", "unknown")
            target_sentence = record.get("target_sentence", "")
            prompt = build_prompt(record)
            
            writer.writerow([mode, target_sentence, prompt])
            
    print(f"Successfully processed {len(records)} records.")
    print(f"Prompts saved to: {output_file}")

if __name__ == "__main__":
    main()
