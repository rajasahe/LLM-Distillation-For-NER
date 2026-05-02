import pandas as pd
import re

INPUT_CSV = r"C:\Users\Raja\Coriolis\Final_Distillation\Test Data\test_ner_prompts_sampled_100 - Copy.csv"
OUTPUT_CSV = r"C:\Users\Raja\Coriolis\Final_Distillation\Test Data\test_ner_prompts_sampled_100_cleaned.csv"

def extract_first_json_array(text):
    if not isinstance(text, str):
        return text
    
    # Use non-greedy regex to find the very first array [...]
    # re.DOTALL makes '.' match newlines as well
    match = re.search(r'\[.*?\]', text, flags=re.DOTALL)
    if match:
        # Return the matched JSON array string
        return match.group(0).strip()
    
    # If no JSON array is found, return the original text
    return text.strip()

def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, encoding='utf-8', encoding_errors='replace')
    
    columns_to_clean = ["Distilled_Student_Output"]
    
    for col in columns_to_clean:
        if col in df.columns:
            print(f"Cleaning column: {col}...")
            df[col] = df[col].apply(extract_first_json_array)
            
    print(f"Saving cleaned outputs to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
