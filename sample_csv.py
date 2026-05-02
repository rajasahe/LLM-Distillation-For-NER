import csv
import random

input_file = r"C:\Users\Raja\Coriolis\Final_Distillation\Test Data\test_ner_prompts_200.csv"
output_file = r"C:\Users\Raja\Coriolis\Final_Distillation\Test Data\test_ner_prompts_sampled_100.csv"

def main():
    records_by_mode = {
        "zero_shot": [],
        "one_shot": [],
        "two_shot": []
    }
    
    # Read the csv
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row.get("mode")
            if mode in records_by_mode:
                records_by_mode[mode].append(row)
    
    sampled_records = []
    
    # Sample to reach 100 total (34, 33, 33)
    target_distribution = {
        "zero_shot": 34,
        "one_shot": 33,
        "two_shot": 33
    }
    
    random.seed(42)  # For reproducibility
    for mode in records_by_mode:
        pool = records_by_mode[mode]
        k = min(target_distribution[mode], len(pool))
        sampled = random.sample(pool, k)
        sampled_records.extend(sampled)
        print(f"Sampled {k} from {mode} (out of {len(pool)} available)")
        
    # Shuffle the final 15 records so they are mixed
    random.shuffle(sampled_records)
        
    # Write to new CSV
    if sampled_records:
        fieldnames = ["mode", "target_sentence", "prompt"]
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sampled_records)
            
        print(f"Successfully sampled {len(sampled_records)} records and saved to:\n{output_file}")
    else:
        print("No records found.")

if __name__ == "__main__":
    main()
