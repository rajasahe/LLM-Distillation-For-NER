import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc

# Define model paths
TEACHER_MODEL_ID = "google/gemma-3-12b-it"
BASE_STUDENT_MODEL_ID = "google/gemma-3-270m-it"
# Update this to point to the actual saved directory of your distilled model
DISTILLED_MODEL_DIR = "./ner_distilled_model" 

INPUT_CSV = r"C:\Users\Raja\Coriolis\Final_Distillation\Test Data\test_ner_prompts_sampled_100.csv"
OUTPUT_CSV = r"C:\Users\Raja\Coriolis\Final_Distillation\Test Data\test_ner_prompts_sampled_100_with_outputs.csv"

def generate_outputs_for_model(model_name_or_path, prompts, device):
    """Loads a model, generates outputs for all prompts, and unloads the model to free VRAM."""
    print(f"\n[{model_name_or_path}] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if "12b" in model_name_or_path.lower() else None
    )
    
    if "12b" not in model_name_or_path.lower():
        model = model.to(device)
        
    model.eval()
    
    outputs = []
    print(f"[{model_name_or_path}] Generating outputs...")
    
    for prompt in tqdm(prompts, desc=f"Generating ({model_name_or_path.split('/')[-1]})"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
            )
            
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        outputs.append(gen_text.strip())
        
    # Free VRAM
    print(f"[{model_name_or_path}] Unloading model and clearing VRAM...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return outputs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the data
    print(f"Loading data from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    if "prompt" not in df.columns:
        raise ValueError("The input CSV must contain a 'prompt' column.")
        
    prompts = df["prompt"].tolist()
    
    # 2 & 3. Generate and store outputs sequentially to avoid OOM errors
    models_to_run = [
        ("Teacher_Output", TEACHER_MODEL_ID),
        ("Base_Student_Output", BASE_STUDENT_MODEL_ID),
        ("Distilled_Student_Output", DISTILLED_MODEL_DIR)
    ]
    
    for col_name, model_path in models_to_run:
        if "Distilled" in col_name and not os.path.exists(model_path):
            print(f"\n[WARNING] Distilled model path '{model_path}' not found! Skipping generation for distilled model.")
            df[col_name] = "MODEL_NOT_FOUND"
            continue
            
        df[col_name] = generate_outputs_for_model(model_path, prompts, device)

    # 4. Save the results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccessfully generated outputs and saved to:\n{OUTPUT_CSV}")

if __name__ == "__main__":
    main()
