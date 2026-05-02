#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER Distillation Training Script v2
=====================================
Teacher  : google/gemma-3-12b-it
Student  : google/gemma-3-270m (full fine-tuning — no LoRA)
Dataset  : mixed_ner_dataset_final.jsonl

Prompt Format:
  Extract named entities from the sentence. Return a JSON array only.
  Each item: {"entity": "exact text from sentence", "type": "ENTITY_TYPE"}
  If no entities exist, return [].

  Rules:
  - Copy entity text exactly. Do not change it.
  - Strip leading "a", "an", "the" from entity text only.
  - Skip generic nouns, job titles alone, and abstract concepts.

Training Strategy:
  - Build mode-aware prompts (zero / one / two-shot) from JSONL records
  - Pass each prompt to the teacher to generate NER JSON output
  - Train student on (prompt + teacher output) with combined KL + CE loss
  - KL uses top-k teacher logits from output_logits=True (no second pass)
  - Prompt tokens are masked out of the loss
  - No LoRA / PEFT adapters — entire student model is updated
"""

# ─────────────────────────────────────────────────────────────
# 1. Install Dependencies (uncomment for Colab)
# ─────────────────────────────────────────────────────────────
# !pip install -q transformers accelerate bitsandbytes tqdm

# ─────────────────────────────────────────────────────────────
# 2. Imports
# ─────────────────────────────────────────────────────────────
import json
import os
import random
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# ─────────────────────────────────────────────────────────────
# 3. Mount Google Drive (uncomment for Colab)
# ─────────────────────────────────────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')

# ─────────────────────────────────────────────────────────────
# 4. Configuration
# ─────────────────────────────────────────────────────────────
TEACHER_NAME        = "google/gemma-3-12b-it"
STUDENT_NAME        = "google/gemma-3-270m"

# ── Dataset path (update if running on Colab with Drive) ──────
DATASET_PATH        = r"C:\Users\Raja\Coriolis\Final_Distillation\Training Data\mixed_ner_dataset_final.jsonl"
# DATASET_PATH      = "/content/drive/MyDrive/Distillation/mixed_ner_dataset_final.jsonl"

SAVE_DIR            = "./ner_distilled_model"
CHECKPOINT_DIR      = "./ner_distill_checkpoints"

# ── Training hyperparameters ──────────────────────────────────
EPOCHS              = 10
BATCH_SIZE          = 1   # WARNING: keep at 1 — values > 1 accumulate live
                          # computation graphs and will OOM on most GPUs.
LEARNING_RATE       = 5e-5
WARMUP_RATIO        = 0.10
LAMBDA_KL           = 0.7        # weight for KL loss (1-λ goes to CE)
TOP_K               = 50         # top-k vocab for KL distillation
MAX_LENGTH          = 768        # max total token length (prompt + output)
MAX_NEW_TOKENS      = 256        # teacher generation budget
GRAD_CLIP           = 1.0
SAVE_EVERY_N_STEPS  = 200
SEED                = 42

# ── Early stopping ────────────────────────────────────────────
PATIENCE            = 50     # BATCHES with no improvement before stopping
MIN_DELTA           = 1e-4   # minimum loss drop to count as improvement
DISTILL_TEMP        = 2.0    # Hinton distillation temperature (softens distributions)

random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────
# 5. Prompt Builder
# ─────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = """\
Extract named entities from the sentence. Return a JSON array only.
Each item: {"entity": "exact text from sentence", "type": "ENTITY_TYPE"}
If no entities exist, return [].

Rules:
- Copy entity text exactly. Do not change it.
- Strip leading "a", "an", "the" from entity text only.
- Skip generic nouns, job titles alone, and abstract concepts.\
"""

# (No hardcoded example — examples are drawn from the dataset's own `examples` field,
#  which was populated from the GPT gold-standard pool during dataset generation.)


def build_prompt_from_record(record: dict) -> str:
    """
    Build a mode-aware NER prompt from a dataset record.

    Record schema:
      {
        "target_sentence": str,
        "mode": "zero_shot" | "one_shot" | "two_shot",
        "examples": [{"sentence": str, "entities": list[dict]}, ...]
      }
    """
    lines = [SYSTEM_INSTRUCTION, ""]

    examples = record.get("examples", [])

    if examples:
        lines.append("Examples:")
        for i, ex in enumerate(examples, 1):
            ent_json = json.dumps(ex["entities"], ensure_ascii=False)
            lines.append(f"Example {i}:")
            lines.append(f"Sentence: \"\"\"{ex['sentence']}\"\"\"")
            lines.append(f"Output: {ent_json}")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(f"Sentence: \"\"\"{record['target_sentence']}\"\"\"")
    lines.append("")
    lines.append("Output:")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 6. Dataset Loader
# ─────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {path}")
    return records


# ─────────────────────────────────────────────────────────────
# 7. Model Loading
# ─────────────────────────────────────────────────────────────
def load_models():
    print(f"Loading tokenizer from '{TEACHER_NAME}' ...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading teacher '{TEACHER_NAME}' ...")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    teacher.eval()

    print(f"Loading student '{STUDENT_NAME}' ...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_NAME,
        torch_dtype=torch.bfloat16,
    )

    # (Vocab size guard removed: Gemma 3 config does not expose vocab_size directly)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device)

    print("Models loaded.")
    print(f"  Teacher params : {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"  Student params : {sum(p.numel() for p in student.parameters()):,}")
    return tokenizer, teacher, student, device


# ─────────────────────────────────────────────────────────────
# 8. Teacher Generation (gets logits too)
# ─────────────────────────────────────────────────────────────
def teacher_generate(prompt: str, tokenizer, teacher, device):
    """
    Run the teacher on `prompt`, return:
      - decoded output string
      - stacked logits tensor  [num_generated_tokens, vocab_size]
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH - MAX_NEW_TOKENS,
    ).to(next(teacher.parameters()).device)

    with torch.no_grad():
        out = teacher.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_logits=True,
        )

    gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    # CRITICAL FIX: skip_special_tokens=True strips the EOS token. 
    # If we don't add it back, the student never learns to stop generating!
    if tokenizer.eos_token and not gen_text.endswith(tokenizer.eos_token):
        gen_text += tokenizer.eos_token
    # out.logits is a tuple of [batch, vocab] tensors — one per generated token.
    # Stack and squeeze the batch dim → [T, vocab]
    teacher_logits = torch.stack(out.logits, dim=0).squeeze(1)   # [T, vocab]
    return gen_text, teacher_logits


# ─────────────────────────────────────────────────────────────
# 9. Combined KL + CE Loss
# ─────────────────────────────────────────────────────────────
def compute_loss(student, tokenizer, prompt_text: str,
                 teacher_logits, gen_text: str, device):
    """
    Compute LAMBDA_KL * KL(student || teacher) + (1-LAMBDA_KL) * CE
    over the generated token positions only.
    """
    full_text = prompt_text + gen_text
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    input_ids = enc["input_ids"].to(device)

    # How many prompt tokens?
    prompt_enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    prompt_len = prompt_enc["input_ids"].shape[1]
    total_len  = input_ids.shape[1]
    gen_len    = total_len - prompt_len

    if gen_len <= 0:
        return None  # nothing to learn from

    # Warn if truncated
    if total_len == MAX_LENGTH:
        print(f"  [WARN] Sequence truncated to {MAX_LENGTH} tokens.")

    # Student forward
    student_out  = student(input_ids=input_ids)
    student_logits = student_out.logits[0]  # [total_len, vocab]

    # Slice to generated positions only
    # student predicts token[i+1] at position[i]
    s_logits = student_logits[prompt_len - 1 : total_len - 1]  # [gen_len, vocab]

    # ── CE Loss ──────────────────────────────────────────────
    labels  = input_ids[0, prompt_len:]                          # [gen_len]
    ce_loss = F.cross_entropy(s_logits, labels)

    # ── KL Loss (with Hinton temperature scaling) ──────────────
    t_len   = teacher_logits.shape[0]
    use_len = min(gen_len, t_len)

    if use_len <= 0:
        return ce_loss, ce_loss.item(), 0.0

    t_logits_slice = teacher_logits[:use_len].to(device)     # [use_len, teacher_vocab]
    s_logits_slice = s_logits[:use_len]                       # [use_len, student_vocab]

    # ── Align vocabularies ───────────────────────────────────
    # Gemma 3 teacher/student can have slightly different padded vocab sizes
    # (e.g. 262208 vs 262144). Slice to the minimum vocab size so KL works.
    min_vocab = min(t_logits_slice.shape[-1], s_logits_slice.shape[-1])
    t_logits_slice = t_logits_slice[..., :min_vocab]
    s_logits_slice = s_logits_slice[..., :min_vocab]

    # top-k mask — zero out tail of teacher distribution
    topk_vals, _ = torch.topk(t_logits_slice, TOP_K, dim=-1)
    threshold     = topk_vals[:, -1].unsqueeze(-1)            # [use_len, 1]
    mask          = t_logits_slice >= threshold                # [use_len, vocab]

    t_masked = t_logits_slice.masked_fill(~mask, float("-inf"))

    # Apply temperature — softens teacher's top-k distribution.
    # Note: we DO NOT mask the student. If we mask the student with -inf, 
    # F.kl_div computes 0 * -inf = NaN. Instead, we let the student compute
    # log_softmax over the full vocab. Since t_probs is 0 outside top-k,
    # the KL loss naturally pulls the student's mass toward the top-k tokens.
    T = DISTILL_TEMP
    t_probs     = F.softmax(t_masked       / T, dim=-1)
    s_log_probs = F.log_softmax(s_logits_slice / T, dim=-1)
    kl_loss     = (T ** 2) * F.kl_div(s_log_probs, t_probs, reduction="batchmean")

    # Guard against NaN
    if torch.isnan(kl_loss):
        print("  [WARN] NaN KL loss, using CE only.")
        return ce_loss, ce_loss.item(), 0.0

    total = LAMBDA_KL * kl_loss + (1.0 - LAMBDA_KL) * ce_loss
    return total, ce_loss.item(), kl_loss.item()


# ─────────────────────────────────────────────────────────────
# 10. Training Loop
# ─────────────────────────────────────────────────────────────
def train():
    # Load data
    records = load_jsonl(DATASET_PATH)

    # Load models
    tokenizer, teacher, student, device = load_models()

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)

    total_steps   = math.ceil(len(records) / BATCH_SIZE) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    all_epoch_losses = []
    global_step      = 0
    batch_loss_log   = []

    # ── Early stopping state ──────────────────────────────────
    best_loss        = float("inf")
    no_improve_count = 0
    best_state_dict  = None   # held in CPU RAM to avoid VRAM pressure
    stop_training    = False  # flag to break outer epoch loop

    print(f"\nStarting training: {EPOCHS} epochs, {len(records)} records, "
          f"batch_size={BATCH_SIZE}, patience={PATIENCE} batches")

    for epoch in range(1, EPOCHS + 1):
        if stop_training: break
        random.shuffle(records)
        epoch_loss    = 0.0
        epoch_ce_loss = 0.0
        epoch_kl_loss = 0.0
        valid_steps   = 0

        # ── Epoch banner ─────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  EPOCH {epoch} / {EPOCHS}  |  {len(records)} records  |  "
              f"step {global_step + 1} onwards")
        print(f"{'='*60}")

        pbar = tqdm(range(0, len(records), BATCH_SIZE),
                    desc=f"Ep {epoch}/{EPOCHS}", unit="batch",
                    dynamic_ncols=True)

        for batch_start in pbar:
            batch = records[batch_start : batch_start + BATCH_SIZE]
            optimizer.zero_grad()
            batch_total_loss = 0.0
            batch_count      = 0
            _batch_results   = []   # [(total, ce, kl), ...]

            for record in batch:
                prompt_text = build_prompt_from_record(record)

                # Teacher generation
                gen_text, teacher_logits = teacher_generate(
                    prompt_text, tokenizer, teacher, device
                )

                # Loss
                result = compute_loss(
                    student, tokenizer, prompt_text,
                    teacher_logits, gen_text, device
                )
                if result is None:
                    continue

                loss, ce_l, kl_l = result
                batch_total_loss += loss
                batch_count      += 1
                _batch_results.append((loss.item(), ce_l, kl_l))

            if batch_count == 0:
                continue

            avg_loss = batch_total_loss / batch_count
            avg_loss.backward()

            torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            loss_val   = avg_loss.item()
            # average CE / KL across the batch records
            avg_ce = sum(r[1] for r in _batch_results) / batch_count
            avg_kl = sum(r[2] for r in _batch_results) / batch_count

            epoch_loss    += loss_val
            epoch_ce_loss += avg_ce
            epoch_kl_loss += avg_kl
            valid_steps   += 1
            global_step   += 1

            running_avg = epoch_loss / valid_steps
            pbar.set_postfix({
                "loss":    f"{loss_val:.4f}",
                "CE":      f"{avg_ce:.4f}",
                "KL":      f"{avg_kl:.4f}",
                "avg":     f"{running_avg:.4f}",
                "lr":      f"{scheduler.get_last_lr()[0]:.2e}",
            })

            batch_loss_log.append({
                "epoch": epoch, "step": global_step,
                "loss": loss_val, "ce_loss": avg_ce, "kl_loss": avg_kl,
            })

            # Print current batch loss to console safely without breaking tqdm
            tqdm.write(
                f"  Step {global_step:>4} | Loss: {loss_val:.4f} "
                f"(CE: {avg_ce:.4f}, KL: {avg_kl:.4f})"
            )

            # Checkpoint
            if global_step % SAVE_EVERY_N_STEPS == 0:
                ckpt_path = os.path.join(
                    CHECKPOINT_DIR, f"step_{global_step}"
                )
                student.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                print(f"\n  [Checkpoint] Saved → {ckpt_path}")

            # ── Early stopping check (per batch) ──────────────────
            # We use `running_avg` instead of raw `loss_val` because raw
            # batch losses fluctuate wildly and would trigger early stops randomly.
            if running_avg < best_loss - MIN_DELTA:
                best_loss        = running_avg
                no_improve_count = 0
                # Snapshot best weights to CPU
                best_state_dict  = {k: v.cpu().clone()
                                     for k, v in student.state_dict().items()}
            else:
                no_improve_count += 1
                if no_improve_count >= PATIENCE:
                    tqdm.write(f"\n  [EarlyStopping] No improvement for {PATIENCE} batches. Stopping early!")
                    stop_training = True
                    break

        mean_epoch_loss = epoch_loss    / max(valid_steps, 1)
        mean_epoch_ce   = epoch_ce_loss / max(valid_steps, 1)
        mean_epoch_kl   = epoch_kl_loss / max(valid_steps, 1)
        all_epoch_losses.append(mean_epoch_loss)

        # ── Epoch summary table ───────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch} Summary")
        print(f"{'─'*60}")
        print(f"  {'Metric':<20} {'Value':>10}")
        print(f"  {'─'*30}")
        print(f"  {'Mean Total Loss':<20} {mean_epoch_loss:>10.4f}")
        print(f"  {'Mean CE Loss':<20} {mean_epoch_ce:>10.4f}")
        print(f"  {'Mean KL Loss':<20} {mean_epoch_kl:>10.4f}")
        print(f"  {'Best Loss So Far':<20} {min(best_loss, mean_epoch_loss):>10.4f}")
        print(f"  {'Steps Completed':<20} {global_step:>10,}")
        print(f"{'─'*60}")

    # ── Restore best weights before saving ───────────────────
    if best_state_dict is not None:
        print("\nRestoring best model weights ...")
        student.load_state_dict(
            {k: v.to(next(student.parameters()).device)
             for k, v in best_state_dict.items()}
        )

    # ── Save logs to disk ─────────────────────────────────────
    log_path = os.path.join(SAVE_DIR, "training_loss_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(batch_loss_log, f, indent=2)
    print(f"\n  [Logs] Saved batch loss history to {log_path}")

    # ── Final training report ─────────────────────────────────
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  {'Epoch':<8} {'Total Loss':>12} {'Status':>12}")
    print(f"  {'─'*34}")
    for i, el in enumerate(all_epoch_losses, 1):
        marker = " ← best" if el == min(all_epoch_losses) else ""
        print(f"  {i:<8} {el:>12.4f}{marker}")
    print(f"  {'─'*34}")
    print(f"  Best loss     : {best_loss:.4f}")
    print(f"  Total steps   : {global_step:,}")
    print(f"  Saved to      : {SAVE_DIR}")
    print(f"{'='*60}")

    return student, tokenizer, all_epoch_losses, batch_loss_log


# ─────────────────────────────────────────────────────────────
# 11. Quick Inference Test
# ─────────────────────────────────────────────────────────────
def inference_test(model, tokenizer, device):
    test_record = {
        "target_sentence": (
            "Spectroscopic analysis of samples from the Allende meteorite revealed "
            "significant concentrations of olivine and pyroxene, suggesting a complex "
            "origin within the early solar nebula and providing insights into the "
            "formation of terrestrial mantle minerals such as peridotite."
        ),
        "mode": "one_shot",
        "examples": [
            {
                "sentence": "Geologists found quartz and feldspar in the basalt core.",
                "entities": [
                    {"entity": "quartz",   "type": "MINERAL"},
                    {"entity": "feldspar", "type": "MINERAL"},
                ]
            }
        ]
    }

    prompt = build_prompt_from_record(test_record)
    print("\n" + "="*60)
    print("INFERENCE TEST")
    print("="*60)
    print("Prompt:\n", prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
        )
    gen_ids  = out[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print("\nModel Output:\n", gen_text)


# ─────────────────────────────────────────────────────────────
# 12. Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    student_model, tok, epoch_losses, step_logs = train()
    device = next(student_model.parameters()).device
    inference_test(student_model, tok, device)
