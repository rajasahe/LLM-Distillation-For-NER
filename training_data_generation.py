#!/usr/bin/env python3
"""
training_data_generation.py
===================
NER distillation dataset builder with Gemma 3 12B Base Sentence Generation.

Generates 100 new sentences using Gemma 3 12B, then creates an equally
distributed (zero_shot, one_shot, two_shot) dataset where one-shot and
two-shot examples are generated via GPT-5.4.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import openai
except ImportError:
    raise SystemExit("openai package not found. Run: pip install openai")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Entity Type Schema
# ─────────────────────────────────────────────────────────────────────────────
ENTITY_TYPES: list[str] = [
    "PERSON", "FICTIONAL_CHARACTER", "DEITY", "HISTORICAL_FIGURE",
    "COMPANY", "STARTUP", "GOV_AGENCY", "MILITARY_UNIT", "SPORT_TEAM",
    "POLITICAL_PARTY", "NEWS_AGENCY", "BAND", "NGO", "UNIVERSITY", "RESEARCH_INSTITUTE",
    "COUNTRY", "CITY", "STATE", "COUNTY", "CONTINENT", "PLANET", "STAR", "GALAXY",
    "RIVER", "MOUNTAIN", "LAKE", "OCEAN", "ISLAND", "DESERT",
    "HOSPITAL", "CLINIC", "AIRPORT", "TRAIN_STATION", "SEAPORT", "MUSEUM",
    "STADIUM", "THEATER", "BRIDGE", "HIGHWAY", "MONUMENT", "POWER_PLANT",
    "DISEASE", "SYMPTOM", "DRUG", "VACCINE", "MEDICAL_PROCEDURE", "BODY_PART",
    "CHEMICAL", "ELEMENT", "PROTEIN", "GENE", "BACTERIA", "VIRUS",
    "ANIMAL_SPECIES", "PLANT_SPECIES", "MINERAL",
    "SOFTWARE", "OS", "VIDEO_GAME", "PROGRAMMING_LANGUAGE", "AI_MODEL",
    "HARDWARE_DEVICE", "VEHICLE_MODEL", "AIRCRAFT_MODEL", "SPACECRAFT", "WEAPON", "CONSUMER_PRODUCT",
    "BOOK", "MOVIE", "TV_SHOW", "SONG", "ALBUM", "PAINTING", "LANGUAGE", "RELIGION",
    "HISTORICAL_EVENT", "WAR", "BATTLE", "SPORT_EVENT", "TOURNAMENT",
    "FESTIVAL", "CONFERENCE", "NATURAL_DISASTER",
    "LAW", "TREATY", "AWARD", "SCIENTIFIC_THEORY", "MATHEMATICAL_THEOREM",
    "CRYPTOCURRENCY", "STOCK_TICKER",
    "DATE", "TIME", "MONEY", "PERCENT", "TEMPERATURE", "MEASUREMENT",
]
ENTITY_TYPES_SET = set(ENTITY_TYPES)

# ─────────────────────────────────────────────────────────────────────────────
# Gemma Prompt Generation
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_HEADER = (
    "You are an expert NER dataset creator. "
    "Generate diverse, natural English sentences with accurate named entity annotations."
)

GENERATION_PROMPT_TEMPLATE = """\
Generate a single, diverse, natural English sentence.

- IMPORTANT: Provide the JSON response DIRECTLY. Do absolutely NOT say "Here is the JSON" or output any text before the dictionary.
- You MUST output EXACTLY ONE JSON dictionary and absolutely nothing else.
- The JSON must have exactly one key: "sentence".
- Sentence topic constraint: {variety_hint}
- Sentence style constraint: MUST be written like a {style_hint}
- CRITICAL RULE: You must gracefully weave the real-world entities into the sentence naturally. You must STRICTLY NEVER type the literal uppercase entity category names in your output! (e.g., do NOT type "(HIGHWAY)" or "the TV_SHOW").

Example acceptable output:
{{"sentence": "Your generated sentence goes here."}}
"""

def get_dynamic_hint(entity_types: list) -> str:
    """Generate an infinitely diverse prompt hint by randomly selecting entity types."""
    import random
    num_entities = random.choices([0, 1, 2, 3, 4], weights=[0.1, 0.2, 0.3, 0.3, 0.1])[0]
    if num_entities == 0:
        return "a completely random everyday sentence with NO named entities"
    chosen_types = random.sample(entity_types, num_entities)
    return f"Write about specific, real-world examples belonging to these categories: {', '.join(chosen_types)}. (Remember: you must NOT copy these uppercase category names into the text!)."

def build_generation_prompt(variety_hint: str, tokenizer) -> str:
    """Return the fully formatted prompt using the chat template."""
    import random
    styles = [
        "news headline or journalistic report", 
        "casual text message or social media post", 
        "formal corporate or legal document", 
        "fiction book excerpt or descriptive narrative", 
        "scientific abstract or medical journal", 
        "direct quote or dialogue between people",
        "interrogative question or inquiry", 
        "urgent emergency alert or broadcast", 
        "historical archive or biography snippet", 
        "technical manual instruction", 
        "personal diary or travel blog entry", 
        "financial earnings call excerpt"
    ]
    style_hint = random.choice(styles)

    content = SYSTEM_HEADER + "\n\n" + GENERATION_PROMPT_TEMPLATE.format(
        variety_hint=variety_hint,
        style_hint=style_hint
    )
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ─────────────────────────────────────────────────────────────────────────────
# GPT Prompts
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert NLP annotator specialising in Named Entity Recognition.

Given a TARGET SENTENCE and a list of relevant entity types, produce 2 short,
diverse EXAMPLE sentences that together demonstrate as many of those entity
types as possible.  Annotate every named entity in each example.

HARD RULES:
- Entity types MUST come from the provided enum — never invent new types.
- Every entity span must appear verbatim (case-insensitive) in its sentence.
- Keep example sentences short (< 40 words) so they fit in the few-shot prompt.
- Make examples topically relevant to the target sentence's entity types.
- Never copy the target sentence itself as an example.

CANONICAL LABEL RULES (these override any other interpretation):
- Continents → CONTINENT (never COUNTRY)
- Reuters, CNN, BBC, AP, Bloomberg → NEWS_AGENCY (never COMPANY)
- Pure elements (iron, gold, carbon …) → ELEMENT (never CHEMICAL)
- Chemical compounds → CHEMICAL
- Insulin, hemoglobin, actin … → PROTEIN (even when therapeutic)
- Museums → MUSEUM
- Painters/artists (as people) → PERSON
- Artworks / paintings → PAINTING
- Historical deceased figures → HISTORICAL_FIGURE (never PERSON)
- Cities → CITY (never STATE or GOV_AGENCY)
- Countries → COUNTRY (never GOV_AGENCY)
- Sports players → PERSON (never SPORT_TEAM)
- Spacecraft → SPACECRAFT
"""

REVIEW_PROMPT = """
Review every annotation you just produced.  For EACH entity check:
1. Does the span appear verbatim in the sentence?  Remove it if not.
2. Does the type follow ALL canonical label rules in the system prompt?
Return corrected JSON only — same schema, no extra commentary.
"""

# JSON schema for structured output
RESPONSE_SCHEMA: dict = {
    "name": "ner_examples",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sentence": {"type": "string"},
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity": {"type": "string"},
                                    "type":   {"type": "string", "enum": ENTITY_TYPES},
                                },
                                "required": ["entity", "type"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["sentence", "entities"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["examples"],
        "additionalProperties": False,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)

def extract_json(text: str) -> Optional[dict]:
    text = re.sub(r'```(?:json)?', '', text).strip()
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        try:
            return json.loads(match.group() + "]}")
        except json.JSONDecodeError:
            return None

def _load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d in '%s': %s", i, path, exc)
    return records

def _append_jsonl(record: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

def _save_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

def extract_types_from_hint(hint: str) -> set[str]:
    words = re.findall(r"\b[A-Z_]{2,}\b", hint)
    return {w for w in words if w in ENTITY_TYPES_SET}

def validate_example(ex: dict) -> bool:
    sentence = ex.get("sentence", "").strip()
    if len(sentence) < 5:
        return False
    for ent in ex.get("entities", []):
        span  = ent.get("entity", "").strip()
        etype = ent.get("type",   "")
        if not span:
            return False
        if span.lower() not in sentence.lower():
            return False
        if etype not in ENTITY_TYPES_SET:
            return False
    return True

def compute_split(total: int) -> dict[str, int]:
    base, rem = divmod(total, 3)
    return {"zero_shot": base + rem, "one_shot": base, "two_shot": base}

# ─────────────────────────────────────────────────────────────────────────────
# Gemma Sentence Generation
# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_name="google/gemma-3-12b-it"):
    log.info("Loading tokenizer from '%s' …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    log.info("Loading model '%s' (dtype=bfloat16, device=auto) …", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer

def generate_batch(prompts: list[str], model, tokenizer) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device if hasattr(model, 'device') else next(model.parameters()).device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    results = []
    for i in range(len(prompts)):
        generated = output_ids[i][prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        results.append(text)
    return results

def generate_sentences_with_gemma(n: int, batch_size: int, temp_file: str) -> list[dict]:
    if Path(temp_file).exists():
        loaded = _load_jsonl(temp_file)
        if len(loaded) >= n:
            log.info("Loaded %d pre-generated sentences from %s", len(loaded), temp_file)
            return loaded[:n]

    model, tokenizer = load_model_and_tokenizer()
    samples = []
    if Path(temp_file).exists():
        samples = _load_jsonl(temp_file)
        
    seen = set(s["sentence"].lower()[:80] for s in samples)
    attempts = 0
    max_attempts = n * 5
    
    log.info("Generating %d sentences using Gemma (currently have %d)...", n, len(samples))
    pbar = tqdm(total=n, initial=len(samples), desc="Gemma Generation") if tqdm else None

    while len(samples) < n and attempts < max_attempts:
        bs = min(batch_size, n - len(samples))
        prompts = []
        hints_used = []
        for _ in range(bs):
            hint = get_dynamic_hint(ENTITY_TYPES)
            hints_used.append(hint)
            prompts.append(build_generation_prompt(hint, tokenizer))
            
        try:
            raw_outputs = generate_batch(prompts, model, tokenizer)
        except Exception as exc:
            log.warning("Batch inference failed: %s", exc)
            attempts += bs
            continue
            
        for raw_idx, raw in enumerate(raw_outputs):
            attempts += 1
            data = extract_json(raw)
            if not data or "sentence" not in data:
                continue
            sentence = data["sentence"].strip()
            if len(sentence) < 5: continue
            
            canonical = sentence.lower()[:80]
            if canonical in seen: continue
            
            samples.append({"sentence": sentence, "hint": hints_used[raw_idx]})
            seen.add(canonical)
            
        if pbar:
            pbar.n = len(samples)
            pbar.refresh()
        else:
            log.info("  Generated %d / %d sentences...", len(samples), n)
        
        # Incrementally save
        if len(samples) > 0:
            _save_jsonl(samples, temp_file)
            
    if pbar:
        pbar.close()
    
    log.info("Saved %d base sentences to '%s'", len(samples), temp_file)
    
    # Free up memory
    del model
    torch.cuda.empty_cache()
    
    return samples[:n]

# ─────────────────────────────────────────────────────────────────────────────
# GPT Annotation (for one-shot and two-shot examples)
# ─────────────────────────────────────────────────────────────────────────────

def _call_gpt(
    client: openai.OpenAI,
    target_sentence: str,
    target_types: set[str],
    n_examples: int,
    max_retries: int,
) -> list[dict]:
    type_hint = (
        f"Relevant entity types for context: {', '.join(sorted(target_types))}. "
        if target_types
        else ""
    )
    user_prompt = (
        f"TARGET SENTENCE:\n{target_sentence}\n\n"
        f"{type_hint}"
        f"Generate exactly {n_examples} short, diverse annotated example sentence(s). "
        "Do NOT reproduce the target sentence."
    )

    for attempt in range(1, max_retries + 1):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ]
            resp1 = client.chat.completions.create(
                model="gpt-5.4",
                messages=messages,
                response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
                temperature=0.7,
            )
            draft = resp1.choices[0].message.content

            messages.append({"role": "assistant", "content": draft})
            messages.append({"role": "user",      "content": REVIEW_PROMPT})
            resp2 = client.chat.completions.create(
                model="gpt-5.4",
                messages=messages,
                response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
                temperature=0.0,
            )
            data     = json.loads(resp2.choices[0].message.content)
            examples = data.get("examples", [])
            valid    = [e for e in examples if validate_example(e)]

            valid = [
                e for e in valid
                if e["sentence"].strip().lower() != target_sentence.strip().lower()
            ]

            return valid[:n_examples]

        except Exception as exc:
            wait = 2 ** attempt
            log.warning(
                "GPT error (attempt %d/%d): %s — retrying in %ds …",
                attempt, max_retries, exc, wait,
            )
            time.sleep(wait)

    log.error("All %d retries failed for sentence: %.80s …", max_retries, target_sentence)
    return []

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    output_path:    str,
    total:          int,
    seed:           int,
    batch_size:     int,
    resume:         bool,
    max_retries:    int,
) -> None:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    client = openai.OpenAI()

    # ── Generate base sentences with Gemma ─────────────────────────────────────
    temp_file = "Training Data/Generated_Base_Sentences_Temp.jsonl"
    all_sentences = generate_sentences_with_gemma(total, batch_size=4, temp_file=temp_file)
    
    if not all_sentences:
        raise ValueError("No valid sentences generated.")

    # ── Shuffle & split ───────────────────────────────────────────────────────
    rng = random.Random(seed)
    shuffled = list(all_sentences)
    rng.shuffle(shuffled)
    shuffled = shuffled[:total]

    split = compute_split(total)
    log.info(
        "Mode split — zero_shot: %d   one_shot: %d   two_shot: %d",
        split["zero_shot"], split["one_shot"], split["two_shot"],
    )

    z_end = split["zero_shot"]
    o_end = z_end + split["one_shot"]

    zero_group = [(s, "zero_shot") for s in shuffled[:z_end]]
    one_group  = [(s, "one_shot")  for s in shuffled[z_end:o_end]]
    two_group  = [(s, "two_shot")  for s in shuffled[o_end:]]

    all_items = zero_group + one_group + two_group
    rng.shuffle(all_items)

    # ── Resume: find already-processed sentences ──────────────────────────────
    already_done: set[str] = set()
    if resume and Path(output_path).exists():
        existing = _load_jsonl(output_path)
        already_done = {r["target_sentence"] for r in existing}
        log.info("Resume mode: %d / %d records already written.", len(already_done), len(all_items))
    elif Path(output_path).exists() and not resume:
        log.warning(
            "Output file '%s' already exists. "
            "Use --resume to append to it, or delete it to start fresh. "
            "Exiting without overwriting.",
            output_path,
        )
        return

    # ── Process ───────────────────────────────────────────────────────────────
    total_items   = len(all_items)
    done_count    = len(already_done)
    pending_items = [(s, m) for (s, m) in all_items if s["sentence"] not in already_done]

    log.info("Processing %d sentences (%d already done, %d remaining) …", total_items, done_count, len(pending_items))

    zero_pending  = [(s, m) for (s, m) in pending_items if m == "zero_shot"]
    few_pending   = [(s, m) for (s, m) in pending_items if m != "zero_shot"]

    for s, mode in zero_pending:
        record = {
            "target_sentence": s["sentence"],
            "mode":            mode,
            "examples":        [],
        }
        _append_jsonl(record, output_path)
        done_count += 1

    log.info("Zero-shot records written: %d", len(zero_pending))
    log.info("Processing %d few-shot sentences via GPT-5.4 (batch_size=%d) …", len(few_pending), batch_size)

    pbar_gpt = tqdm(total=len(few_pending), desc="GPT Annotation") if tqdm else None

    for idx, (s, mode) in enumerate(few_pending, 1):
        n_examples = 1 if mode == "one_shot" else 2
        target_types = extract_types_from_hint(s.get("hint", ""))

        examples = _call_gpt(
            client=client,
            target_sentence=s["sentence"],
            target_types=target_types,
            n_examples=n_examples,
            max_retries=max_retries,
        )

        if len(examples) < n_examples:
            log.warning("[%d/%d] Got %d / %d examples for: %.70s …", idx, len(few_pending), len(examples), n_examples, s["sentence"])

        record = {
            "target_sentence": s["sentence"],
            "mode":            mode,
            "examples":        examples[:n_examples],
        }
        _append_jsonl(record, output_path)
        done_count += 1

        if pbar_gpt:
            pbar_gpt.update(1)
        elif idx % batch_size == 0 or idx == len(few_pending):
            log.info("  Progress: %d / %d few-shot done  |  Total written: %d / %d", idx, len(few_pending), done_count, total_items)

    if pbar_gpt:
        pbar_gpt.close()

    log.info("=" * 60)
    log.info("DONE — %d records written to '%s'.", done_count, output_path)
    log.info("=" * 60)

    final = _load_jsonl(output_path)
    from collections import Counter
    mode_counts = Counter(r.get("mode") for r in final)
    log.info("Final distribution:")
    for mode in ("zero_shot", "one_shot", "two_shot"):
        log.info("  %-12s : %d", mode, mode_counts.get(mode, 0))
    log.info("Total          : %d", len(final))

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate mixed zero/one/two-shot NER dataset using Gemma 3 12B for base sentences and GPT-5.4 for few-shot example annotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output",
        default="Training Data/mixed_ner_dataset_final.jsonl",
        help="Path for the output dataset JSONL",
    )
    p.add_argument("--total", type=int, default=100, help="Number of records to produce")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--batch-size", type=int, default=5, help="Log progress every N few-shot sentences")
    p.add_argument("--resume", action="store_true", help="Append to an existing output file instead of starting fresh")
    p.add_argument("--max-retries", type=int, default=5, help="Max GPT API retry attempts per sentence")
    
    return p.parse_args(["--resume"])

if __name__ == "__main__":
    args = parse_args()

    log.info("NER Dataset Generator (Gemma + GPT-5.4)")
    log.info("  Output      : %s", args.output)
    log.info("  Total       : %d", args.total)
    log.info("  Seed        : %d", args.seed)
    log.info("  Batch size  : %d", args.batch_size)
    log.info("  Resume      : %s", args.resume)

    build_dataset(
        output_path=args.output,
        total=args.total,
        seed=args.seed,
        batch_size=args.batch_size,
        resume=args.resume,
        max_retries=args.max_retries,
    )