#!/usr/bin/env python3
"""
NER Synthetic Dataset Generator
================================
Uses Gemma 3 12B (via Hugging Face Transformers) to generate 5,000
high-quality Named Entity Recognition samples distributed equally
across zero-shot, one-shot, and two-shot prompt modes.

Output Format (JSONL):
    {
      "type": "zero_shot" | "one_shot" | "two_shot",
      "sentence": "<target sentence>",
      "examples": [{"sentence": "...", "entities": [...]}],   # empty for zero-shot
      "expected_entities": [{"entity": "...", "type": "..."}]
    }

Usage:
    python generate_ner_dataset.py --output ner_dataset.jsonl
    python generate_ner_dataset.py --resume               # continue from checkpoint
"""

import argparse
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import openai
except ImportError:
    openai = None

try:
    from tqdm import tqdm
except ImportError:
    # graceful fallback if tqdm is missing
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
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Model
    model_name: str = "google/gemma-3-12b-it"
    device: str = "auto"                    # "auto" | "cuda" | "cpu"
    dtype: str = "bfloat16"               # "bfloat16" | "float16" | "float32"
    max_new_tokens: int = 512
    temperature: float = 0.85              # slight randomness for diversity
    do_sample: bool = True
    top_p: float = 0.92

    # Dataset
    total_samples: int = 3000              # must be divisible by 3
    seed: int = 42

    # Generation
    batch_size: int = 4                    # prompts per forward pass
    max_retries: int = 3                   # retries per sample on parse failure
    checkpoint_every: int = 100            # save progress every N base samples

    # Paths
    output_path: str = "ner_dataset.jsonl"
    checkpoint_path: str = "ner_checkpoint.jsonl"

    # Entity types supported (Ultra-Rich Fine-grained)
    entity_types: list = field(default_factory=lambda: [
        "PERSON", "FICTIONAL_CHARACTER", "DEITY", "HISTORICAL_FIGURE",
        "COMPANY", "STARTUP", "GOV_AGENCY", "MILITARY_UNIT", "SPORT_TEAM", "POLITICAL_PARTY", "NEWS_AGENCY", "BAND", "NGO", "UNIVERSITY", "RESEARCH_INSTITUTE",
        "COUNTRY", "CITY", "STATE", "COUNTY", "CONTINENT", "PLANET", "STAR", "GALAXY", "RIVER", "MOUNTAIN", "LAKE", "OCEAN", "ISLAND", "DESERT",
        "HOSPITAL", "CLINIC", "AIRPORT", "TRAIN_STATION", "SEAPORT", "MUSEUM", "STADIUM", "THEATER", "BRIDGE", "HIGHWAY", "MONUMENT", "POWER_PLANT", 
        "DISEASE", "SYMPTOM", "DRUG", "VACCINE", "MEDICAL_PROCEDURE", "BODY_PART", "CHEMICAL", "ELEMENT", "PROTEIN", "GENE", "BACTERIA", "VIRUS", "ANIMAL_SPECIES", "PLANT_SPECIES", "MINERAL", 
        "SOFTWARE", "OS", "VIDEO_GAME", "PROGRAMMING_LANGUAGE", "AI_MODEL", "HARDWARE_DEVICE", "VEHICLE_MODEL", "AIRCRAFT_MODEL", "SPACECRAFT", "WEAPON", "CONSUMER_PRODUCT", 
        "BOOK", "MOVIE", "TV_SHOW", "SONG", "ALBUM", "PAINTING", "LANGUAGE", "RELIGION", 
        "HISTORICAL_EVENT", "WAR", "BATTLE", "SPORT_EVENT", "TOURNAMENT", "FESTIVAL", "CONFERENCE", "NATURAL_DISASTER", 
        "LAW", "TREATY", "AWARD", "SCIENTIFIC_THEORY", "MATHEMATICAL_THEOREM", "CRYPTOCURRENCY", "STOCK_TICKER",
        "DATE", "TIME", "MONEY", "PERCENT", "TEMPERATURE", "MEASUREMENT"
    ])

    @property
    def samples_per_mode(self) -> dict[str, int]:
        """Split total_samples equally across three modes. Any remainder goes to zero_shot."""
        base, rem = divmod(self.total_samples, 3)
        return {
            "zero_shot": base + rem,   # absorbs any remainder so one/two stay equal
            "one_shot":  base,
            "two_shot":  base,
        }

    @property
    def total_base_sentences(self) -> int:
        """Unique sentences needed (= total_samples; each sample has its own sentence)."""
        return self.total_samples


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

# The generation prompt asks the model to produce a sentence AND its entities
# in a single structured JSON response.  This avoids a second model call for
# annotation and ensures entity spans are always present verbatim.

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
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: Config):
    """Load Gemma 3 12B and its tokenizer from Hugging Face Hub."""
    log.info("Loading tokenizer from '%s' …", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for batched generation

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

    log.info("Loading model '%s' (dtype=%s, device=%s) …", cfg.model_name, cfg.dtype, cfg.device)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map=cfg.device,
        dtype=torch_dtype,
    )
    model.eval()
    log.info("Model loaded. Parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# JSON Extraction Helpers
# ─────────────────────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def extract_json(text: str) -> Optional[dict]:
    """
    Extract and parse the first JSON object found in model output.
    Returns None on failure.
    """
    # Strip common markdown code fences
    text = re.sub(r'```(?:json)?', '', text).strip()

    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        # Attempt to recover a truncated JSON by appending ]}
        try:
            return json.loads(match.group() + "]}")
        except json.JSONDecodeError:
            return None


def validate_sample(data: dict, cfg: Config, require_entities: bool = True) -> bool:
    """
    Validate a parsed sample dict:
      - Has "sentence" (non-empty str)
      - If require_entities, validates "entities" format and verbatim match.
    """
    if not isinstance(data, dict):
        return False
    sentence = data.get("sentence", "")

    if not isinstance(sentence, str) or len(sentence.strip()) < 5:
        return False

    if not require_entities:
        return True

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        return False

    for ent in entities:
        if not isinstance(ent, dict):
            return False
        span = ent.get("entity", "")
        etype = ent.get("type", "").upper()
        if not span or span.lower() not in sentence.lower():
            return False
        if etype not in cfg.entity_types:
            log.warning("Validation failed: hallucinated entity type %r in %r", etype, span)
            return False           # unknown type — reject

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Sentence Generation (Core)
# ─────────────────────────────────────────────────────────────────────────────

def generate_batch(
    prompts: list[str],
    model,
    tokenizer,
    cfg: Config,
) -> list[str]:
    """
    Run a batch of generation prompts through the model.
    Returns a list of decoded strings (one per prompt).
    """
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
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip prompt)
    prompt_len = inputs["input_ids"].shape[1]
    results = []
    for i in range(len(prompts)):
        generated = output_ids[i][prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        results.append(text)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# GPT Few-Shot Example Generation
# ─────────────────────────────────────────────────────────────────────────────

def get_gpt_example_pool(cfg: Config, num_examples: int = 50) -> list[dict]:
    """Uses OpenAI to generate a pristine set of diverse examples."""
    cache_path = Path("gpt_example_pool.jsonl")
    if cache_path.exists():
        log.info("Loading existing GPT example pool from %s", cache_path)
        return _load_jsonl(str(cache_path))
        
    if not openai:
        raise ImportError("openai package is required for Option B. Run: pip install openai")
    
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable is required.")
        
    log.info("Generating %d pristine few-shot examples using GPT-4o-mini...", num_examples)
    client = openai.OpenAI()
    pool = []
    
    schema = {
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
                                        "type": {"type": "string", "enum": list(cfg.entity_types)}
                                    },
                                    "required": ["entity", "type"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["sentence", "entities"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["examples"],
            "additionalProperties": False
        }
    }
    
    batch_size = 10
    system_prompt = (
        "You are an expert NLP annotator. Your task is to generate highly diverse, "
        "complex English sentences and exhaustively tag all named entities. "
        "Entities MUST STRICTLY use the provided types. Do NOT invent new types. "
        "Every span must appear verbatim in the sentence."
    )
    
    needed = num_examples
    while needed > 0:
        req_size = min(batch_size, needed)
        try:
            hint = get_dynamic_hint(cfg.entity_types)
            user_prompt = f"Generate {req_size} sentences covering varied domains. Topic constraint: {hint}"
            response = client.chat.completions.create(
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_schema", "json_schema": schema},
                temperature=0.9
            )
            data = json.loads(response.choices[0].message.content)
            pool.extend(data["examples"])
            needed -= len(data["examples"])
            log.info("  Generated %d / %d GPT examples...", len(pool), num_examples)
        except Exception as e:
            log.warning("GPT API call failed (retrying in 3s): %s", e)
            time.sleep(3)
            
    valid_pool = [x for x in pool if validate_sample(x, cfg)]
    _save_jsonl(valid_pool, str(cache_path))
    return valid_pool


def generate_base_samples(
    n: int,
    model,
    tokenizer,
    cfg: Config,
    gpt_examples: list[dict],
    existing: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Generate `n` unique base samples (sentence + entities) using Gemma 3 12B.

    Strategy:
    - Build generation prompts with rotating variety hints for diversity.
    - Process in batches of cfg.batch_size.
    - Retry failed parses up to cfg.max_retries times.
    - Checkpoint every cfg.checkpoint_every successful samples.

    Returns:
        List of dicts: [{"sentence": str, "entities": list[dict]}, ...]
    """
    samples: list[dict] = list(existing or [])
    seen_sentences: set[str] = {s["sentence"].lower()[:80] for s in samples}

    needed = n - len(samples)
    log.info("Need to generate %d more base samples (already have %d).", needed, len(samples))

    attempts = 0
    max_attempts = needed * cfg.max_retries * 2

    # Provide a visual progress bar if tqdm is installed
    pbar = tqdm(total=n, initial=len(samples), desc="Generating Sentences") if tqdm else None

    while len(samples) < n and attempts < max_attempts:
        # ── Build a batch of prompts ───────────────────────────────────────
        batch_size = min(cfg.batch_size, n - len(samples))
        prompts = []
        hints_used = []
        for i in range(batch_size):
            hint = get_dynamic_hint(cfg.entity_types)
            hints_used.append(hint)
            prompts.append(build_generation_prompt(hint, tokenizer))

        # ── Model inference ───────────────────────────────────────────────
        try:
            raw_outputs = generate_batch(prompts, model, tokenizer, cfg)
        except Exception as exc:
            log.warning("Batch inference failed: %s — skipping batch.", exc)
            attempts += batch_size
            continue

        # ── Parse & validate each output ──────────────────────────────────
        for raw_idx, raw in enumerate(raw_outputs):
            attempts += 1
            data = extract_json(raw)
            if data is None:
                log.warning("JSON extraction failed. Raw: %r", raw[:120])
                continue
            if not validate_sample(data, cfg, require_entities=False):
                log.warning("Validation failed for: %r", data)
                continue

            # Normalise
            sentence = data["sentence"].strip()
            canonical = sentence.lower()[:80]
            if canonical in seen_sentences:
                log.debug("Duplicate sentence skipped.")
                continue

            samples.append({"sentence": sentence, "hint": hints_used[raw_idx]})
            seen_sentences.add(canonical)

        # Save the dataset incrementally after EVERY batch
        if len(samples) > 0:
            _save_jsonl(samples, cfg.checkpoint_path)

        if pbar:
            pbar.n = len(samples)
            pbar.refresh()
        else:
            log.info(
                "Progress: %d / %d base samples (attempts=%d) — [Batch appended to checkpoint].",
                len(samples), n, attempts,
            )

    if pbar:
        pbar.close()

    if len(samples) < n:
        log.warning(
            "Only generated %d / %d samples after %d attempts.",
            len(samples), n, attempts,
        )

    return samples[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Few-Shot Example Pool
# ─────────────────────────────────────────────────────────────────────────────

class ExamplePool:
    """
    Maintains a pool of (sentence, entities) pairs from which
    few-shot examples are sampled without replacement per target sample
    to prevent leakage.
    """

    def __init__(self, all_samples: list[dict], seed: int):
        self._pool = all_samples
        self._rng = random.Random(seed)

    def sample_examples(
        self,
        n: int,
        exclude_sentence: str,
        target_entities: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Draw `n` examples from the pool that differ from exclude_sentence.
        Prefers examples that share entity types with the target sentence.
        Returns list of {"sentence": ..., "entities": [...]} dicts.
        """
        if target_entities is None:
            target_entities = []
            
        target_types = {e["type"] for e in target_entities}

        # Prefer entity-bearing examples
        candidates = [
            s for s in self._pool
            if s["sentence"] != exclude_sentence and len(s["entities"]) > 0
        ]
        if len(candidates) < n:
            # Fall back to all (including entity-free)
            candidates = [
                s for s in self._pool
                if s["sentence"] != exclude_sentence
            ]

        if len(candidates) < n:
            return []

        # Score candidates by how many entity types they share with the target
        if target_types:
            def score(candidate):
                cand_types = {e["type"] for e in candidate["entities"]}
                return len(cand_types.intersection(target_types))
            
            # Sort descending by score, using random to break ties
            scored_candidates = [(score(c), self._rng.random(), c) for c in candidates]
            scored_candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
            chosen = [c for _, _, c in scored_candidates[:n]]
        else:
            chosen = self._rng.sample(candidates, n)

        return [{"sentence": c["sentence"], "entities": c["entities"]} for c in chosen]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_dataset(
    base_samples: list[dict],
    gpt_examples: list[dict],
    cfg: Config,
    seed: int,
) -> list[dict]:
    """
    Distribute base_samples across zero/one/two-shot modes and add
    few-shot examples where appropriate.

    Returns list of fully-formatted dataset records.
    """
    rng = random.Random(seed)
    shuffled = list(base_samples)
    rng.shuffle(shuffled)

    counts = cfg.samples_per_mode
    zero_targets  = shuffled[: counts["zero_shot"]]
    one_targets   = shuffled[counts["zero_shot"] : counts["zero_shot"] + counts["one_shot"]]
    two_targets   = shuffled[counts["zero_shot"] + counts["one_shot"] :]

    pool = ExamplePool(gpt_examples, seed=seed + 1)
    records: list[dict] = []

    # Helper to extract intended entity types from the prompt hint
    def get_target_entities(hint_str: str) -> list[dict]:
        import re
        words = re.findall(r'\b[A-Z_]+\b', hint_str)
        return [{"type": w} for w in words if w in cfg.entity_types]

    # ── Zero-shot ──────────────────────────────────────────────────────────
    for s in zero_targets:
        records.append({
            "type":             "zero_shot",
            "sentence":         s["sentence"],
            "examples":         [],
        })

    # ── One-shot ───────────────────────────────────────────────────────────
    for s in one_targets:
        target_ents = get_target_entities(s.get("hint", ""))
        examples = pool.sample_examples(1, exclude_sentence=s["sentence"], target_entities=target_ents)
        records.append({
            "type":             "one_shot",
            "sentence":         s["sentence"],
            "examples":         examples,
        })

    # ── Two-shot ───────────────────────────────────────────────────────────
    for s in two_targets:
        target_ents = get_target_entities(s.get("hint", ""))
        examples = pool.sample_examples(2, exclude_sentence=s["sentence"], target_entities=target_ents)
        records.append({
            "type":             "two_shot",
            "sentence":         s["sentence"],
            "examples":         examples,
        })

    # Final shuffle so modes are interleaved
    rng.shuffle(records)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# I/O Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_jsonl(data: list[dict], path: str) -> None:
    """Write list of dicts to a JSONL file (overwrites)."""
    with open(path, "w", encoding="utf-8") as fh:
        for record in data:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_checkpoint(cfg: Config) -> list[dict]:
    """Return existing base samples from checkpoint, or empty list."""
    if Path(cfg.checkpoint_path).exists():
        data = _load_jsonl(cfg.checkpoint_path)
        # Checkpoint stores base samples (sentence only)
        valid = [d for d in data if "sentence" in d]
        log.info("Resumed from checkpoint: %d base samples loaded.", len(valid))
        return valid
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(records: list[dict]) -> None:
    """Print a summary of the generated dataset."""
    from collections import Counter

    mode_counts = Counter(r["type"] for r in records)
    entity_type_counts: Counter = Counter()
    entity_bearing = 0
    entity_free = 0

    for r in records:
        ents = r.get("expected_entities", [])
        if ents:
            entity_bearing += 1
            for e in ents:
                entity_type_counts[e["type"]] += 1
        else:
            entity_free += 1

    log.info("=" * 60)
    log.info("DATASET STATISTICS")
    log.info("=" * 60)
    log.info("Total records   : %d", len(records))
    log.info("Mode distribution:")
    for mode, count in sorted(mode_counts.items()):
        log.info("  %-12s : %d", mode, count)
    log.info("Entity-bearing  : %d (%.1f%%)", entity_bearing, 100 * entity_bearing / len(records))
    log.info("Entity-free     : %d (%.1f%%)", entity_free, 100 * entity_free / len(records))
    log.info("Entity type distribution:")
    for etype, count in entity_type_counts.most_common():
        log.info("  %-20s : %d", etype, count)
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# Default path for pre-generated sentences (used when --from-sentences is not explicitly set)
_DEFAULT_SENTENCES_PATH = "Training Data/Generated_Sentences.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NER Synthetic Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Fast path (no Gemma) — uses pre-generated sentences:\n"
            "  python generate_ner_dataset.py --output ner_dataset.jsonl\n\n"
            "  # Explicit sentences file:\n"
            "  python generate_ner_dataset.py --from-sentences my_sentences.jsonl\n\n"
            "  # Force Gemma sentence generation:\n"
            "  python generate_ner_dataset.py --generate\n"
        ),
    )
    p.add_argument("--output",        default="ner_dataset.jsonl",    help="Output JSONL path")
    p.add_argument("--checkpoint",    default="ner_checkpoint.jsonl", help="Checkpoint path")
    p.add_argument("--model",         default="google/gemma-3-12b-it",help="HF model name or path (Gemma path only)")
    p.add_argument("--total",         type=int, default=3000,         help="Total samples (divisible by 3)")
    p.add_argument("--batch-size",    type=int, default=4,            help="Inference batch size (Gemma path only)")
    p.add_argument("--seed",          type=int, default=42,           help="Random seed")
    p.add_argument("--resume",        action="store_true",            help="Resume from checkpoint (Gemma path only)")
    p.add_argument("--dtype",         default="bfloat16",             choices=["bfloat16","float16","float32"])
    p.add_argument("--temperature",   type=float, default=0.85,       help="Sampling temperature (Gemma path only)")
    p.add_argument("--max-new-tokens",type=int, default=256,          help="Max new tokens per generation (Gemma path only)")
    p.add_argument(
        "--from-sentences",
        default=None,
        metavar="PATH",
        help=(
            f"Path to a pre-generated sentences JSONL file. "
            f"Defaults to '{_DEFAULT_SENTENCES_PATH}' if it exists. "
            "When active, Gemma is NOT loaded — GPT-5.4 is used only for "
            "the few-shot example pool."
        ),
    )
    p.add_argument(
        "--generate",
        action="store_true",
        help=(
            "Force Gemma sentence generation even if a pre-generated "
            "sentences file exists. Overrides --from-sentences."
        ),
    )
    import sys
    if "ipykernel" in sys.modules:
        return p.parse_args(args=[])
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    cfg = Config(
        model_name=args.model,
        total_samples=args.total,
        batch_size=args.batch_size,
        seed=args.seed,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        dtype=args.dtype,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # Validate total divisible by 3 (remainder absorbed into zero_shot)
    if cfg.total_samples % 3 != 0:
        log.warning(
            "total_samples=%d is not divisible by 3; %d extra sentence(s) will be "
            "absorbed into zero_shot.",
            cfg.total_samples,
            cfg.total_samples % 3,
        )

    random.seed(cfg.seed)
    log.info("Config: %s", cfg)
    log.info("Samples per mode: %s", cfg.samples_per_mode)

    # ── Resolve which path to take ────────────────────────────────────────
    # Priority:
    #   1. --generate flag  →  always use Gemma (full path)
    #   2. --from-sentences <path>  →  use that file (fast path)
    #   3. default sentences file exists  →  fast path automatically
    #   4. nothing  →  fall through to Gemma (full path)

    sentences_path: Optional[str] = None

    if args.generate:
        log.info("--generate flag set: Gemma sentence generation will be used.")
    else:
        # Resolve sentences file: explicit arg takes priority, then default
        candidate = args.from_sentences or _DEFAULT_SENTENCES_PATH
        if Path(candidate).exists():
            sentences_path = candidate
            log.info(
                "Pre-generated sentences file found at '%s' — using FAST PATH "
                "(Gemma will NOT be loaded). Pass --generate to override.",
                sentences_path,
            )
        elif args.from_sentences:
            # User explicitly requested a file but it doesn't exist
            log.error("Sentences file not found: %s", args.from_sentences)
            return
        else:
            log.info(
                "No pre-generated sentences file found at '%s'. "
                "Falling back to Gemma sentence generation.",
                _DEFAULT_SENTENCES_PATH,
            )

    # ════════════════════════════════════════════════════════════════════
    # FAST PATH  ──  pre-generated sentences file available
    # GPT-5.4 is used ONLY for the few-shot example pool. Gemma is never loaded.
    # ════════════════════════════════════════════════════════════════════
    if sentences_path is not None:
        log.info("[FAST PATH] Loading sentences from '%s' …", sentences_path)
        base_samples = _load_jsonl(sentences_path)
        base_samples = [
            s for s in base_samples
            if isinstance(s.get("sentence"), str) and s["sentence"].strip()
        ]
        log.info("  Loaded %d valid sentences.", len(base_samples))

        if len(base_samples) == 0:
            log.error("No valid sentences found in '%s'. Exiting.", sentences_path)
            return

        # Honour --total; warn if the file has fewer sentences than requested
        if len(base_samples) < cfg.total_samples:
            log.warning(
                "File has only %d sentences but --total=%d was requested. "
                "Using all %d sentences.",
                len(base_samples), cfg.total_samples, len(base_samples),
            )
            cfg = Config(
                **{k: v for k, v in vars(cfg).items() if k != "total_samples"},
                total_samples=len(base_samples),
            )
        else:
            base_samples = base_samples[: cfg.total_samples]

        log.info(
            "Mode split —  zero_shot: %d   one_shot: %d   two_shot: %d",
            cfg.samples_per_mode["zero_shot"],
            cfg.samples_per_mode["one_shot"],
            cfg.samples_per_mode["two_shot"],
        )

        # Build GPT-5.4 few-shot example pool (cached in gpt_example_pool.jsonl)
        log.info("Building GPT-5.4 few-shot example pool …")
        gpt_examples = get_gpt_example_pool(cfg, num_examples=500)
        log.info("  GPT-5.4 pool ready: %d examples.", len(gpt_examples))

        log.info("Assembling dataset into zero/one/two-shot records …")
        records = assemble_dataset(base_samples, gpt_examples, cfg, seed=cfg.seed)

        _save_jsonl(records, cfg.output_path)
        log.info("Saved %d records to '%s'.", len(records), cfg.output_path)
        print_stats(records)
        return   # ── FAST PATH complete; Gemma was never loaded ──

    # ════════════════════════════════════════════════════════════════════
    # FULL PATH  ──  generate sentences with Gemma, then assemble
    # ════════════════════════════════════════════════════════════════════

    # ── Load checkpoint if resuming ────────────────────────────────────────
    existing_base: list[dict] = []
    if args.resume:
        existing_base = load_checkpoint(cfg)

    # ── GPT-5.4 Example Pool ──────────────────────────────────────────────
    gpt_examples = get_gpt_example_pool(cfg, num_examples=500)

    # ── Load Gemma model ───────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg)

    # ── Generate base samples ──────────────────────────────────────────────
    t0 = time.time()
    base_samples = generate_base_samples(
        n=cfg.total_base_sentences,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        gpt_examples=gpt_examples,
        existing=existing_base,
    )
    elapsed = time.time() - t0
    log.info(
        "Base sample generation complete: %d samples in %.1f s (%.2f s/sample).",
        len(base_samples), elapsed, elapsed / max(len(base_samples), 1),
    )

    if len(base_samples) == 0:
        log.error("No samples generated. Exiting.")
        return

    # Trim to total if too many generated due to retries
    base_samples = base_samples[: cfg.total_base_sentences]

    # ── Assemble final dataset ─────────────────────────────────────────────
    log.info("Assembling dataset into zero/one/two-shot records …")
    records = assemble_dataset(base_samples, gpt_examples, cfg, seed=cfg.seed)

    # ── Save output ────────────────────────────────────────────────────────
    _save_jsonl(records, cfg.output_path)
    log.info("Saved %d records to '%s'.", len(records), cfg.output_path)

    # ── Print statistics ───────────────────────────────────────────────────
    print_stats(records)

    # Remove checkpoint after successful completion
    # if Path(cfg.checkpoint_path).exists():
    #     Path(cfg.checkpoint_path).unlink()
    #     log.info("Checkpoint removed (generation complete).")


if __name__ == "__main__":
    main()
