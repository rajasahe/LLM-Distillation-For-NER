#!/usr/bin/env python3
"""
generate_dataset.py
===================
Single-pass NER distillation dataset builder.

Reads  : Training Data/Generated_Sentences.jsonl
         Each line: {"sentence": "...", "hint": "..."}

Writes : Training Data/mixed_ner_dataset.jsonl
         Each line:
           {
               "target_sentence": "...",
               "mode":            "zero_shot" | "one_shot" | "two_shot",
               "examples":        []                              # zero_shot
                               | [{"sentence": ..., "entities": [...]}]  # one_shot
                               | [{...}, {...}]                           # two_shot
           }

Distribution (for 3 000 sentences): 1 000 zero-shot, 1 000 one-shot, 1 000 two-shot.

Strategy
--------
*  Sentences are shuffled once (--seed) then split into three equal groups.
*  Zero-shot sentences require no API call.
*  One-shot and two-shot sentences are sent to GPT-5.4 in small batches.
   GPT is asked to produce annotated examples whose entity types match the
   hint of the target sentence (relevance-guided selection).
*  A two-turn generate → self-review pattern (same as generate_fewshot_pool.py)
   ensures annotation quality.
*  The script is fully resumable: run it again with --resume and it will skip
   sentences already processed, appending only the remainder.

Usage
-----
    # Set your key first:
    set OPENAI_API_KEY=sk-...          (Windows)
    export OPENAI_API_KEY=sk-...       (Linux/Mac)

    python generate_dataset.py
    python generate_dataset.py --resume
    python generate_dataset.py \\
        --sentences "Training Data/Generated_Sentences.jsonl" \\
        --output    "Training Data/mixed_ner_dataset.jsonl" \\
        --total 3000 --seed 42 --batch-size 5
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

try:
    import openai
except ImportError:
    raise SystemExit("openai package not found. Run: pip install openai")

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
# GPT Prompts
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
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

REVIEW_PROMPT = """\
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
    """Append a single record to a JSONL file (creates the file if missing)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_types_from_hint(hint: str) -> set[str]:
    """Return the entity type tokens embedded in a sentence's hint field."""
    words = re.findall(r"\b[A-Z_]{2,}\b", hint)
    return {w for w in words if w in ENTITY_TYPES_SET}


def validate_example(ex: dict) -> bool:
    """Return True only if the example passes basic sanity checks."""
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
    """Split total into equal thirds; any remainder goes to zero_shot."""
    base, rem = divmod(total, 3)
    return {"zero_shot": base + rem, "one_shot": base, "two_shot": base}

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
    """
    Ask GPT-5.4 to produce `n_examples` annotated NER examples that are
    topically relevant to the target sentence.

    Uses a two-turn generate → self-review pattern.
    Returns a list of validated {sentence, entities} dicts (may be empty on failure).
    """
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
            # Turn 1 — generate
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

            # Turn 2 — self-review / correction
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

            # Remove any example whose sentence matches the target
            valid = [
                e for e in valid
                if e["sentence"].strip().lower() != target_sentence.strip().lower()
            ]

            return valid[:n_examples]   # return exactly as many as needed

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
    sentences_path: str,
    output_path:    str,
    total:          int,
    seed:           int,
    batch_size:     int,
    resume:         bool,
    max_retries:    int,
) -> None:
    """
    Full pipeline: load sentences → split → annotate via GPT → save.

    The output file is written incrementally (one append per sentence) so that
    progress is never lost if the script is interrupted.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "  Windows : set OPENAI_API_KEY=sk-...\n"
            "  Linux   : export OPENAI_API_KEY=sk-..."
        )

    client = openai.OpenAI()

    # ── Load sentences ────────────────────────────────────────────────────────
    if not Path(sentences_path).exists():
        raise FileNotFoundError(f"Sentences file not found: {sentences_path}")

    raw = _load_jsonl(sentences_path)
    sentences = [s for s in raw if isinstance(s.get("sentence"), str) and s["sentence"].strip()]
    log.info("Loaded %d valid sentences from '%s'.", len(sentences), sentences_path)

    if not sentences:
        raise ValueError("No valid sentences found in the sentences file.")

    if len(sentences) < total:
        log.warning(
            "File has %d sentences but --total=%d was requested. "
            "Using all %d sentences.", len(sentences), total, len(sentences),
        )
        total = len(sentences)

    # ── Shuffle & split ───────────────────────────────────────────────────────
    rng = random.Random(seed)
    shuffled = list(sentences)
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

    # Interleave modes so the output isn't sorted by mode
    all_items: list[tuple[dict, str]] = []
    for trio in zip(zero_group, one_group, two_group):
        all_items.extend(trio)
    # Append any tail that zip() dropped
    max_len = max(len(zero_group), len(one_group), len(two_group))
    for items in (zero_group, one_group, two_group):
        if len(items) == max_len and len(items) > len(all_items) // 3:
            pass  # already included via zip
    # Simpler: just concatenate and re-shuffle
    all_items = zero_group + one_group + two_group
    rng.shuffle(all_items)

    # ── Resume: find already-processed sentences ──────────────────────────────
    already_done: set[str] = set()
    if resume and Path(output_path).exists():
        existing = _load_jsonl(output_path)
        already_done = {r["target_sentence"] for r in existing}
        log.info(
            "Resume mode: %d / %d records already written.",
            len(already_done), len(all_items),
        )
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

    log.info(
        "Processing %d sentences (%d already done, %d remaining) …",
        total_items, done_count, len(pending_items),
    )

    # Batch the few-shot sentences to reduce API calls
    # (zero-shot items don't need API calls so they are handled inline)
    zero_pending  = [(s, m) for (s, m) in pending_items if m == "zero_shot"]
    few_pending   = [(s, m) for (s, m) in pending_items if m != "zero_shot"]

    # ── Write zero-shot records immediately (no API needed) ───────────────────
    for s, mode in zero_pending:
        record = {
            "target_sentence": s["sentence"],
            "mode":            mode,
            "examples":        [],
        }
        _append_jsonl(record, output_path)
        done_count += 1

    log.info("Zero-shot records written: %d", len(zero_pending))
    log.info("Processing %d few-shot sentences via GPT-5.4 (batch_size=%d) …",
             len(few_pending), batch_size)

    # ── Process few-shot in batches ───────────────────────────────────────────
    # We process them as individual API calls (one target → N examples) rather
    # than truly "batching" them because each target needs its own relevance-
    # guided examples.  We log progress every `batch_size` items.
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

        # If GPT returned fewer examples than needed, pad with empty dicts
        # so the record is always valid (logged as a warning).
        if len(examples) < n_examples:
            log.warning(
                "[%d/%d] Got %d / %d examples for: %.70s …",
                idx, len(few_pending), len(examples), n_examples, s["sentence"],
            )

        record = {
            "target_sentence": s["sentence"],
            "mode":            mode,
            "examples":        examples[:n_examples],
        }
        _append_jsonl(record, output_path)
        done_count += 1

        if idx % batch_size == 0 or idx == len(few_pending):
            log.info(
                "  Progress: %d / %d few-shot done  |  Total written: %d / %d",
                idx, len(few_pending), done_count, total_items,
            )

    log.info("=" * 60)
    log.info("DONE — %d records written to '%s'.", done_count, output_path)
    log.info("=" * 60)

    # ── Final stats ───────────────────────────────────────────────────────────
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
        description=(
            "Generate mixed zero/one/two-shot NER dataset from Generated_Sentences.jsonl "
            "using GPT-5.4 for few-shot example annotation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sentences",
        default="Training Data/Generated_Sentences.jsonl",
        help="Path to Generated_Sentences.jsonl",
    )
    p.add_argument(
        "--output",
        default="Training Data/mixed_ner_dataset.jsonl",
        help="Path for the output dataset JSONL",
    )
    p.add_argument(
        "--total", type=int, default=3000,
        help="Number of records to produce (uses first N sentences after shuffle)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible shuffling and mode assignment",
    )
    p.add_argument(
        "--batch-size", type=int, default=5,
        help="Log progress every N few-shot sentences (does not affect API calls)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Append to an existing output file instead of starting fresh",
    )
    p.add_argument(
        "--max-retries", type=int, default=5,
        help="Max GPT API retry attempts per sentence",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("NER Dataset Generator")
    log.info("  Sentences   : %s", args.sentences)
    log.info("  Output      : %s", args.output)
    log.info("  Total       : %d", args.total)
    log.info("  Seed        : %d", args.seed)
    log.info("  Batch size  : %d (progress logging interval)", args.batch_size)
    log.info("  Resume      : %s", args.resume)
    log.info("  Max retries : %d", args.max_retries)

    build_dataset(
        sentences_path=args.sentences,
        output_path=args.output,
        total=args.total,
        seed=args.seed,
        batch_size=args.batch_size,
        resume=args.resume,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
