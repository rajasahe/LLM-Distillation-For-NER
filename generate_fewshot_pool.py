#!/usr/bin/env python3
"""
generate_fewshot_pool.py
========================
Step 1 of 2 — Generate a pool of pristine few-shot NER examples using GPT-5.4.

Saves results to:  Training Data/gpt_example_pool.jsonl

Each record:
    {
        "sentence": "Apple acquired Beats in 2014 for $3 billion.",
        "entities": [
            {"entity": "Apple",  "type": "COMPANY"},
            {"entity": "Beats",  "type": "COMPANY"},
            {"entity": "2014",   "type": "DATE"},
            {"entity": "$3 billion", "type": "MONEY"}
        ]
    }

Usage:
    python generate_fewshot_pool.py
    python generate_fewshot_pool.py --num-examples 500 --output "Training Data/gpt_example_pool.jsonl"
    python generate_fewshot_pool.py --resume    # append to existing pool
"""

import argparse
import json
import logging
import os
import random
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
# Entity Type Schema  (must match generate_ner_dataset.py and assemble_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────
ENTITY_TYPES = [
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

# ─────────────────────────────────────────────────────────────────────────────
# JSON schema for structured output
# ─────────────────────────────────────────────────────────────────────────────
RESPONSE_SCHEMA = {
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

SYSTEM_PROMPT = """\
You are an expert NLP annotator. Generate highly diverse, complex English sentences \
and exhaustively tag ALL named entities present.

Rules:
- Entity types MUST strictly use the provided enum — do NOT invent new types.
- Every entity span must appear verbatim in the sentence.
- Each sentence should mix multiple entity domains for richness.

CANONICAL LABEL RULES (these override any other interpretation):
- Continents (Europe, Asia, Africa, North America, South America, Australia, Antarctica)
    → always CONTINENT, never COUNTRY or GOV_AGENCY
- News / media organisations (Reuters, CNN, BBC, Bloomberg, The Guardian, AP, Al Jazeera)
    → always NEWS_AGENCY, never COMPANY
- Pure chemical elements (iron, gold, carbon, silicon, argon, chlorine, mercury, hydrogen, zinc…)
    → always ELEMENT, never CHEMICAL
- Chemical compounds (methane, ethylene, hydrazine, salicylic acid…)
    → always CHEMICAL, never ELEMENT
- Biological proteins (insulin, hemoglobin, actin, myosin, tubulin, albumin…)
    → always PROTEIN, even when used therapeutically
- Museums (The Louvre, The Met, Smithsonian, British Museum…)
    → always MUSEUM, never PAINTING or LANDMARK
- Painters / artists as people (Leonardo da Vinci, Picasso, Van Gogh, Vermeer…)
    → always PERSON, never PAINTING
- Artworks / paintings (Mona Lisa, The Starry Night, Guernica…)
    → always PAINTING
- Historical deceased figures (Napoleon, Churchill, Caesar, Cleopatra…)
    → always HISTORICAL_FIGURE, never PERSON
- Cities (Austin, Geneva, Amsterdam, Colombo, Salt Lake City, Monterey…)
    → always CITY, never STATE or GOV_AGENCY
- Countries (Germany, Brazil, Spain, Jamaica…)
    → always COUNTRY, never GOV_AGENCY
- Spacecraft (Artemis I, Apollo 11, Sputnik, Voyager…)
    → always SPACECRAFT, never PROGRAMMING_LANGUAGE
- Conferences / events (PyCon, GDC, Devcon, NeurIPS…)
    → always CONFERENCE, never PERSON or COMPANY
- Sports players → always PERSON, never SPORT_TEAM
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_dynamic_hint() -> str:
    """Pick a random set of entity types to guide diversity."""
    num = random.choices([1, 2, 3, 4], weights=[0.2, 0.35, 0.3, 0.15])[0]
    chosen = random.sample(ENTITY_TYPES, num)
    return (
        f"Generate sentences containing real-world examples of: {', '.join(chosen)}. "
        "Do NOT write the uppercase category names in the sentences."
    )


def validate(example: dict) -> bool:
    """Basic sanity check: sentence non-empty, all entity spans present verbatim."""
    sentence = example.get("sentence", "").strip()
    if len(sentence) < 5:
        return False
    for ent in example.get("entities", []):
        span = ent.get("entity", "")
        etype = ent.get("type", "")
        if not span or span.lower() not in sentence.lower():
            return False
        if etype not in ENTITY_TYPES:
            return False
    return True


def _load_jsonl(path: str) -> list[dict]:
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


def _save_jsonl(data: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in data:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# Core Generation
# ─────────────────────────────────────────────────────────────────────────────

REVIEW_PROMPT = """\
Review the annotations you just produced. For EVERY entity check:
1. Is the entity span present verbatim in the sentence? Remove it if not.
2. Does the type follow the CANONICAL LABEL RULES in the system prompt?
   Pay special attention to: CONTINENT vs COUNTRY, NEWS_AGENCY vs COMPANY,
   ELEMENT vs CHEMICAL, PROTEIN vs DRUG, MUSEUM vs PAINTING,
   PERSON (artist) vs PAINTING (artwork), HISTORICAL_FIGURE vs PERSON,
   CITY vs STATE vs GOV_AGENCY, SPACECRAFT vs PROGRAMMING_LANGUAGE,
   CONFERENCE vs PERSON, SPORT player → PERSON not SPORT_TEAM.
Return the corrected JSON only — same schema, no extra commentary.
"""


def generate_pool(
    num_examples: int,
    output_path: str,
    batch_size: int = 10,
    resume: bool = False,
    max_retries: int = 5,
    max_per_type: int = 0,
) -> list[dict]:
    """
    Generate `num_examples` annotated examples using GPT-5.4 and save to output_path.

    Args:
        num_examples:  Total examples to generate.
        output_path:   Path to save the JSONL pool file.
        batch_size:    How many examples to request per API call.
        resume:        If True, load existing pool and continue from where it left off.
        max_retries:   Retry limit per failed API call.
        max_per_type:  If > 0, cap how many examples can share the same dominant entity
                       type. Enforces broader coverage across types.

    Returns:
        List of validated example dicts.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    client = openai.OpenAI()
    pool: list[dict] = []

    # Resume: load existing pool
    if resume and Path(output_path).exists():
        pool = _load_jsonl(output_path)
        log.info("Resumed: loaded %d existing examples from '%s'.", len(pool), output_path)
    elif Path(output_path).exists():
        log.info(
            "Output file '%s' already exists with %d examples. "
            "Use --resume to append, or delete it to regenerate.",
            output_path, len(_load_jsonl(output_path)),
        )
        return _load_jsonl(output_path)

    needed = num_examples - len(pool)
    if needed <= 0:
        log.info("Pool already has %d / %d examples. Nothing to generate.", len(pool), num_examples)
        return pool

    # Track per-type counts for diversity cap
    type_counts: dict[str, int] = {}
    if max_per_type > 0:
        for ex in pool:
            for ent in ex.get("entities", []):
                t = ent["type"]
                type_counts[t] = type_counts.get(t, 0) + 1
        log.info("Diversity cap: max %d examples per dominant entity type.", max_per_type)

    log.info("Generating %d examples using GPT-5.4 (batch_size=%d) …", needed, batch_size)

    consecutive_failures = 0

    while len(pool) < num_examples:
        req_size = min(batch_size, num_examples - len(pool))
        hint = get_dynamic_hint()
        user_prompt = f"Generate exactly {req_size} diverse sentences. Topic hint: {hint}"

        retries = 0
        success = False
        while retries < max_retries:
            try:
                # ── Turn 1: generate ─────────────────────────────────────────
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ]
                response = client.chat.completions.create(
                    model="gpt-5.4",
                    messages=messages,
                    response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
                    temperature=0.7,
                )
                draft_content = response.choices[0].message.content

                # ── Turn 2: self-review ───────────────────────────────────────
                messages.append({"role": "assistant", "content": draft_content})
                messages.append({"role": "user",      "content": REVIEW_PROMPT})
                review_response = client.chat.completions.create(
                    model="gpt-5.4",
                    messages=messages,
                    response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
                    temperature=0.0,   # deterministic correction pass
                )
                data = json.loads(review_response.choices[0].message.content)
                examples = data.get("examples", [])

                valid = [e for e in examples if validate(e)]

                # ── Diversity cap ─────────────────────────────────────────────
                if max_per_type > 0:
                    capped = []
                    for ex in valid:
                        # Dominant type = most frequent type in this example
                        ex_type_counts = {}
                        for ent in ex.get("entities", []):
                            t = ent["type"]
                            ex_type_counts[t] = ex_type_counts.get(t, 0) + 1
                        dominant = max(ex_type_counts, key=ex_type_counts.get) if ex_type_counts else None
                        if dominant and type_counts.get(dominant, 0) >= max_per_type:
                            continue   # skip — this type is already saturated
                        capped.append(ex)
                        if dominant:
                            type_counts[dominant] = type_counts.get(dominant, 0) + 1
                    skipped = len(valid) - len(capped)
                    if skipped:
                        log.info("  Diversity cap: skipped %d over-represented examples.", skipped)
                    valid = capped

                pool.extend(valid)

                log.info(
                    "  Batch done: got %d valid / %d returned  |  Pool: %d / %d",
                    len(valid), len(examples), len(pool), num_examples,
                )
                success = True
                consecutive_failures = 0
                break

            except Exception as exc:
                retries += 1
                wait = 2 ** retries          # exponential back-off: 2, 4, 8, 16 …
                log.warning("API error (attempt %d/%d): %s — retrying in %ds …",
                            retries, max_retries, exc, wait)
                time.sleep(wait)

        if not success:
            consecutive_failures += 1
            log.error("Batch failed after %d retries. Consecutive failures: %d",
                      max_retries, consecutive_failures)
            if consecutive_failures >= 3:
                log.error("3 consecutive batch failures — saving progress and stopping.")
                break

        # Save after every successful batch
        _save_jsonl(pool, output_path)

    final = pool[:num_examples]
    _save_jsonl(final, output_path)
    log.info("=" * 60)
    log.info("DONE — Saved %d examples to '%s'.", len(final), output_path)
    log.info("=" * 60)
    return final


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate GPT-5.4 few-shot NER example pool (Step 1 of 2)",
    )
    p.add_argument("--num-examples", type=int, default=1500,
                   help="Number of annotated examples to generate (default: 1500)")
    p.add_argument("--output", default="Training Data/gpt_example_pool.jsonl",
                   help="Output JSONL path (default: Training Data/gpt_example_pool.jsonl)")
    p.add_argument("--batch-size", type=int, default=10,
                   help="Examples per API call (default: 10)")
    p.add_argument("--resume", action="store_true",
                   help="Append to existing pool instead of starting fresh")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for hint diversity (default: 42)")
    p.add_argument("--max-per-type", type=int, default=80,
                   help="Max examples per dominant entity type (0 = no cap, default: 80)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    log.info("GPT-5.4 Few-Shot Pool Generator")
    log.info("  Target examples : %d", args.num_examples)
    log.info("  Output path     : %s", args.output)
    log.info("  Batch size      : %d", args.batch_size)
    log.info("  Resume          : %s", args.resume)
    log.info("  Max per type    : %s", args.max_per_type or "unlimited")

    pool = generate_pool(
        num_examples=args.num_examples,
        output_path=args.output,
        batch_size=args.batch_size,
        resume=args.resume,
        max_per_type=args.max_per_type,
    )

    # Quick stats
    entity_counts: dict[str, int] = {}
    for ex in pool:
        for ent in ex.get("entities", []):
            t = ent.get("type", "UNKNOWN")
            entity_counts[t] = entity_counts.get(t, 0) + 1

    log.info("Entity type distribution across pool:")
    for etype, count in sorted(entity_counts.items(), key=lambda x: -x[1])[:20]:
        log.info("  %-25s : %d", etype, count)
    log.info("  … (%d unique types total)", len(entity_counts))


if __name__ == "__main__":
    main()
