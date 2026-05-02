#!/usr/bin/env python3
"""
assemble_dataset.py
===================
Step 2 of 2 — Combine pre-generated sentences with the GPT-5.4 few-shot pool
to produce the final NER dataset JSONL.

Inputs:
    1.  Sentences file  : Training Data/Generated_Sentences.jsonl
                          Each line: {"sentence": "...", "hint": "..."}

    2.  Few-shot pool   : Training Data/gpt_example_pool.jsonl
                          Each line: {"sentence": "...", "entities": [...]}

Output:
    Training Data/ner_dataset.jsonl
    Each line: {
        "type": "zero_shot" | "one_shot" | "two_shot",
        "sentence": "...",
        "examples": [...]      # empty list for zero_shot
    }

Distribution (for 3000 sentences):
    zero_shot  → 1000 sentences  (no examples)
    one_shot   → 1000 sentences  (1 GPT example attached)
    two_shot   → 1000 sentences  (2 GPT examples attached)

Example selection strategy:
    For each target sentence, GPT examples are ranked by how many entity
    types they share with the sentence's "hint" field. Top-scoring examples
    are selected, so the few-shot demonstrations are always topically relevant.

Usage:
    python assemble_dataset.py
    python assemble_dataset.py --sentences "Training Data/Generated_Sentences.jsonl" \\
                               --pool     "Training Data/gpt_example_pool.jsonl" \\
                               --output   "Training Data/ner_dataset.jsonl" \\
                               --total 3000 --seed 42
"""

import argparse
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Optional

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
# Entity Type Schema  (must match generate_fewshot_pool.py)
# ─────────────────────────────────────────────────────────────────────────────
ENTITY_TYPES = set([
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
])

# ─────────────────────────────────────────────────────────────────────────────
# I/O Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("Skipping malformed line %d in '%s': %s", i, path, e)
    return records


def _save_jsonl(data: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in data:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# Example Pool  (relevance-ranked sampling)
# ─────────────────────────────────────────────────────────────────────────────

class ExamplePool:
    """
    Wraps the GPT-generated examples and provides relevance-ranked sampling.

    For each target sentence, examples are scored by how many entity types
    they share with the sentence's hint.  Top-ranked examples are selected,
    breaking ties randomly to prevent repetition.
    """

    def __init__(self, examples: list[dict], seed: int):
        # Only keep examples that have at least one entity
        self._pool = [e for e in examples if e.get("entities")]
        self._all  = examples          # fallback if pool is thin
        self._rng  = random.Random(seed)

        log.info(
            "ExamplePool ready: %d examples total (%d entity-bearing).",
            len(self._all), len(self._pool),
        )

    def sample(self, n: int, exclude_sentence: str, target_types: set[str]) -> list[dict]:
        """
        Draw `n` examples from the pool.

        Args:
            n:                 Number of examples to return.
            exclude_sentence:  The target sentence — never returned as an example.
            target_types:      Set of entity type strings from the sentence's hint.

        Returns:
            List of {"sentence": ..., "entities": [...]} dicts.
        """
        # Prefer entity-bearing examples; fall back to all if pool is thin
        candidates = [e for e in self._pool if e["sentence"] != exclude_sentence]
        if len(candidates) < n:
            candidates = [e for e in self._all if e["sentence"] != exclude_sentence]
        if len(candidates) < n:
            log.warning("Pool has fewer than %d candidates — returning %d.", n, len(candidates))
            return candidates

        if target_types:
            # Score: number of shared entity types
            def score(ex: dict) -> int:
                ex_types = {ent["type"] for ent in ex.get("entities", [])}
                return len(ex_types & target_types)

            scored = [(score(c), self._rng.random(), c) for c in candidates]
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            chosen = [c for _, _, c in scored[:n]]
        else:
            chosen = self._rng.sample(candidates, n)

        return [{"sentence": c["sentence"], "entities": c["entities"]} for c in chosen]

# ─────────────────────────────────────────────────────────────────────────────
# Hint Parsing
# ─────────────────────────────────────────────────────────────────────────────

def extract_types_from_hint(hint: str) -> set[str]:
    """
    Parse the entity types encoded in a sentence's hint string.

    Example hint:
        "Write about real-world examples of: COMPANY, DATE, MONEY."
    Returns:
        {"COMPANY", "DATE", "MONEY"}
    """
    words = re.findall(r"\b[A-Z_]{2,}\b", hint)
    return {w for w in words if w in ENTITY_TYPES}

# ─────────────────────────────────────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────────────────────────────────────

def compute_split(total: int) -> dict[str, int]:
    """
    Split total into equal thirds. Any remainder goes to zero_shot.

    For 3000: zero=1000, one=1000, two=1000
    For 3001: zero=1001, one=1000, two=1000
    """
    base, rem = divmod(total, 3)
    return {"zero_shot": base + rem, "one_shot": base, "two_shot": base}


def assemble(
    sentences:  list[dict],
    pool:       ExamplePool,
    total:      int,
    seed:       int,
) -> list[dict]:
    """
    Distribute sentences across zero/one/two-shot modes and attach few-shot examples.

    Args:
        sentences:  List of {"sentence": ..., "hint": ...} dicts.
        pool:       ExamplePool to draw few-shot examples from.
        total:      How many records to produce.
        seed:       Random seed for reproducibility.

    Returns:
        List of assembled dataset records.
    """
    rng = random.Random(seed)

    # Shuffle sentences for unbiased split
    shuffled = list(sentences)
    rng.shuffle(shuffled)
    shuffled = shuffled[:total]

    split = compute_split(len(shuffled))
    log.info(
        "Mode split — zero_shot: %d   one_shot: %d   two_shot: %d",
        split["zero_shot"], split["one_shot"], split["two_shot"],
    )

    z_end = split["zero_shot"]
    o_end = z_end + split["one_shot"]

    zero_group = shuffled[:z_end]
    one_group  = shuffled[z_end:o_end]
    two_group  = shuffled[o_end:]

    records: list[dict] = []

    # ── Zero-shot ────────────────────────────────────────────────────────────
    for s in zero_group:
        records.append({
            "type":     "zero_shot",
            "sentence": s["sentence"],
            "examples": [],
        })

    # ── One-shot ─────────────────────────────────────────────────────────────
    for s in one_group:
        types = extract_types_from_hint(s.get("hint", ""))
        examples = pool.sample(1, exclude_sentence=s["sentence"], target_types=types)
        records.append({
            "type":     "one_shot",
            "sentence": s["sentence"],
            "examples": examples,
        })

    # ── Two-shot ─────────────────────────────────────────────────────────────
    for s in two_group:
        types = extract_types_from_hint(s.get("hint", ""))
        examples = pool.sample(2, exclude_sentence=s["sentence"], target_types=types)
        records.append({
            "type":     "two_shot",
            "sentence": s["sentence"],
            "examples": examples,
        })

    # Interleave modes in the final output
    rng.shuffle(records)
    return records

# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(records: list[dict]) -> None:
    mode_counts = Counter(r["type"] for r in records)
    one_two = [r for r in records if r["type"] in ("one_shot", "two_shot")]
    shots_per_type: Counter = Counter()
    for r in one_two:
        for ex in r["examples"]:
            for ent in ex.get("entities", []):
                shots_per_type[ent.get("type", "")] += 1

    log.info("=" * 60)
    log.info("DATASET STATISTICS")
    log.info("=" * 60)
    log.info("Total records : %d", len(records))
    for mode in ("zero_shot", "one_shot", "two_shot"):
        log.info("  %-12s : %d", mode, mode_counts.get(mode, 0))
    log.info("Top entity types in few-shot examples:")
    for etype, count in shots_per_type.most_common(15):
        log.info("  %-25s : %d", etype, count)
    log.info("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Assemble NER dataset from sentences + GPT few-shot pool (Step 2 of 2)",
    )
    p.add_argument(
        "--sentences",
        default="Training Data/Generated_Sentences.jsonl",
        help="Path to pre-generated sentences JSONL (default: Training Data/Generated_Sentences.jsonl)",
    )
    p.add_argument(
        "--pool",
        default="Training Data/gpt_example_pool.jsonl",
        help="Path to GPT few-shot pool JSONL (default: Training Data/gpt_example_pool.jsonl)",
    )
    p.add_argument(
        "--output",
        default="Training Data/ner_dataset.jsonl",
        help="Output JSONL path (default: Training Data/ner_dataset.jsonl)",
    )
    p.add_argument("--total", type=int, default=3000,
                   help="Number of records to produce (default: 3000)")
    p.add_argument("--seed",  type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()

    log.info("NER Dataset Assembler")
    log.info("  Sentences : %s", args.sentences)
    log.info("  Pool      : %s", args.pool)
    log.info("  Output    : %s", args.output)
    log.info("  Total     : %d", args.total)
    log.info("  Seed      : %d", args.seed)

    # ── Load sentences ────────────────────────────────────────────────────────
    if not Path(args.sentences).exists():
        raise FileNotFoundError(f"Sentences file not found: {args.sentences}")
    sentences = _load_jsonl(args.sentences)
    sentences = [s for s in sentences if isinstance(s.get("sentence"), str) and s["sentence"].strip()]
    log.info("Loaded %d valid sentences.", len(sentences))

    if len(sentences) == 0:
        raise ValueError("No valid sentences found. Check the sentences file.")

    if len(sentences) < args.total:
        log.warning(
            "Sentences file has %d entries but --total=%d requested. "
            "Using all %d sentences.",
            len(sentences), args.total, len(sentences),
        )
        args.total = len(sentences)

    # ── Load GPT few-shot pool ────────────────────────────────────────────────
    if not Path(args.pool).exists():
        raise FileNotFoundError(
            f"Few-shot pool not found: {args.pool}\n"
            "Run generate_fewshot_pool.py first."
        )
    raw_pool = _load_jsonl(args.pool)
    log.info("Loaded %d examples from GPT pool.", len(raw_pool))

    if len(raw_pool) < 2:
        raise ValueError("Pool has fewer than 2 examples — cannot assemble one/two-shot records.")

    # ── Assemble ──────────────────────────────────────────────────────────────
    pool = ExamplePool(raw_pool, seed=args.seed + 1)
    records = assemble(sentences, pool, total=args.total, seed=args.seed)

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_jsonl(records, args.output)
    log.info("Saved %d records to '%s'.", len(records), args.output)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print_stats(records)


if __name__ == "__main__":
    main()
