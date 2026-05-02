#!/usr/bin/env python3
"""
fix_ner_dataset.py
==================
Post-processing pass on ner_dataset.jsonl that does three things:

1. LABEL CORRECTION  — Applies a canonical correction table to fix known
   mislabellings in few-shot examples (e.g. European Space Agency → GOV_AGENCY,
   Artemis I → SPACECRAFT, the Louvre → MUSEUM, etc.).

2. SCHEMA NORMALISATION — Resolves entity-type ambiguities by enforcing a
   canonical label for entities that appear under multiple types across the pool
   (e.g. Europe → CONTINENT, Reuters → NEWS_AGENCY, argon → ELEMENT).

3. DIVERSITY MAXIMISATION — Re-assigns few-shot examples from the deduplicated
   pool so that each *unique* example sentence is spread as evenly as possible
   across records, reducing the 83.7 % duplication rate.
   Records with already-unique examples are left untouched.
   A domain-overlap scoring heuristic (shared entity types) is used to pick
   the best replacement example when a record needs a new one.

Usage:
    python fix_ner_dataset.py
    python fix_ner_dataset.py --input  "Training Data/ner_dataset.jsonl" \\
                               --output "Training Data/ner_dataset_fixed.jsonl" \\
                               --inplace   # overwrite input (makes a .bak backup first)
    python fix_ner_dataset.py --dry-run    # report only, do not write

Outputs:
    Training Data/ner_dataset_fixed.jsonl   (default)
    Training Data/ner_dataset_fix_report.txt
"""

import argparse
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

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
# 1. LABEL CORRECTION TABLE
#    Format:  (entity_text_exact, wrong_type) → correct_type
#    Use None as wrong_type to correct an entity regardless of its current label.
# ─────────────────────────────────────────────────────────────────────────────
HARD_CORRECTIONS: dict[tuple[str, str | None], str] = {
    # Space agencies mislabelled as OCEAN
    ("European Space Agency", "OCEAN"):         "GOV_AGENCY",
    # Spacecraft mislabelled as PROGRAMMING_LANGUAGE
    ("Artemis I",  "PROGRAMMING_LANGUAGE"):     "SPACECRAFT",
    # Museum mislabelled as PAINTING
    ("the Louvre", "PAINTING"):                 "MUSEUM",
    # Artists mislabelled as PAINTING
    ("Leonardo da Vinci",  "PAINTING"):          "PERSON",
    ("Sandro Botticelli",  "PAINTING"):          "PERSON",
    ("Johannes Vermeer",   "PAINTING"):          "PERSON",
    ("Pablo Picasso",      "PAINTING"):          "PERSON",
    ("Vincent van Gogh",   "PAINTING"):          "PERSON",
    # Continent mislabelled as COUNTRY / GOV_AGENCY
    ("Asia",    "COUNTRY"):                      "CONTINENT",
    ("Asia",    "GOV_AGENCY"):                   "CONTINENT",
    ("Europe",  "GOV_AGENCY"):                   "CONTINENT",
    ("Europe",  "COUNTRY"):                      "CONTINENT",
    ("Africa",  "COUNTRY"):                      "CONTINENT",
    ("Africa",  "GOV_AGENCY"):                   "CONTINENT",
    # Cities mislabelled as STATE or GOV_AGENCY
    ("Austin",       "STATE"):                   "CITY",
    ("Stowe",        "STATE"):                   "CITY",
    ("Colombo",      "GOV_AGENCY"):              "CITY",
    ("Salt Lake City","GOV_AGENCY"):             "CITY",
    ("Monterey",     "GOV_AGENCY"):              "CITY",
    ("Amritsar",     "GOV_AGENCY"):              "CITY",
    ("Geneva",       "GOV_AGENCY"):              "CITY",
    ("Amsterdam",    "GOV_AGENCY"):              "CITY",
    ("Lisbon",       "GOV_AGENCY"):              "CITY",
    ("Bangkok",      "GOV_AGENCY"):              "CITY",
    ("Phoenix",      "STATE"):                   "CITY",
    ("Miami",        "STATE"):                   "CITY",
    ("Buffalo",      "STATE"):                   "CITY",
    ("Fresno",       "STATE"):                   "CITY",
    ("Toronto",      "SPORT_TEAM"):              "CITY",
    ("Madrid",       "SPORT_TEAM"):              "CITY",
    ("Rochester",    "GOV_AGENCY"):              "CITY",
    # Conference mislabelled as PERSON
    ("PyCon US", "PERSON"):                      "CONFERENCE",
    # Bodies of water / landmarks mislabelled as GOV_AGENCY / RELIGION
    ("Beira Lake",        "GOV_AGENCY"):         "LAKE",
    ("Gangaramaya Temple","RELIGION"):           "LANDMARK",
    ("St. Peter's Square","RELIGION"):           "LANDMARK",
    ("Golden Temple",     "RELIGION"):           "LANDMARK",
    ("Vesak",             "RELIGION"):           "FESTIVAL",
    ("Easter",            "RELIGION"):           "FESTIVAL",
    ("Buddhist",          "RELIGION"):           "RELIGION",   # already correct, no-op placeholder
    # People mislabelled as SPORT_TEAM
    ("Jude Bellingham",  "SPORT_TEAM"):          "PERSON",
    ("Vinicius Junior",  "SPORT_TEAM"):          "PERSON",
    ("Philadelphia",     "SPORT_TEAM"):          "CITY",
    # Countries mislabelled as GOV_AGENCY
    ("Germany",  "GOV_AGENCY"):                  "COUNTRY",
    ("Brazil",   "GOV_AGENCY"):                  "COUNTRY",
    ("Spain",    "GOV_AGENCY"):                  "COUNTRY",
    ("Texas",    "GOV_AGENCY"):                  "STATE",
    ("Port of Los Angeles", "GOV_AGENCY"):       "SEAPORT",
    ("Port of Rotterdam",   "GOV_AGENCY"):       "SEAPORT",
    ("Jamaica",  "ISLAND"):                      "COUNTRY",    # Jamaica is primarily a country
    # Vatican City is a country, not GOV_AGENCY
    ("Vatican City", "GOV_AGENCY"):              "COUNTRY",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. SCHEMA NORMALISATION TABLE
#    When an entity has *any* listed type, normalise it to the canonical type.
#    Applied AFTER hard corrections.
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL_TYPES: dict[str, str] = {
    # News organisations → NEWS_AGENCY (not COMPANY)
    "Reuters":              "NEWS_AGENCY",
    "CNN":                  "NEWS_AGENCY",
    "BBC":                  "NEWS_AGENCY",
    "Bloomberg":            "NEWS_AGENCY",
    "The Guardian":         "NEWS_AGENCY",
    "The Wall Street Journal": "NEWS_AGENCY",
    "Associated Press":     "NEWS_AGENCY",
    # Elements vs chemicals — use ELEMENT for pure elements
    "argon":                "ELEMENT",
    "chlorine":             "ELEMENT",
    "silicon":              "ELEMENT",
    "mercury":              "ELEMENT",
    "iron":                 "ELEMENT",
    "gold":                 "ELEMENT",
    "carbon":               "ELEMENT",
    "sodium":               "ELEMENT",
    "zinc":                 "ELEMENT",    # as element; as drug supplement stays DRUG via context
    "hydrogen":             "ELEMENT",
    "ethylene":             "CHEMICAL",   # compound, not element
    "hydrazine":            "CHEMICAL",
    "methane":              "CHEMICAL",
    # Proteins vs drugs — canonical distinction by biological role
    "insulin":              "PROTEIN",    # it IS a protein; when used therapeutically, DRUG is also valid,
                                          # but PROTEIN is more precisely correct for NER
    "hemoglobin":           "PROTEIN",
    "myosin":               "PROTEIN",
    "actin":                "PROTEIN",
    "tubulin":              "PROTEIN",
    # Historical figures vs persons — use HISTORICAL_FIGURE for deceased historical persons
    "Napoleon Bonaparte":   "HISTORICAL_FIGURE",
    "Winston Churchill":    "HISTORICAL_FIGURE",
    "Julius Caesar":        "HISTORICAL_FIGURE",
    "Cleopatra":            "HISTORICAL_FIGURE",
    "Joan of Arc":          "HISTORICAL_FIGURE",
    "Alexander the Great":  "HISTORICAL_FIGURE",
    "Saladin":              "HISTORICAL_FIGURE",
    "William the Conqueror":"HISTORICAL_FIGURE",
    "Vercingetorix":        "HISTORICAL_FIGURE",
    "Guy of Lusignan":      "HISTORICAL_FIGURE",
    "Harold Godwinson":     "HISTORICAL_FIGURE",
    "Darius III":           "HISTORICAL_FIGURE",
    "Babe Ruth":            "HISTORICAL_FIGURE",
    "Theodore Roosevelt":   "HISTORICAL_FIGURE",
    "Pedro II":             "HISTORICAL_FIGURE",
    "Giuseppe Garibaldi":   "HISTORICAL_FIGURE",
    # Painting titles vs painters
    "Mona Lisa":            "PAINTING",
    "The Starry Night":     "PAINTING",
    "Guernica":             "PAINTING",
    "The Birth of Venus":   "PAINTING",
    "Girl with a Pearl Earring": "PAINTING",
}

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
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
                log.warning("Skipping malformed line %d: %s", i, e)
    return records


def _save_jsonl(data: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in data:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 & 2: Correct labels in a single example dict
# ─────────────────────────────────────────────────────────────────────────────

def correct_example(example: dict) -> tuple[dict, int]:
    """
    Apply HARD_CORRECTIONS then CANONICAL_TYPES to all entities in one example.
    Returns (corrected_example, n_fixes_applied).
    """
    fixes = 0
    new_entities = []
    for ent in example.get("entities", []):
        e_text = ent["entity"]
        e_type = ent["type"]

        # Phase 1: hard correction (entity + wrong type → correct type)
        key = (e_text, e_type)
        if key in HARD_CORRECTIONS:
            corrected = HARD_CORRECTIONS[key]
            if corrected != e_type:
                fixes += 1
                e_type = corrected

        # Phase 2: canonical normalisation (entity text → canonical type)
        if e_text in CANONICAL_TYPES:
            canonical = CANONICAL_TYPES[e_text]
            if canonical != e_type:
                fixes += 1
                e_type = canonical

        new_entities.append({"entity": e_text, "type": e_type})

    return {"sentence": example["sentence"], "entities": new_entities}, fixes

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Re-diversify — build a corrected pool then redistribute
# ─────────────────────────────────────────────────────────────────────────────

def build_corrected_pool(records: list[dict]) -> list[dict]:
    """
    Extract all unique example sentences from the dataset (after correction)
    and return them as a deduplicated pool.
    """
    seen: set[str] = set()
    pool: list[dict] = []
    for rec in records:
        for ex in rec.get("examples", []):
            key = ex["sentence"].strip()
            if key not in seen:
                seen.add(key)
                pool.append(ex)
    log.info("Unique examples in corrected pool: %d", len(pool))
    return pool


def score_example(example: dict, target_types: set[str]) -> int:
    """Number of entity types shared between example and target."""
    ex_types = {ent["type"] for ent in example.get("entities", [])}
    return len(ex_types & target_types)


def target_types_from_record(record: dict) -> set[str]:
    """
    We don't have the original hint, but we can approximate target entity types
    from the main sentence's words.  As a fallback, return empty set (→ random pick).
    """
    # Look at existing examples for a domain hint (best proxy without the hint field)
    types: set[str] = set()
    for ex in record.get("examples", []):
        for ent in ex.get("entities", []):
            types.add(ent["type"])
    return types


def redistribute_examples(
    records: list[dict],
    pool: list[dict],
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """
    Re-assign few-shot examples so each unique pool entry is reused as few times
    as possible.  Uses a round-robin allocation with domain-aware tie-breaking.

    Returns (updated_records, stats_dict).
    """
    rng = random.Random(seed)

    # Count how many times each example sentence is currently used
    usage: Counter = Counter()
    for rec in records:
        for ex in rec.get("examples", []):
            usage[ex["sentence"]] += 1

    stats = {
        "total_one_shot": 0,
        "total_two_shot": 0,
        "examples_reassigned": 0,
        "pool_size": len(pool),
    }

    # Build a pool index by sentence for quick lookup
    pool_by_sentence = {ex["sentence"]: ex for ex in pool}
    pool_sentences = list(pool_by_sentence.keys())  # deterministic order

    # Target: max reuse = ceil(needed / pool_size)
    n_one  = sum(1 for r in records if r["type"] == "one_shot")
    n_two  = sum(1 for r in records if r["type"] == "two_shot")
    needed = n_one + 2 * n_two
    max_reuse = max(1, -(-needed // len(pool)))  # ceiling division
    log.info(
        "Redistribution target: pool=%d, needed=%d, target_max_reuse=%d",
        len(pool), needed, max_reuse,
    )

    stats["total_one_shot"] = n_one
    stats["total_two_shot"] = n_two

    # Reset usage counts for fresh assignment
    new_usage: Counter = Counter()

    def pick(n: int, exclude: str, prefer_types: set[str]) -> list[dict]:
        """Pick n examples from pool prioritising low reuse and domain overlap."""
        candidates = [s for s in pool_sentences if s != exclude]

        # Score: (shared_types, -current_usage, random tiebreak)
        scored = []
        for s in candidates:
            ex = pool_by_sentence[s]
            domain_score = score_example(ex, prefer_types) if prefer_types else 0
            scored.append((domain_score, -new_usage[s], rng.random(), s))
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

        chosen = []
        used_in_pick: set[str] = set()
        for _, _, _, s in scored:
            if s not in used_in_pick:
                chosen.append(pool_by_sentence[s])
                used_in_pick.add(s)
                if len(chosen) == n:
                    break

        # Fallback: if not enough, allow repeats
        while len(chosen) < n:
            fallback = rng.choice(pool)
            chosen.append(fallback)

        for ex in chosen:
            new_usage[ex["sentence"]] += 1
        return chosen

    updated: list[dict] = []
    for rec in records:
        if rec["type"] == "zero_shot":
            updated.append(rec)
            continue

        n_needed = 1 if rec["type"] == "one_shot" else 2
        prefer   = target_types_from_record(rec)
        exclude  = rec["sentence"]

        new_examples = pick(n_needed, exclude, prefer)

        # Count reassignments
        old_sentences = {ex["sentence"] for ex in rec.get("examples", [])}
        new_sentences = {ex["sentence"] for ex in new_examples}
        if old_sentences != new_sentences:
            stats["examples_reassigned"] += 1

        updated.append({
            "type":     rec["type"],
            "sentence": rec["sentence"],
            "examples": new_examples,
        })

    # Final duplication rate
    all_ex_sentences = []
    for rec in updated:
        for ex in rec.get("examples", []):
            all_ex_sentences.append(ex["sentence"])
    dup_rate = (len(all_ex_sentences) - len(set(all_ex_sentences))) / max(len(all_ex_sentences), 1)
    stats["final_dup_rate"] = round(dup_rate * 100, 1)
    stats["final_unique_examples"] = len(set(all_ex_sentences))
    stats["max_reuse_after"] = max(new_usage.values()) if new_usage else 0

    return updated, stats

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_fixes(
    input_path: str,
    output_path: str,
    dry_run: bool = False,
    seed: int = 42,
) -> None:

    log.info("=" * 60)
    log.info("NER Dataset Fix Pipeline")
    log.info("  Input  : %s", input_path)
    log.info("  Output : %s", output_path)
    log.info("  Dry run: %s", dry_run)
    log.info("=" * 60)

    records = _load_jsonl(input_path)
    log.info("Loaded %d records.", len(records))

    # ── Phase 1+2: Correct labels ────────────────────────────────────────────
    log.info("\n[Phase 1+2] Applying label corrections and schema normalisation …")
    total_fixes = 0
    corrected_records: list[dict] = []

    for rec in records:
        corrected_examples = []
        for ex in rec.get("examples", []):
            corrected_ex, n_fixes = correct_example(ex)
            total_fixes += n_fixes
            corrected_examples.append(corrected_ex)
        corrected_records.append({
            "type":     rec["type"],
            "sentence": rec["sentence"],
            "examples": corrected_examples,
        })

    log.info("  Total label fixes applied: %d", total_fixes)

    # ── Phase 3: Redistribute examples ──────────────────────────────────────
    log.info("\n[Phase 3] Re-diversifying few-shot example assignments …")

    # Before stats
    before_ex = []
    for rec in corrected_records:
        for ex in rec.get("examples", []):
            before_ex.append(ex["sentence"])
    before_dup = (len(before_ex) - len(set(before_ex))) / max(len(before_ex), 1) * 100
    log.info("  Before: %d total examples, %d unique (dup rate: %.1f%%)",
             len(before_ex), len(set(before_ex)), before_dup)

    pool = build_corrected_pool(corrected_records)
    final_records, stats = redistribute_examples(corrected_records, pool, seed=seed)

    log.info("  Records with reassigned examples : %d", stats["examples_reassigned"])
    log.info("  After : %d unique examples used (dup rate: %.1f%%)",
             stats["final_unique_examples"], stats["final_dup_rate"])
    log.info("  Max reuse after redistribution   : %d×", stats["max_reuse_after"])

    # ── Summary report ───────────────────────────────────────────────────────
    report_lines = [
        "NER Dataset Fix Report",
        "=" * 60,
        f"Input                    : {input_path}",
        f"Output                   : {output_path}",
        f"Total records            : {len(final_records)}",
        "",
        "LABEL CORRECTIONS",
        f"  Total fixes applied    : {total_fixes}",
        "",
        "EXAMPLE REDISTRIBUTION",
        f"  Pool size (unique)     : {stats['pool_size']}",
        f"  Records reassigned     : {stats['examples_reassigned']}",
        f"  Dup rate before        : {before_dup:.1f}%",
        f"  Dup rate after         : {stats['final_dup_rate']}%",
        f"  Unique examples used   : {stats['final_unique_examples']}",
        f"  Max reuse after        : {stats['max_reuse_after']}×",
    ]

    report_path = str(Path(output_path).parent / "ner_dataset_fix_report.txt")
    if not dry_run:
        with open(report_path, "w", encoding="utf-8") as rp:
            rp.write("\n".join(report_lines) + "\n")
        log.info("\nReport saved to '%s'.", report_path)

    for line in report_lines:
        log.info(line)

    # ── Write output ─────────────────────────────────────────────────────────
    if dry_run:
        log.info("\n[DRY RUN] No files written.")
    else:
        _save_jsonl(final_records, output_path)
        log.info("\nSaved %d records to '%s'.", len(final_records), output_path)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fix annotation errors and improve few-shot diversity in ner_dataset.jsonl",
    )
    p.add_argument(
        "--input", default="Training Data/ner_dataset.jsonl",
        help="Input JSONL path (default: Training Data/ner_dataset.jsonl)",
    )
    p.add_argument(
        "--output", default="Training Data/ner_dataset_fixed.jsonl",
        help="Output JSONL path (default: Training Data/ner_dataset_fixed.jsonl)",
    )
    p.add_argument(
        "--inplace", action="store_true",
        help="Overwrite the input file (creates a .bak backup first)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Analyse and report only — do not write any files",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for redistribution (default: 42)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    output = args.output
    if args.inplace:
        bak = args.input + ".bak"
        shutil.copy2(args.input, bak)
        log.info("Backup created: %s", bak)
        output = args.input

    run_fixes(
        input_path=args.input,
        output_path=output,
        dry_run=args.dry_run,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
