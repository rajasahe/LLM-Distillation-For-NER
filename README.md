
# 🧠 LLM Distillation for Named Entity Recognition

> **Behavioural Knowledge Distillation** — Compressing a 12 billion parameter teacher model (Gemma 3 12B) into a 270 million parameter student model (Gemma 3 270M) for fine-grained Named Entity Recognition, with no performance cliff.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Entity Type Schema](#entity-type-schema)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Stage 1 — Synthetic Dataset Generation](#stage-1--synthetic-dataset-generation)
  - [Stage 2 — Dataset Cleaning & Normalisation](#stage-2--dataset-cleaning--normalisation)
  - [Stage 3 — Distillation Training](#stage-3--distillation-training)
  - [Stage 4 — Test Data Generation](#stage-4--test-data-generation)
  - [Stage 5 — Evaluation](#stage-5--evaluation)
- [Prompt Format](#prompt-format)
- [Training Configuration](#training-configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Setup & Usage](#setup--usage)
- [File Reference](#file-reference)
- [Requirements](#requirements)

---

## Overview

This repository implements an end-to-end **behavioural fractional distillation** pipeline for NER. The core idea is to teach a tiny 270M-parameter student model to reproduce the entity-extraction behaviour of a 12B-parameter teacher, using a rich synthetic dataset and a combined KL Divergence + Cross-Entropy training objective.

**Key highlights:**

| Property | Value |
|---|---|
| Teacher Model | `google/gemma-3-12b-it` |
| Student Model | `google/gemma-3-270m` |
| Fine-tuning strategy | Full parameter update (no LoRA / PEFT) |
| Training samples | 3,000 JSONL records |
| Entity type schema | 100+ fine-grained types |
| Prompt modes | Zero-shot, One-shot, Two-shot |
| Loss function | 0.7 × KL Divergence + 0.3 × Cross-Entropy |
| Distillation temperature | 2.0 (Hinton softened distributions) |
| Ground-truth annotator | Claude + ChatGPT (human-quality gold labels) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DISTILLATION PIPELINE                        │
│                                                                 │
│  ┌─────────────────┐     ┌─────────────────────────────────┐   │
│  │  Gemma 3 12B    │────▶│   Synthetic Sentence Generation  │   │
│  │  (Generator)    │     │   (training_data_generation.py)  │   │
│  └─────────────────┘     └──────────────┬──────────────────┘   │
│                                         │                       │
│  ┌─────────────────┐                    ▼                       │
│  │   GPT-5.4       │────▶  Few-Shot Example Annotation          │
│  │  (Annotator)    │       (2-turn generate → self-review)      │
│  └─────────────────┘              │                             │
│                                   ▼                             │
│                        mixed_ner_dataset_final.jsonl            │
│                        (3,000 records, ~1.6 MB)                 │
│                                   │                             │
│                                   ▼                             │
│  ┌─────────────────┐     ┌────────────────────┐                 │
│  │  Teacher 12B    │────▶│  Distillation Loop  │                │
│  │  (generates     │     │  KL + CE Loss        │                │
│  │   logits+text)  │     │  NER_Distillation_v2 │                │
│  └─────────────────┘     └────────┬───────────┘                 │
│                                   │                             │
│                                   ▼                             │
│                        ner_distilled_model/                     │
│                        (Gemma 3 270M, fine-tuned)               │
│                                   │                             │
│                                   ▼                             │
│         ┌─────────────────────────────────────────┐            │
│         │         EVALUATION (100 samples)         │            │
│         │  Teacher  |  Base Student  |  Distilled  │            │
│         │  Precision / Recall / F1 per shot mode   │            │
│         └─────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Entity Type Schema

The model is trained on a **100+ class fine-grained entity taxonomy** spanning multiple domains:

| Domain | Entity Types |
|---|---|
| **People** | `PERSON`, `FICTIONAL_CHARACTER`, `DEITY`, `HISTORICAL_FIGURE` |
| **Organisations** | `COMPANY`, `STARTUP`, `GOV_AGENCY`, `MILITARY_UNIT`, `SPORT_TEAM`, `POLITICAL_PARTY`, `NEWS_AGENCY`, `BAND`, `NGO`, `UNIVERSITY`, `RESEARCH_INSTITUTE` |
| **Geography** | `COUNTRY`, `CITY`, `STATE`, `COUNTY`, `CONTINENT`, `PLANET`, `STAR`, `GALAXY`, `RIVER`, `MOUNTAIN`, `LAKE`, `OCEAN`, `ISLAND`, `DESERT` |
| **Infrastructure** | `HOSPITAL`, `CLINIC`, `AIRPORT`, `TRAIN_STATION`, `SEAPORT`, `MUSEUM`, `STADIUM`, `THEATER`, `BRIDGE`, `HIGHWAY`, `MONUMENT`, `POWER_PLANT` |
| **Medicine & Biology** | `DISEASE`, `SYMPTOM`, `DRUG`, `VACCINE`, `MEDICAL_PROCEDURE`, `BODY_PART`, `CHEMICAL`, `ELEMENT`, `PROTEIN`, `GENE`, `BACTERIA`, `VIRUS`, `ANIMAL_SPECIES`, `PLANT_SPECIES`, `MINERAL` |
| **Technology** | `SOFTWARE`, `OS`, `VIDEO_GAME`, `PROGRAMMING_LANGUAGE`, `AI_MODEL`, `HARDWARE_DEVICE`, `VEHICLE_MODEL`, `AIRCRAFT_MODEL`, `SPACECRAFT`, `WEAPON`, `CONSUMER_PRODUCT` |
| **Culture & Media** | `BOOK`, `MOVIE`, `TV_SHOW`, `SONG`, `ALBUM`, `PAINTING`, `LANGUAGE`, `RELIGION` |
| **Events** | `HISTORICAL_EVENT`, `WAR`, `BATTLE`, `SPORT_EVENT`, `TOURNAMENT`, `FESTIVAL`, `CONFERENCE`, `NATURAL_DISASTER` |
| **Legal & Academic** | `LAW`, `TREATY`, `AWARD`, `SCIENTIFIC_THEORY`, `MATHEMATICAL_THEOREM` |
| **Finance** | `CRYPTOCURRENCY`, `STOCK_TICKER` |
| **Numerical** | `DATE`, `TIME`, `MONEY`, `PERCENT`, `TEMPERATURE`, `MEASUREMENT` |

---

## Project Structure

```
Final_Distillation_Final/
│
├── 📓 Notebooks (Google Colab)
│   ├── LLM_Distillation.ipynb          # Main distillation training notebook
│   ├── Test_Data_Generation.ipynb      # Multi-model inference on test sets
│   └── Testing_General.ipynb           # Ad-hoc model testing
│
├── 🔧 Dataset Generation
│   ├── training_data_generation.py     # End-to-end: Gemma sentences + GPT annotation
│   ├── generate_dataset.py             # Dataset builder from pre-generated sentences
│   ├── generate_ner_dataset.py         # Alternative full-pipeline generator
│   ├── generate_fewshot_pool.py        # Standalone few-shot example pool builder
│   └── assemble_dataset.py            # Assembles/merges dataset components
│
├── 🛠️ Dataset Cleaning & Analysis
│   ├── fix_ner_dataset.py              # Label correction + diversity maximisation
│   ├── analyze_fixed_dataset.py        # Post-fix dataset statistics
│   └── analyze_ner_quality.py          # Quality analysis of raw annotations
│
├── 🎓 Training
│   └── NER_Distillation_v2.py          # Standalone distillation training script
│
├── 📊 Evaluation & Testing
│   ├── evaluate_100_samples.py         # Normalised match evaluation (100 samples)
│   ├── evaluate_100_exact.py           # Exact match evaluation (100 samples)
│   ├── evaluate_15_samples.py          # Evaluation on 15-sample pilot set
│   ├── evaluate_test_prompt.py         # Fuzzy + strict evaluation with mode-aware matching
│   ├── generate_test_outputs.py        # Runs inference for all 3 models on test CSV
│   ├── parse_to_csv.py                 # Converts JSONL test set to prompt CSV
│   ├── clean_outputs.py                # Post-inference JSON extraction from model outputs
│   └── sample_csv.py                   # Samples rows from a CSV for quick testing
│
├── 📁 Training Data/
│   └── mixed_ner_dataset_final.jsonl   # 3,000 training records (zero/one/two-shot)
│
├── 📁 Test Data/
│   ├── Test_Data_100.csv               # 100-sample evaluation set (all model outputs)
│   └── Test_Data_15.csv                # 15-sample pilot evaluation set
│
└── 📄 Prompt_Structure.txt             # NER prompt format reference examples
```

---

## Pipeline Walkthrough

### Stage 1 — Synthetic Dataset Generation

The training dataset is built entirely synthetically using a two-model collaboration.

#### Step 1a: Base Sentence Generation (Gemma 3 12B)

`training_data_generation.py` uses Gemma 3 12B to generate diverse, natural English sentences across a wide variety of styles and domains.

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=sk-...       # Linux/Mac
set OPENAI_API_KEY=sk-...          # Windows

python training_data_generation.py \
    --output "Training Data/mixed_ner_dataset_final.jsonl" \
    --total 3000 \
    --seed 42 \
    --resume
```

Generation is guided by a dynamic entity-type hint system that randomly selects 1–4 entity categories per sentence, with 12 writing style variations (news, dialogue, social media, scientific abstract, etc.).

#### Step 1b: Few-Shot Example Annotation (GPT-5.4)

For one-shot and two-shot records, GPT-5.4 is called to produce **topically relevant** annotated example sentences using a **two-turn generate → self-review** pattern:

- **Turn 1** (`temperature=0.7`): Generate diverse annotated examples matching the target sentence's entity domain.
- **Turn 2** (`temperature=0.0`): Self-review and correct any annotation errors (verbatim span check, canonical type enforcement).

The dataset is split equally: **1,000 zero-shot / 1,000 one-shot / 1,000 two-shot**, interleaved and shuffled.

#### Alternatively — from pre-generated sentences:

```bash
python generate_dataset.py \
    --sentences "Training Data/Generated_Sentences.jsonl" \
    --output   "Training Data/mixed_ner_dataset.jsonl" \
    --total 3000 --seed 42 --batch-size 5 --resume
```

---

### Stage 2 — Dataset Cleaning & Normalisation

`fix_ner_dataset.py` applies a three-phase post-processing pass:

**Phase 1 — Hard Label Corrections**
Fixes known mislabellings using an explicit correction table (e.g., `European Space Agency → GOV_AGENCY`, `the Louvre → MUSEUM`, `Artemis I → SPACECRAFT`).

**Phase 2 — Schema Normalisation**
Enforces canonical types for ambiguous entities (e.g., `Reuters → NEWS_AGENCY`, `insulin → PROTEIN`, `Europe → CONTINENT`).

**Phase 3 — Diversity Maximisation**
Redistributes few-shot examples to reduce duplication, using a domain-overlap scoring heuristic to find the best replacement example from the deduplicated pool.

```bash
python fix_ner_dataset.py \
    --input  "Training Data/ner_dataset.jsonl" \
    --output "Training Data/ner_dataset_fixed.jsonl"

# Dry run (report only, no writes):
python fix_ner_dataset.py --dry-run

# In-place fix (creates .bak backup first):
python fix_ner_dataset.py --inplace
```

---

### Stage 3 — Distillation Training

Training is run via `LLM_Distillation.ipynb` (Google Colab) or `NER_Distillation_v2.py` (local).

#### How it works

For each training record:
1. A **mode-aware prompt** is built (zero/one/two-shot) from the JSONL record's `examples` field.
2. The **teacher (12B)** generates the NER JSON output and returns raw logits via `output_logits=True` — no second forward pass required.
3. The **student (270M)** runs a forward pass on the full `[prompt + teacher_output]` sequence.
4. Loss is computed **only over the generated token positions** (prompt tokens are masked).

#### Loss Function

```
Total Loss = λ · KL(student ∥ teacher) + (1 - λ) · CrossEntropy
           = 0.7 · KL + 0.3 · CE
```

- **KL Divergence**: Computed over top-50 teacher logits with Hinton temperature scaling (T=2.0) to soften the teacher distribution.
- **Vocabulary alignment**: Teacher and student vocab sizes are matched by slicing to `min(teacher_vocab, student_vocab)`.
- **EOS enforcement**: Teacher outputs always end with an EOS token so the student learns to stop generating.

```bash
# Local execution:
python NER_Distillation_v2.py
```

> **Note**: Update `DATASET_PATH` in the script (or the Colab notebook cell) to point to your `mixed_ner_dataset_final.jsonl` before running. Checkpoints are saved to `./ner_distill_checkpoints/` every 200 steps, and the final model to `./ner_distilled_model/`.

---

### Stage 4 — Test Data Generation

`Test_Data_Generation.ipynb` (Colab) or `generate_test_outputs.py` (local) runs inference on the test CSV for all three models sequentially (to avoid VRAM OOM):

| Column | Model |
|---|---|
| `Teacher_Output` | `google/gemma-3-12b-it` |
| `Base_Student_Output` | `google/gemma-3-270m-it` (untrained baseline) |
| `Distilled_Student_Output` | Fine-tuned student from `./ner_distilled_model` |

```bash
# Generate prompts from JSONL test set
python parse_to_csv.py

# Run all three models
python generate_test_outputs.py

# Clean over-generated outputs (extract first JSON array)
python clean_outputs.py
```

---

### Stage 5 — Evaluation

Two complementary evaluation scripts are provided:

#### Normalised Evaluation (`evaluate_100_samples.py`)
Applies text normalisation (lowercase, strip punctuation) and type normalisation (abbreviation mapping) before matching. This is a **fair** comparison that tolerates minor formatting differences.

```bash
python evaluate_100_samples.py
```

#### Exact Match Evaluation (`evaluate_100_exact.py`)
Performs **strict** character-level matching on both entity text and type label simultaneously — no normalisation applied.

```bash
python evaluate_100_exact.py
```

Both scripts report **Precision / Recall / F1** for each model × prompt mode combination. Below are the actual performance metrics obtained from the 100-sample test set (`Test_Data_100.csv`):

### Normalised Evaluation Results (evaluate_100_samples.py)
```text
Mode         | Model           | Precision  | Recall     | F1 Score  
-----------------------------------------------------------------
zero_shot    | Teacher         | 0.2206     | 0.2521     | 0.2353    
zero_shot    | Base_Student    | 0.0362     | 0.0420     | 0.0389    
zero_shot    | Distilled       | 0.2101     | 0.2437     | 0.2257    
-----------------------------------------------------------------
one_shot     | Teacher         | 0.6693     | 0.6855     | 0.6773    
one_shot     | Base_Student    | 0.1808     | 0.3790     | 0.2448    
one_shot     | Distilled       | 0.6641     | 0.7016     | 0.6824    
-----------------------------------------------------------------
two_shot     | Teacher         | 0.7213     | 0.6875     | 0.7040    
two_shot     | Base_Student    | 0.1659     | 0.2969     | 0.2129    
two_shot     | Distilled       | 0.6923     | 0.7031     | 0.6977    
```

### Exact Evaluation Results (evaluate_100_exact.py)
```text
Mode         | Model           | TEXT ONLY (P/R/F1)             | STRICT (P/R/F1)
-----------------------------------------------------------------------------------------------
zero_shot    | Teacher         | 0.7868 / 0.8992 / 0.8392       | 0.1544 / 0.1765 / 0.1647
zero_shot    | Base_Student    | 0.3333 / 0.3866 / 0.3580       | 0.0217 / 0.0252 / 0.0233
zero_shot    | Distilled       | 0.7826 / 0.9076 / 0.8405       | 0.1522 / 0.1765 / 0.1634
-----------------------------------------------------------------------------------------------
one_shot     | Teacher         | 0.8819 / 0.9032 / 0.8924       | 0.6693 / 0.6855 / 0.6773
one_shot     | Base_Student    | 0.2510 / 0.5242 / 0.3394       | 0.1654 / 0.3468 / 0.2240
one_shot     | Distilled       | 0.8015 / 0.8468 / 0.8235       | 0.6489 / 0.6855 / 0.6667
-----------------------------------------------------------------------------------------------
two_shot     | Teacher         | 0.8770 / 0.8359 / 0.8560       | 0.7131 / 0.6797 / 0.6960
two_shot     | Base_Student    | 0.2489 / 0.4453 / 0.3193       | 0.1616 / 0.2891 / 0.2073
two_shot     | Distilled       | 0.8846 / 0.8984 / 0.8915       | 0.6923 / 0.7031 / 0.6977
```

#### Advanced Evaluation (`evaluate_test_prompt.py`)
Implements mode-aware fuzzy matching for zero-shot (abbreviation expansion, 0.8 similarity threshold, substring matching, type equivalency buckets) alongside strict matching for one/two-shot modes. Outputs an overall macro-averaged summary.

---

## Prompt Format

All three shot modes follow the same base structure:

```
Extract named entities from the sentence. Return a JSON array only.
Each item: {"entity": "exact text from sentence", "type": "ENTITY_TYPE"}
If no entities exist, return [].

Rules:
- Copy entity text exactly. Do not change it.
- Strip leading "a", "an", "the" from entity text only.
- Skip generic nouns, job titles alone, and abstract concepts.

[Examples: (one-shot / two-shot only)]
Example 1:
Sentence: """<example sentence>"""
Output: [{"entity": "...", "type": "..."}, ...]

---

Sentence: """<target sentence>"""

Output:
```

Zero-shot records omit the `Examples:` block entirely.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 10 |
| Batch size | 1 |
| Learning rate | 5e-5 |
| LR schedule | Cosine with warmup (10%) |
| Gradient clipping | 1.0 |
| `λ_KL` | 0.7 |
| KL temperature | 2.0 |
| Top-K (KL) | 50 |
| Max sequence length | 768 tokens |
| Max new tokens (teacher) | 256 |
| Early stopping patience | 50 batches |
| Min improvement delta | 1e-4 |
| Checkpoint interval | Every 200 steps |
| Random seed | 42 |

---

## Evaluation Metrics

- **Ground Truth Source**: Claude + ChatGPT annotations (human-quality gold labels stored in the `Claude` column of the test CSVs).
- **Evaluation Modes**: Zero-shot, One-shot, Two-shot evaluated independently.
- **Match Types**:
  - **Text-Only Match**: Entity span match only (ignores type).
  - **Strict Match (Text + Type)**: Both entity span and type label must match exactly.
- **Metrics**: Precision, Recall, F1 Score (micro-averaged per mode).

---

## Setup & Usage

### Prerequisites

```bash
pip install transformers accelerate bitsandbytes tqdm openai pandas torch
```

### Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Final_Distillation_Final

# 2. Generate training data (requires OpenAI API key + GPU for Gemma 12B)
export OPENAI_API_KEY=sk-...
python training_data_generation.py --total 3000 --resume

# 3. Clean the dataset
python fix_ner_dataset.py --input "Training Data/mixed_ner_dataset.jsonl" \
                           --output "Training Data/mixed_ner_dataset_final.jsonl"

# 4. Run distillation training (GPU required, ~24GB VRAM for 12B teacher)
python NER_Distillation_v2.py

# 5. Evaluate the distilled model
python evaluate_100_samples.py
python evaluate_100_exact.py
```

### Google Colab Workflow

| Notebook | Purpose |
|---|---|
| `LLM_Distillation.ipynb` | Mount Google Drive, run distillation training, download checkpoints |
| `Test_Data_Generation.ipynb` | Load all three models, run inference on test prompts, save output CSV |
| `Testing_General.ipynb` | Ad-hoc prompt testing and output inspection |

---

## File Reference

| File | Description |
|---|---|
| `NER_Distillation_v2.py` | Complete distillation training script (standalone) |
| `LLM_Distillation.ipynb` | Colab-ready training notebook |
| `training_data_generation.py` | Gemma sentence generation + GPT annotation in one pipeline |
| `generate_dataset.py` | Dataset builder from pre-existing sentence JSONL |
| `generate_ner_dataset.py` | Full alternative NER dataset generator |
| `generate_fewshot_pool.py` | Builds standalone few-shot example pool via GPT |
| `assemble_dataset.py` | Merges/assembles dataset component files |
| `fix_ner_dataset.py` | Label corrections, schema normalisation, diversity maximisation |
| `analyze_fixed_dataset.py` | Statistics on the fixed dataset |
| `analyze_ner_quality.py` | Raw annotation quality analysis |
| `Test_Data_Generation.ipynb` | Multi-model NER inference on test CSVs (Colab) |
| `generate_test_outputs.py` | Local multi-model inference script |
| `parse_to_csv.py` | Converts JSONL test records to prompt CSV format |
| `clean_outputs.py` | Extracts first valid JSON array from model outputs |
| `sample_csv.py` | Samples N rows from a CSV for quick sanity checks |
| `evaluate_100_samples.py` | Normalised Precision/Recall/F1 on 100-sample test set |
| `evaluate_100_exact.py` | Exact-match Precision/Recall/F1 (text-only + strict) |
| `evaluate_15_samples.py` | Evaluation on 15-sample pilot set |
| `evaluate_test_prompt.py` | Advanced fuzzy + strict evaluation with overall summary |
| `Prompt_Structure.txt` | Annotated prompt format examples for reference |
| `Training Data/mixed_ner_dataset_final.jsonl` | Final 3,000-record training dataset |
| `Test Data/Test_Data_100.csv` | 100-sample evaluation set with all model outputs |
| `Test Data/Test_Data_15.csv` | 15-sample pilot evaluation set |

---

## Requirements

```
torch>=2.0
transformers>=4.40
accelerate>=0.28
bitsandbytes>=0.43
tqdm
openai>=1.0
pandas
```

> **Hardware**: The distillation training loop loads both teacher (12B, bfloat16) and student (270M, bfloat16) simultaneously. A GPU with ≥ 24 GB VRAM (e.g., A100 40GB, RTX 4090) is recommended. Google Colab A100 instances work well for this pipeline.

---

## Notes

- The `--resume` flag on all generation scripts allows safe interruption and continuation — no data is lost on crash.
- Checkpoint weights are saved to `ner_distill_checkpoints/step_<N>/` and the best model (by running average loss) is restored automatically before final save.
- The student is trained with full parameter updates — no LoRA, no frozen layers — for maximum knowledge transfer.
