import json
from collections import Counter, defaultdict

path = r'Training Data/ner_dataset_fixed.jsonl'

records = []
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception as e:
            print(f'Parse error on line {i+1}: {e}')

print(f'Total records: {len(records)}')

# Shot type distribution
shot_types = Counter(r['type'] for r in records)
print('\n--- Shot type distribution ---')
for k, v in sorted(shot_types.items()):
    print(f'  {k}: {v}')

# Example count consistency
zero_with_examples = sum(1 for r in records if r['type']=='zero_shot' and len(r['examples'])>0)
one_with_wrong_count = sum(1 for r in records if r['type']=='one_shot' and len(r['examples'])!=1)
two_with_wrong_count = sum(1 for r in records if r['type']=='two_shot' and len(r['examples'])!=2)
print(f'\n--- Example count consistency ---')
print(f'  zero_shot with examples: {zero_with_examples}')
print(f'  one_shot with != 1 example: {one_with_wrong_count}')
print(f'  two_shot with != 2 examples: {two_with_wrong_count}')

# Duplicate main sentences
sentences = [r['sentence'] for r in records]
print(f'\n--- Duplicate main sentences: {len(sentences) - len(set(sentences))} ---')

# Few-shot example duplication
example_sentences = []
for r in records:
    for ex in r['examples']:
        example_sentences.append(ex['sentence'])
dup_ex = len(example_sentences) - len(set(example_sentences))
print(f'--- Total example sentences: {len(example_sentences)} ---')
print(f'--- Duplicate example sentences: {dup_ex} ({100*dup_ex/max(len(example_sentences),1):.1f}%) ---')
print(f'--- Unique example sentences: {len(set(example_sentences))} ---')

# Most reused
ex_counter = Counter(example_sentences)
print(f'\n--- Top 15 most reused example sentences ---')
for sent, count in ex_counter.most_common(15):
    print(f'  ({count}x) {sent[:120]}')

# Check previously known errors — are they fixed?
print(f'\n--- Checking previously known errors (should be 0) ---')
known_errors = [
    ('European Space Agency', 'OCEAN'),
    ('Artemis I', 'PROGRAMMING_LANGUAGE'),
    ('the Louvre', 'PAINTING'),
    ('Leonardo da Vinci', 'PAINTING'),
    ('Asia', 'COUNTRY'),
    ('Colombo', 'GOV_AGENCY'),
    ('PyCon US', 'PERSON'),
    ('Jude Bellingham', 'SPORT_TEAM'),
    ('Vinicius Junior', 'SPORT_TEAM'),
    ('Austin', 'STATE'),
    ('Stowe', 'STATE'),
]
all_clear = True
for entity_text, wrong_type in known_errors:
    count = sum(
        1 for r in records for ex in r['examples']
        for ent in ex.get('entities', [])
        if ent['entity'] == entity_text and ent['type'] == wrong_type
    )
    status = '[OK]' if count == 0 else f'[!!] still {count}x'
    if count > 0:
        all_clear = False
    print(f'  {status}  "{entity_text}" as {wrong_type}')
print(f'\n  All errors fixed: {"YES" if all_clear else "SOME REMAIN"}')

# Check canonical normalisation applied
print(f'\n--- Canonical normalisation check ---')
canon_checks = [
    ('Reuters', ['NEWS_AGENCY'], ['COMPANY']),
    ('Europe', ['CONTINENT'], ['GOV_AGENCY', 'COUNTRY']),
    ('Asia', ['CONTINENT'], ['COUNTRY', 'GOV_AGENCY']),
    ('Germany', ['COUNTRY'], ['GOV_AGENCY']),
    ('insulin', ['PROTEIN'], ['DRUG']),
    ('argon', ['ELEMENT'], ['CHEMICAL']),
    ('Napoleon Bonaparte', ['HISTORICAL_FIGURE'], ['PERSON']),
]
for entity, good_types, bad_types in canon_checks:
    good = bad = 0
    for r in records:
        for ex in r['examples']:
            for ent in ex.get('entities', []):
                if ent['entity'] == entity:
                    if ent['type'] in good_types:
                        good += 1
                    elif ent['type'] in bad_types:
                        bad += 1
    status = '[OK]' if bad == 0 else f'[!!] {bad} bad remaining'
    print(f'  {status}  "{entity}": {good} correct, {bad} wrong')

# Remaining inconsistencies
print(f'\n--- Entities still with inconsistent type labels ---')
entity_types_seen = defaultdict(Counter)
for r in records:
    for ex in r['examples']:
        for ent in ex.get('entities', []):
            entity_types_seen[ent['entity']][ent['type']] += 1

inconsistent = 0
for entity, type_counter in sorted(entity_types_seen.items()):
    if len(type_counter) > 1:
        inconsistent += 1
        if inconsistent <= 20:
            print(f'  "{entity}": {dict(type_counter)}')
print(f'\n  Total entities with inconsistent types: {inconsistent}')

# Entity type distribution in pool
all_entity_types = Counter()
for r in records:
    for ex in r['examples']:
        for ent in ex.get('entities', []):
            all_entity_types[ent['type']] += 1

print(f'\n--- Entity type distribution in pool (top 30) ---')
for etype, count in all_entity_types.most_common(30):
    print(f'  {etype}: {count}')
print(f'\n  Total unique entity types: {len(all_entity_types)}')

# Entity text not found in example sentence
mismatch_count = 0
for r in records:
    for ex in r['examples']:
        for ent in ex.get('entities', []):
            if ent['entity'].lower() not in ex['sentence'].lower():
                mismatch_count += 1
print(f'\n--- Entity spans not found verbatim in sentence: {mismatch_count} ---')

# Sentence length stats
lengths = [len(r['sentence'].split()) for r in records]
print(f'\n--- Main sentence length (words) ---')
print(f'  Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}')
