import json
from collections import Counter, defaultdict

path = r'Training Data/ner_dataset.jsonl'

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

# Examples count
zero_with_examples = sum(1 for r in records if r['type']=='zero_shot' and len(r['examples'])>0)
one_with_wrong_count = sum(1 for r in records if r['type']=='one_shot' and len(r['examples'])!=1)
two_with_wrong_count = sum(1 for r in records if r['type']=='two_shot' and len(r['examples'])!=2)
print(f'\n--- Example count consistency ---')
print(f'  zero_shot with examples (should be 0): {zero_with_examples}')
print(f'  one_shot with != 1 example: {one_with_wrong_count}')
print(f'  two_shot with != 2 examples: {two_with_wrong_count}')

# Sentence length analysis
sentence_lengths = [len(r['sentence'].split()) for r in records]
print(f'\n--- Sentence length (words) ---')
print(f'  Min: {min(sentence_lengths)}')
print(f'  Max: {max(sentence_lengths)}')
print(f'  Avg: {sum(sentence_lengths)/len(sentence_lengths):.1f}')
short = sum(1 for l in sentence_lengths if l < 10)
long = sum(1 for l in sentence_lengths if l > 150)
print(f'  < 10 words: {short}')
print(f'  > 150 words: {long}')

# Entity type distribution across example pool
all_entity_types = Counter()
for r in records:
    for ex in r['examples']:
        for ent in ex.get('entities', []):
            all_entity_types[ent['type']] += 1

print(f'\n--- Entity type distribution (top 30) ---')
for etype, count in all_entity_types.most_common(30):
    print(f'  {etype}: {count}')

print(f'\n--- Total unique entity types in pool: {len(all_entity_types)} ---')

# Duplicate sentences
sentences = [r['sentence'] for r in records]
dup_sents = len(sentences) - len(set(sentences))
print(f'\n--- Duplicate main sentences: {dup_sents} ---')

# Duplicate few-shot examples
example_sentences = []
for r in records:
    for ex in r['examples']:
        example_sentences.append(ex['sentence'])
dup_ex = len(example_sentences) - len(set(example_sentences))
print(f'--- Total example sentences: {len(example_sentences)} ---')
print(f'--- Duplicate example sentences: {dup_ex} ({100*dup_ex/max(len(example_sentences),1):.1f}%) ---')

# Most reused example sentences
ex_counter = Counter(example_sentences)
print(f'\n--- Top 15 most reused example sentences ---')
for sent, count in ex_counter.most_common(15):
    print(f'  ({count}x) {sent[:120]}')

# Known annotation errors
known_errors = [
    ('European Space Agency', 'OCEAN'),
    ('Artemis I', 'PROGRAMMING_LANGUAGE'),
    ('Colombo', 'GOV_AGENCY'),
    ('Beira Lake', 'GOV_AGENCY'),
    ('Gangaramaya Temple', 'RELIGION'),
    ('Austin', 'STATE'),
    ('Stowe', 'STATE'),
    ('the Louvre', 'PAINTING'),
    ('Leonardo da Vinci', 'PAINTING'),
    ('Sandro Botticelli', 'PAINTING'),
    ('Asia', 'COUNTRY'),
    ('Europe', 'GOV_AGENCY'),
    ('PyCon US', 'PERSON'),
    ('Salt Lake City', 'GOV_AGENCY'),
    ('Geneva', 'GOV_AGENCY'),
    ('Amsterdam', 'GOV_AGENCY'),
    ('Monterey', 'GOV_AGENCY'),
    ('Amritsar', 'GOV_AGENCY'),
    ('Jude Bellingham', 'SPORT_TEAM'),
    ('Vinicius Junior', 'SPORT_TEAM'),
]
print(f'\n--- Known annotation errors ---')
for entity_text, wrong_type in known_errors:
    count = 0
    for r in records:
        for ex in r['examples']:
            for ent in ex.get('entities', []):
                if ent['entity'] == entity_text and ent['type'] == wrong_type:
                    count += 1
    if count > 0:
        print(f'  "{entity_text}" labeled as {wrong_type}: {count} occurrence(s)')

# Scan all entity annotations for suspicious type mismatches
suspicious_combos = defaultdict(Counter)
for r in records:
    for ex in r['examples']:
        for ent in ex.get('entities', []):
            suspicious_combos[ent['entity']][ent['type']] += 1

# Entities labeled with multiple different types
print(f'\n--- Entities with inconsistent type labels ---')
inconsistent = 0
for entity, type_counter in suspicious_combos.items():
    if len(type_counter) > 1:
        inconsistent += 1
        if inconsistent <= 20:
            print(f'  "{entity}": {dict(type_counter)}')
print(f'\n  Total entities with inconsistent types: {inconsistent}')

# Entity text not found in example sentence
mismatch_count = 0
mismatch_samples = []
for r in records:
    for ex in r['examples']:
        sent = ex['sentence']
        for ent in ex.get('entities', []):
            if ent['entity'].lower() not in sent.lower():
                mismatch_count += 1
                if len(mismatch_samples) < 10:
                    mismatch_samples.append((ent['entity'], ent['type'], sent[:100]))
print(f'\n--- Entity text NOT FOUND in example sentence: {mismatch_count} ---')
for e, t, s in mismatch_samples:
    print(f'  Entity: "{e}" ({t})')
    print(f'  Sentence: {s}')
    print()

# Highly entity-dense examples (possible overcrowding)
print(f'\n--- Examples with > 7 entities ---')
dense_count = 0
for r in records:
    for ex in r['examples']:
        n = len(ex.get('entities', []))
        if n > 7:
            dense_count += 1
            print(f'  ({n} entities) {ex["sentence"][:100]}')
print(f'  Total overly dense examples: {dense_count}')

# Check for empty entity lists 
empty_entity_examples = 0
for r in records:
    for ex in r['examples']:
        if len(ex.get('entities', [])) == 0:
            empty_entity_examples += 1
print(f'\n--- Examples with 0 entities: {empty_entity_examples} ---')
