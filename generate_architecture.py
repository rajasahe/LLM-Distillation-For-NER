import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(13, 20))
ax.set_xlim(0, 13)
ax.set_ylim(0, 20)
ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')

# ── colour palette ──
C = {
    'blue':   '#1565C0', 'blue_bg': '#E3F2FD',
    'green':  '#2E7D32', 'green_bg': '#E8F5E9',
    'orange': '#E65100', 'orange_bg': '#FFF3E0',
    'purple': '#4A148C', 'purple_bg': '#F3E5F5',
    'gray':   '#37474F', 'gray_bg':  '#ECEFF1',
    'white':  '#FFFFFF', 'border':   '#90A4AE',
}

def box(ax, x, y, w, h, label, sublabel='', fc='#FFFFFF', ec='#888888',
        lw=1.5, fontsize=9, subfontsize=7.5, bold=False, radius=0.3):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.05,rounding_size={radius}",
                           facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(patch)
    weight = 'bold' if bold else 'normal'
    cy = y + h / 2 + (0.12 if sublabel else 0)
    ax.text(x + w/2, cy, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color='#1A1A2E', zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.18, sublabel, ha='center', va='center',
                fontsize=subfontsize, color='#37474F', zorder=4, style='italic')

def section_label(ax, x, y, text, color):
    ax.text(x, y, text, ha='left', va='center', fontsize=9.5,
            fontweight='bold', color=color, zorder=4,
            fontfamily='monospace')

def arrow(ax, x1, y1, x2, y2, color='#37474F'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8), zorder=5)

def h_line(ax, x1, y, x2, color='#37474F'):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8), zorder=5)

def elbow(ax, x1, y1, xm, ym, x2, y2, color='#37474F'):
    ax.plot([x1, xm, xm], [y1, y1, ym], color=color, lw=1.8, zorder=5)
    ax.annotate('', xy=(x2, y2), xytext=(xm, ym),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8), zorder=5)

# ══════════════════════════════════════════════
# STAGE 1 — DATASET GENERATION
# ══════════════════════════════════════════════
section_label(ax, 0.3, 19.5, '[1]  DATASET GENERATION', C['blue'])

box(ax, 0.3, 18.3, 2.8, 0.8, 'Gemma 3 12B', '(Sentence Generator)',
    fc=C['blue_bg'], ec=C['blue'], bold=True)
box(ax, 4.0, 18.3, 4.2, 0.8, 'Generated Sentences',
    '(diverse domains, entity-type hints)', fc=C['white'], ec=C['border'])
arrow(ax, 3.1, 18.7, 4.0, 18.7)

box(ax, 0.3, 17.1, 2.8, 0.8, 'GPT-5.4', '(Annotator)',
    fc=C['blue_bg'], ec=C['blue'], bold=True)
box(ax, 4.0, 17.1, 4.2, 0.8, 'mixed_ner_dataset.jsonl',
    '3,000 raw records (zero / one / two-shot)', fc=C['white'], ec=C['border'])
arrow(ax, 3.1, 17.5, 4.0, 17.5)
arrow(ax, 1.7, 18.3, 1.7, 17.9)

# annotation: two-turn
ax.text(4.2, 18.05, 'Two-turn annotate + self-review', fontsize=7,
        color='#555', style='italic', va='center')

# ══════════════════════════════════════════════
# STAGE 2 — CLEANING
# ══════════════════════════════════════════════
arrow(ax, 6.1, 17.1, 6.1, 16.6)
section_label(ax, 0.3, 16.45, '[2]  DATA CLEANING & NORMALISATION  (fix_ner_dataset.py)', C['green'])

phase_bg = FancyBboxPatch((0.3, 15.3), 9.5, 0.85, boxstyle='round,pad=0.05,rounding_size=0.2',
                          facecolor=C['green_bg'], edgecolor=C['green'], linewidth=1.4, zorder=3)
ax.add_patch(phase_bg)
phases = [
    '● Phase 1: Hard Label Corrections   (e.g. "Artemis I" → SPACECRAFT)',
    '● Phase 2: Schema Normalisation      (e.g. "Reuters"   → NEWS_AGENCY)',
    '● Phase 3: Diversity Maximisation    (redistribute few-shot examples)',
]
for i, p in enumerate(phases):
    ax.text(0.55, 15.97 - i * 0.27, p, fontsize=7.5, color='#1B5E20', va='center', zorder=4)

arrow(ax, 5.05, 15.3, 5.05, 14.85)
box(ax, 2.8, 14.3, 4.5, 0.8, 'mixed_ner_dataset_final.jsonl',
    '3,000 curated, balanced records', fc=C['white'], ec=C['border'])

# ══════════════════════════════════════════════
# STAGE 3 — DISTILLATION TRAINING
# ══════════════════════════════════════════════
arrow(ax, 5.05, 14.3, 5.05, 13.75)
section_label(ax, 0.3, 13.6, '[3]  DISTILLATION TRAINING  (NER_Distillation_v2.py)', C['orange'])

box(ax, 0.3, 12.3, 3.0, 0.9, 'Teacher — Gemma 3 12B', '(Frozen weights)',
    fc=C['orange_bg'], ec=C['orange'], bold=True)
box(ax, 6.8, 12.3, 3.0, 0.9, 'Student — Gemma 3 270M', '(Full parameter update)',
    fc=C['orange_bg'], ec=C['orange'], bold=True)

# Loss box
loss_bg = FancyBboxPatch((2.5, 11.1), 5.1, 0.85,
                         boxstyle='round,pad=0.05,rounding_size=0.2',
                         facecolor='#FFF8E1', edgecolor=C['orange'], linewidth=1.5, zorder=3)
ax.add_patch(loss_bg)
ax.text(5.05, 11.77, 'Loss Function', ha='center', va='center',
        fontsize=9, fontweight='bold', color=C['orange'], zorder=4)
ax.text(5.05, 11.42, r'$\mathcal{L}$ = 0.7 × KL(T=2, Top-50)  +  0.3 × CrossEntropy',
        ha='center', va='center', fontsize=8.5, color='#1A1A2E', zorder=4)

# arrows into loss
elbow(ax, 1.8, 12.3, 1.8, 11.52, 2.5, 11.52)
ax.text(1.85, 11.95, 'logits + text', fontsize=7, color='#555', rotation=90, va='center')
elbow(ax, 8.3, 12.3, 8.3, 11.52, 7.6, 11.52)
ax.text(7.8, 11.95, 'forward pass', fontsize=7, color='#555', rotation=90, va='center')

arrow(ax, 5.05, 11.1, 5.05, 10.55)
box(ax, 2.8, 9.9, 4.5, 0.85, 'ner_distilled_model/',
    'Gemma 3 270M — fine-tuned student', fc=C['white'], ec=C['border'])

# ══════════════════════════════════════════════
# STAGE 4 — INFERENCE
# ══════════════════════════════════════════════
arrow(ax, 5.05, 9.9, 5.05, 9.4)
section_label(ax, 0.3, 9.25, '[4]  MULTI-MODEL INFERENCE  (generate_test_outputs.py)', C['purple'])

box(ax, 3.3, 8.25, 4.5, 0.75, 'Test_Data_100.csv', '100 unseen evaluation samples',
    fc=C['white'], ec=C['border'])

# three output boxes
box(ax, 0.2, 6.9, 3.2, 0.75, 'Teacher 12B', 'Teacher_Output',
    fc=C['purple_bg'], ec=C['purple'])
box(ax, 4.45, 6.9, 3.2, 0.75, 'Base Student 270M', 'Base_Student_Output',
    fc=C['purple_bg'], ec=C['purple'])
box(ax, 8.7, 6.9, 3.2, 0.75, 'Distilled 270M', 'Distilled_Student_Output',
    fc=C['purple_bg'], ec=C['purple'])

elbow(ax, 5.05, 8.25, 5.05, 8.0, 1.8, 8.0); arrow(ax, 1.8, 8.0, 1.8, 7.65)
arrow(ax, 6.05, 8.25, 6.05, 7.65)
elbow(ax, 5.05, 8.25, 5.05, 8.0, 10.3, 8.0); arrow(ax, 10.3, 8.0, 10.3, 7.65)

# ══════════════════════════════════════════════
# STAGE 5 — EVALUATION
# ══════════════════════════════════════════════
arrow(ax, 1.8, 6.9, 1.8, 6.3); arrow(ax, 6.05, 6.9, 6.05, 6.3); arrow(ax, 10.3, 6.9, 10.3, 6.3)
ax.plot([1.8, 6.05, 10.3], [6.3, 6.3, 6.3], color=C['gray'], lw=1.8, zorder=5)
arrow(ax, 6.05, 6.3, 6.05, 5.9)

section_label(ax, 0.3, 5.75, '[5]  EVALUATION  (vs Claude Ground Truth)', C['gray'])

eval_bg = FancyBboxPatch((0.3, 4.5), 11.5, 1.0, boxstyle='round,pad=0.05,rounding_size=0.2',
                         facecolor=C['gray_bg'], edgecolor=C['gray'], linewidth=1.4, zorder=3)
ax.add_patch(eval_bg)
evals = [
    '● Exact Text Match          →  evaluate_100_exact.py       (character-level span matching)',
    '● Normalized Entity Type  →  evaluate_100_samples.py  (type mapping + text normalisation)',
    '● Metrics: Precision / Recall / F1  per prompt mode  (Zero-shot · One-shot · Two-shot)',
]
for i, e in enumerate(evals):
    ax.text(0.6, 5.28 - i * 0.27, e, fontsize=7.8, color='#263238', va='center', zorder=4)

# ground truth note
ax.text(6.05, 4.3, '★  Ground Truth = Claude-annotated NER labels (Test_Data_100.csv)',
        ha='center', va='center', fontsize=8, color='#4A148C',
        fontweight='bold', zorder=4,
        bbox=dict(boxstyle='round,pad=0.3', fc='#F3E5F5', ec='#7B1FA2', lw=1))

plt.tight_layout(pad=0.2)
plt.savefig('architecture_diagram.png', dpi=180, bbox_inches='tight',
            facecolor='#F8F9FA')
print("Saved architecture_diagram.png")
