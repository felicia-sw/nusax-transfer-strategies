"""Figure 1 for the Week 5 Methods draft: the four strategies as one pipeline.

    python src/fig_methods_pipeline.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "results" / "fig1_methods_pipeline.png"

NAVY, TEAL, GREY, PALE = "#1B2A4A", "#0D7C8F", "#5B6573", "#EEF3F7"

fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 54)
ax.axis("off")


def box(x, y, w, h, title, sub="", fill=PALE, edge=NAVY, tc=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fill, ec=edge, lw=1.1))
    ax.text(x + w / 2, y + h / 2 + (1.4 if sub else 0), title, ha="center", va="center",
            fontsize=8.6, fontweight="bold", color=tc, family="Arial")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.0, sub, ha="center", va="center",
                fontsize=7, color=GREY, family="Arial", linespacing=1.15)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 lw=0.9, color=GREY, shrinkA=0, shrinkB=0))


# data column: NusaX-Senti, one box per input the strategies consume
ax.text(1, 51.5, "NusaX-Senti  \u00b7  12 parallel languages, 3 classes", fontsize=8,
        fontweight="bold", color=NAVY, family="Arial")
box(1, 39.8, 17, 7, "Target train", "500 sentences", fill="white")
box(1, 27.8, 17, 7, "Indonesian train", "500 sentences", fill="white")
box(1, 3.8, 17, 19, "Target test", "400 sentences\n(identical across\nall 12 languages)", fill="white")

rows = [  # y, strategy, step 1, step 2
    (42, "S1  Direct fine-tuning", ("IndoBERT fine-tuned", "on target-language train"), None),
    (30, "S2  Zero-shot transfer", ("IndoBERT fine-tuned", "on Indonesian train (once)"), None),
    (18, "S3  Translate-test", ("NLLB-200 (600M)", "target test \u2192 Indonesian"),
     ("Indonesian IndoBERT", "same model as S2")),
    (6, "S4  LLM prompting", ("Qwen2.5-3B-Instruct", "fixed prompt, label-token logits"), None),
]
for y, name, s1, s2 in rows:
    ax.text(23, y + 5.6, name, fontsize=8, fontweight="bold", color=TEAL, family="Arial")
    box(23, y - 2.2, 22, 7, s1[0], s1[1])
    if s2:
        box(49, y - 2.2, 21, 7, s2[0], s2[1])
        arrow(45.6, y + 1.3, 48.6, y + 1.3)
        arrow(70.6, y + 1.3, 75.6, y + 1.3)
    else:
        arrow(45.6, y + 1.3, 75.6, y + 1.3)

arrow(18.6, 43.3, 22.6, 43.3)          # target train -> S1
arrow(18.6, 31.3, 22.6, 31.3)          # Indonesian train -> S2
arrow(18.6, 19.3, 22.6, 19.3)          # target test -> S3 translation
arrow(18.6, 7.3, 22.6, 7.3)            # target test -> S4

ax.add_patch(FancyBboxPatch((76, 2), 23, 47, boxstyle="round,pad=0.25,rounding_size=1.2",
                            fc=NAVY, ec=NAVY, lw=1.1))
ax.text(87.5, 45, "Evaluation", ha="center", va="center", fontsize=9.5,
        fontweight="bold", color="white", family="Arial")
ax.text(87.5, 25, "scored on the\ntarget-language test\n\nmacro-F1 (primary)\naccuracy\n\n"
        "3 seeds for S1 and S3\n\nFriedman + Nemenyi\nWilcoxon signed-rank\nSpearman \u03c1",
        ha="center", va="center", fontsize=7.6, color="#D6E2EC", family="Arial", linespacing=1.3)

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(OUT)
