"""Publication figures: vector SVG plus 300 dpi PNG, as the rubric asks.

Every figure reads `outputs/macro_f1_matrix.csv`, so whatever conditions have
been run is what gets drawn. Nothing here recomputes a score.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nusax_data as nd
from stats import friedman_nemenyi

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "outputs" / "figures"

# Okabe-Ito: colourblind-safe, prints legibly in greyscale.
INK = "#1a1a1a"
PALETTE = {
    "direct": "#0072B2", "direct_bert": "#56B4E9",
    "transfer": "#D55E00", "transfer_bert": "#E69F00",
    "translate": "#009E73", "translate_bert": "#66C2A5",
    "zeroshot": "#CC79A7",
}
NICE = {
    "direct": "Direct (char n-gram)", "direct_bert": "Direct (IndoBERT)",
    "transfer": "Transfer (char n-gram)", "transfer_bert": "Transfer (IndoBERT)",
    "translate": "Translate-test (char n-gram)",
    "translate_bert": "Translate-test (IndoBERT)",
    "zeroshot": "LLM zero-shot",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.linewidth": 0.8,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})


def save(fig, stem: str) -> None:
    """Write one figure in all three formats. Called by `main`, not by the
    figure functions themselves, so a notebook can render them instead."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.svg", transparent=True)
    fig.savefig(FIG / f"{stem}.pdf", transparent=True)
    fig.savefig(FIG / f"{stem}.png", dpi=300, facecolor="white")
    print(f"  wrote {stem}.svg / .pdf / .png")


def matrix() -> pd.DataFrame:
    w = pd.read_csv(ROOT / "outputs" / "macro_f1_matrix.csv", index_col=0)
    order = [c for c in PALETTE if c in w.columns]
    return w[order]


# ------------------------------------------------------------------ figures

def fig_by_language(w: pd.DataFrame):
    """Grouped bars: every condition on every language."""
    loc = w.loc[nd.LOCAL]
    x = np.arange(len(loc))
    conds = list(loc.columns)
    width = 0.8 / len(conds)

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    for i, c in enumerate(conds):
        ax.bar(x + i * width - 0.4 + width / 2, loc[c].values, width,
               label=NICE[c], color=PALETTE[c], edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([nd.LANGS[l] for l in loc.index], rotation=35, ha="right")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1.0)
    ax.axhline(1 / 3, ls=":", lw=0.9, color=INK, alpha=.55)
    ax.text(len(loc) - 0.45, 1 / 3 + .015, "chance", fontsize=7.5, alpha=.7)
    ax.legend(frameon=False, ncols=3, fontsize=7.5, loc="upper left",
              bbox_to_anchor=(0, 1.22))
    ax.grid(axis="y", lw=.4, alpha=.25)
    ax.set_axisbelow(True)
    return fig


def fig_degradation(w: pd.DataFrame):
    """How far each strategy falls from Indonesian, languages ordered by drop."""
    base = w.loc["indonesian"]
    loc = w.loc[nd.LOCAL]
    order = (loc["transfer_bert"] if "transfer_bert" in loc else loc.iloc[:, 0]) \
        .sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for c in loc.columns:
        ax.plot(range(len(order)), loc.loc[order, c] - base[c], marker="o",
                ms=4, lw=1.5, color=PALETTE[c], label=NICE[c])
    ax.axhline(0, color=INK, lw=.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([nd.LANGS[l] for l in order], rotation=35, ha="right")
    ax.set_ylabel("macro-F1 relative to Indonesian")
    ax.legend(frameon=False, fontsize=7.5, ncols=2)
    ax.grid(axis="y", lw=.4, alpha=.25)
    ax.set_axisbelow(True)
    return fig


def fig_cd(w: pd.DataFrame, conds: list[str], title: str):
    """Critical-difference diagram: average ranks, with the Nemenyi bar.

    Rank 1 (best) sits on the right. Labels get their own margin outside the
    rank axis so connectors never run underneath text.
    """
    sub = w.loc[nd.LOCAL, conds].dropna()
    s = friedman_nemenyi({c: sub[c].tolist() for c in conds})
    ranks = s["avg_ranks"]
    cd, n, k = s["cd"], s["n_blocks"], len(conds)

    # The x-axis is inverted (rank 1 on the right), so *smaller* data values
    # sit further right. Labels therefore use ha="left" and are placed at a
    # value below the elbow, which draws them rightward into the free margin.
    label_pad = 1.75          # rank units reserved for text, right of rank 1
    fig, ax = plt.subplots(figsize=(7.4, 1.25 + 0.42 * k))
    ax.set_xlim(k + 0.30, 1 - label_pad)
    ax.axis("off")

    # Rank axis across the rank range only.
    top = k
    ax.hlines(top, 1, k, color=INK, lw=1.1)
    for t in range(1, k + 1):
        ax.vlines(t, top - 0.11, top, color=INK, lw=1.1)
        ax.text(t, top + 0.15, str(t), ha="center", va="bottom", fontsize=8.5)

    # Thick bars join strategies whose ranks differ by less than the CD, i.e.
    # the Nemenyi test cannot separate them. Only maximal groups are drawn.
    rs = sorted(ranks.values())
    groups = []
    for i in range(k):
        j = max(j for j in range(i, k) if rs[j] - rs[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))
    for g, (i, j) in enumerate(groups):
        y = top - 0.20 - g * 0.13
        ax.plot([rs[i] - 0.03, rs[j] + 0.03], [y, y], color=INK, lw=3.2, solid_capstyle="butt", zorder=4)

    elbow = 1 - 0.30          # where connectors stop
    text_x = elbow - 0.06     # labels start here and run right into the margin
    last_y = top - 0.58 - (k - 1) * 0.50
    for i, (name, r) in enumerate(sorted(ranks.items(), key=lambda kv: kv[1])):
        y = top - 0.58 - i * 0.50
        col = PALETTE[name]
        ax.plot([r, r], [y, top], color=col, lw=1.4, solid_capstyle="round")
        ax.plot([r, elbow], [y, y], color=col, lw=1.4, solid_capstyle="round")
        ax.plot(r, top, "o", ms=5.5, color=col, zorder=3)
        ax.text(text_x, y, f"{NICE[name]}  ({r:.2f})",
                ha="left", va="center", fontsize=8.5)

    # Nemenyi bar, anchored at the worst rank.
    bar_y = last_y - 0.55
    ax.set_ylim(bar_y - 0.48, top + 0.52)
    ax.plot([k, max(1, k - cd)], [bar_y, bar_y], color=INK, lw=2.6,
            solid_capstyle="butt")
    ax.text((k + max(1, k - cd)) / 2, bar_y - 0.12, f"CD = {cd:.2f}",
            ha="center", va="top", fontsize=8.5)

    fp = "p < .001" if s["p"] < .001 else f"p = {s['p']:.3f}".replace("0.", ".")
    ax.set_title(f"{title}\nFriedman {fp},  N = {n} languages",
                 fontsize=9, pad=8)
    return fig


def fig_delta(w: pd.DataFrame):
    """Translate-test minus direct training, per language."""
    if not {"translate_bert", "direct_bert"} <= set(w.columns):
        return None
    d = (w.loc[nd.LOCAL, "translate_bert"] - w.loc[nd.LOCAL, "direct_bert"]).dropna()
    d = d.sort_values()
    colours = ["#009E73" if v > 0 else "#D55E00" for v in d]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.barh([nd.LANGS[i] for i in d.index], d.values, color=colours, edgecolor="none")
    ax.axvline(0, color=INK, lw=.9)
    ax.set_xlabel("macro-F1 difference  (translate-test $-$ direct)")
    for y, v in enumerate(d.values):
        ax.text(v + (.002 if v > 0 else -.002), y, f"{v:+.3f}", va="center",
                ha="left" if v > 0 else "right", fontsize=7.5)
    ax.set_xlim(min(d.min() * 1.5, -.02), max(d.max() * 1.45, .02))
    ax.grid(axis="x", lw=.4, alpha=.25)
    ax.set_axisbelow(True)
    return fig


def fig_coverage(w: pd.DataFrame):
    """Which languages a translate-test pipeline can even reach."""
    if "translate_bert" not in w.columns:
        return None
    loc = w.loc[nd.LOCAL]
    covered = loc["translate_bert"].notna()

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.barh([nd.LANGS[i] for i in loc.index],
            [1] * len(loc),
            color=["#009E73" if c else "#D55E00" for c in covered],
            edgecolor="none")
    for y, c in enumerate(covered):
        ax.text(.5, y, "reachable" if c else "no NLLB-200 language code",
                ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title(f"Machine-translation coverage: {covered.sum()} of {len(loc)} "
                 "local languages", fontsize=9)
    return fig


def main() -> None:
    # Headless only when run as a script. Forcing it at import time would
    # disable inline rendering for any notebook that imports this module.
    matplotlib.use("Agg")
    w = matrix()
    print(f"drawing from {len(w.columns)} conditions: {', '.join(w.columns)}")
    save(fig_by_language(w), "fig1_by_language")
    save(fig_degradation(w), "fig2_degradation")
    # Main test: these three need no machine translation, so all ten local
    # languages are usable and the test keeps its power.
    trio = [c for c in ["direct_bert", "transfer_bert", "zeroshot"]
            if c in w.columns]
    if len(trio) == 3:
        save(fig_cd(w, trio, "Average ranks: strategies needing no machine translation"),
             "fig3_critical_difference")
    # Secondary: adds translate-test, at the cost of three languages.
    quad = [c for c in ["direct_bert", "transfer_bert", "translate_bert", "zeroshot"]
            if c in w.columns]
    if len(quad) == 4:
        save(fig_cd(w, quad, "Average ranks including translate-test (MT-covered languages)"),
             "fig6_critical_difference_mt")
    save(fig_delta(w), "fig4_translate_delta")
    save(fig_coverage(w), "fig5_mt_coverage")


if __name__ == "__main__":
    main()
