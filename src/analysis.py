"""Numbers and figures for the results reported in the article.

Reads only committed outputs: the results matrix, the seed runs, the cached
zero-shot predictions, and the seed-42 predictions written by
save_predictions.py. Writes stats.json and five figures to results/, or to the
folder given as the first argument. The article uses --three-seed-transfer:
zero-shot transfer then uses the mean of the seed-42, 7 and 1234 Indonesian models
(outputs/srw/encoder_runs.csv) instead of seed 42 alone, and stats.json gains the
robustness checks run for the journal version (XLM-R, LLM calibration, chrF).

    python src/analysis.py [out_dir] [--three-seed-transfer]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

import eda
import figures as F
import nusax_data as nd
from stats import friedman_nemenyi

ROOT = Path(__file__).resolve().parent.parent
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(ARGS[0]).resolve() if ARGS else ROOT / "results"
THREE_SEED_TRANSFER = "--three-seed-transfer" in sys.argv
SRW = ROOT / "outputs" / "srw"
PRED = ROOT / "outputs" / "predictions"
LABELS = ["negative", "neutral", "positive"]
MAIN = ["direct_bert", "transfer_bert", "translate_bert", "zeroshot"]
NAME = {"direct_bert": "Direct fine-tuning", "transfer_bert": "Zero-shot transfer",
        "translate_bert": "Translate-test", "zeroshot": "LLM prompting",
        "direct": "Direct (TF-IDF + LR)", "transfer": "Transfer (TF-IDF + LR)",
        "translate": "Translate-test (TF-IDF + LR)"}


def preds(cond: str, lang: str) -> pd.DataFrame | None:
    f = (ROOT / "data" / "zeroshot" / f"{lang}.csv") if cond == "zeroshot" else PRED / f"{cond}_{lang}.csv"
    return pd.read_csv(f)[["label", "pred"]] if f.exists() else None


def t_ci(x) -> list[float]:
    x = np.asarray(pd.Series(x).dropna(), float)
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(ROOT / "outputs" / "macro_f1_matrix.csv", index_col=0)
    loc = m.loc[nd.LOCAL]
    seed = pd.read_csv(ROOT / "outputs" / "seed_summary.csv", index_col=0)
    sim = eda.similarity_to_indonesian()
    S: dict = {}
    if THREE_SEED_TRANSFER:
        enc = pd.read_csv(SRW / "encoder_runs.csv")
        tr = enc[(enc.encoder == "indobert") & (enc.strategy == "transfer")].groupby("lang").macro_f1
        assert (tr.count().reindex(nd.LOCAL) == 3).all(), "three seeds per regional language"
        m.loc[nd.LOCAL, "transfer_bert"] = tr.mean().reindex(nd.LOCAL)
        seed["transfer_bert_sd"] = tr.std(ddof=1)
        S["transfer_seed_range"] = (tr.max() - tr.min()).round(4).to_dict()
        loc = m.loc[nd.LOCAL]

    # ---- overall (Table 1)
    trio = friedman_nemenyi({c: loc[c].tolist() for c in ["direct_bert", "transfer_bert", "zeroshot"]})
    cov = loc[MAIN].dropna()
    quad = friedman_nemenyi({c: cov[c].tolist() for c in MAIN})
    S["overall"] = {c: {"mean": float(loc[c].mean()), "sd": float(loc[c].std(ddof=1)),
                        "ci": t_ci(loc[c]), "min": float(loc[c].min()), "max": float(loc[c].max()),
                        "n": int(loc[c].notna().sum()), "indonesian": float(m.loc["indonesian", c]),
                        "english": float(m.loc["english", c]),
                        "rank10": trio["avg_ranks"].get(c), "rank7": quad["avg_ranks"].get(c),
                        "mean7": float(cov[c].mean()) if c in cov else None}
                    for c in m.columns}
    S["friedman10"] = {"chi2": trio["statistic"], "p": trio["p"], "cd": trio["cd"],
                       "W": trio["statistic"] / (10 * 2), "pairs": trio["pairs"]}
    S["friedman7"] = {"chi2": quad["statistic"], "p": quad["p"], "cd": quad["cd"],
                      "W": quad["statistic"] / (7 * 3), "pairs": quad["pairs"]}

    # ---- pairwise Wilcoxon with effect size (matched-pairs rank-biserial)
    def pair(a, b, frame):
        d = (frame[a] - frame[b]).dropna()
        r = st.rankdata(abs(d))
        tp, tm = r[d > 0].sum(), r[d < 0].sum()
        w = st.wilcoxon(frame.loc[d.index, a], frame.loc[d.index, b])
        return {"gap": float(d.mean()), "ci": t_ci(d), "wins": int((d > 0).sum()), "n": len(d),
                "W": float(w.statistic), "p": float(w.pvalue), "r_rb": float((tp - tm) / (tp + tm))}
    S["pairs"] = {f"{a}|{b}": pair(a, b, loc) for a, b in [
        ("translate_bert", "direct_bert"), ("direct_bert", "transfer_bert"), ("direct_bert", "zeroshot"),
        ("transfer_bert", "zeroshot"), ("direct_bert", "direct"), ("transfer_bert", "transfer"),
        ("translate_bert", "translate")]}

    # ---- single seed vs three seeds (sensitivity)
    sr = pd.read_csv(ROOT / "outputs" / "seed_runs.csv").drop_duplicates(["lang", "seed"], keep="last")
    sx = pd.read_csv(ROOT / "outputs" / "seed_translate.csv").drop_duplicates(["lang", "seed"], keep="last")
    a = sr[sr.seed == 42].set_index("lang").macro_f1
    b = sx[sx.seed == 42].set_index("lang").macro_f1
    j = b.index
    w1 = st.wilcoxon(b[j], a[j])
    S["seed"] = {"single_gap": float((b[j] - a[j]).mean()), "single_wins": int((b[j] > a[j]).sum()),
                 "single_p": float(w1.pvalue), "single_W": float(w1.statistic),
                 "range": sr.groupby("lang").macro_f1.agg(lambda x: x.max() - x.min()).round(4).to_dict(),
                 "sd_max": float(seed[["direct_bert_sd", "translate_bert_sd"]].max().max())}

    # ---- similarity correlations
    sim = sim.loc[nd.LOCAL]
    S["spearman"] = {}
    for c in ["direct_bert", "transfer_bert", "zeroshot"]:
        for meas in ["lexical_overlap", "trigram_similarity"]:
            r, p = st.spearmanr(sim[meas], loc[c])
            S["spearman"][f"{c}|{meas}"] = {"rho": float(r), "p": float(p)}
    S["similarity"] = sim[["lexical_overlap", "trigram_similarity"]].round(3).to_dict(orient="index")

    # ---- per-class, pooled over regional languages (Table 3, Figure 2)
    S["per_class"], S["confusion"], S["seed42"] = {}, {}, {}
    for c in ["direct", "direct_bert", "transfer_bert", "translate_bert", "zeroshot"]:
        parts = [(l, preds(c, l)) for l in nd.LOCAL]
        parts = [(l, p) for l, p in parts if p is not None]
        if not parts:
            continue
        allp = pd.concat([p for _, p in parts])
        pr, rc, f1, sup = precision_recall_fscore_support(allp.label, allp.pred, labels=LABELS, zero_division=0)
        S["per_class"][c] = {"langs": len(parts), "accuracy": float(accuracy_score(allp.label, allp.pred)),
                             "macro_f1_pooled": float(f1_score(allp.label, allp.pred, average="macro")),
                             **{lab: {"p": float(pr[i]), "r": float(rc[i]), "f1": float(f1[i]), "n": int(sup[i])}
                                for i, lab in enumerate(LABELS)},
                             "pred_share": allp.pred.value_counts(normalize=True).reindex(LABELS).fillna(0).to_dict()}
        S["confusion"][c] = confusion_matrix(allp.label, allp.pred, labels=LABELS).tolist()
        S["seed42"][c] = {l: float(f1_score(p.label, p.pred, average="macro")) for l, p in parts}
        S["per_class"][c]["per_lang_neutral_f1"] = {
            l: float(f1_score(p.label, p.pred, labels=["neutral"], average=None, zero_division=0)[0]) for l, p in parts}

    # ---- error analysis for prompting
    z = []
    for l in nd.LANGS:
        p = preds("zeroshot", l)
        z.append({"lang": l, "neutral_share": float((p.pred == "neutral").mean()),
                  "macro_f1": float(f1_score(p.label, p.pred, average="macro")),
                  "accuracy": float(accuracy_score(p.label, p.pred))})
    z = pd.DataFrame(z).set_index("lang")
    S["zeroshot"] = z.round(4).to_dict(orient="index")
    zl = z.loc[nd.LOCAL]
    r1 = st.spearmanr(zl.neutral_share, zl.macro_f1)
    r2 = st.spearmanr(sim.lexical_overlap, zl.neutral_share)
    S["zeroshot_corr"] = {"share_vs_f1": {"rho": float(r1.statistic), "p": float(r1.pvalue)},
                          "overlap_vs_share": {"rho": float(r2.statistic), "p": float(r2.pvalue)}}

    if THREE_SEED_TRANSFER:
        robustness(S)
    (OUT / "stats.json").write_text(json.dumps(S, indent=1, default=float))
    figs(loc, seed, sim, z, S)
    print("wrote", OUT / "stats.json")


def robustness(S):
    """Checks added for the journal version: a multilingual encoder (XLM-R), label
    scoring and calibration of the prompted LLM, and how close translations come to
    the Indonesian originals (chrF, exact matches)."""
    enc = pd.read_csv(SRW / "encoder_runs.csv")
    enc = enc[enc.lang.isin(nd.LOCAL)]
    S["encoders"] = enc.groupby(["encoder", "strategy"]).macro_f1.mean().round(4).unstack().to_dict(orient="index")

    cf = pd.read_csv(SRW / "llm_cf.csv", keep_default_na=False)
    cols = [f"marginal_{l}" for l in LABELS]
    prior = np.log(np.exp(cf[cols].values).mean(0))          # content-free prior, Zhao et al. (2021)
    rules = {"space": [f"space_{l}" for l in LABELS], "cap": [f"cap_{l}" for l in LABELS], "calibrated": cols}
    rows = {}
    for l in nd.LANGS:
        d = pd.read_csv(SRW / "llm_scores" / f"{l}.csv")
        rows[l] = {}
        for rule, c in rules.items():
            s = d[c].values - (prior if rule == "calibrated" else 0)
            p = np.array(LABELS)[s.argmax(1)]
            rows[l][rule] = {"macro_f1": float(f1_score(d.label, p, average="macro")),
                             "neutral_share": float((p == "neutral").mean())}
    assert all(abs(rows[l]["space"]["macro_f1"] - S["zeroshot"][l]["macro_f1"]) < 1e-3 for l in nd.LANGS), \
        "the space rule must reproduce the original prompting scores"
    S["llm_rules"] = rows
    S["llm_rules_mean10"] = {rule: {k: float(np.mean([rows[l][rule][k] for l in nd.LOCAL]))
                                    for k in ["macro_f1", "neutral_share"]} for rule in rules}

    ch = pd.read_csv(SRW / "chrf.csv")
    S["chrf"] = {"raw_mean": float(ch.chrf_raw_vs_ind.mean()), "mt_mean": float(ch.chrf_mt_vs_ind.mean()),
                 "exact_max": float(ch.exact_match.max()),
                 "gain_vs_gap": dict(zip(["rho", "p"], map(float, st.spearmanr(ch.chrf_gain, ch.gap_translate_minus_direct))))}


# --------------------------------------------------------------------------
# Figures: one colour per strategy everywhere (figures.PALETTE, Okabe-Ito)
# --------------------------------------------------------------------------

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def figs(loc, seed, sim, z, S):
    order = sim.sort_values("lexical_overlap", ascending=False).index.tolist()

    # Figure 1: macro-F1 by language, sorted by lexical overlap with Indonesian
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    x = np.arange(len(order)); wd = 0.2
    err = {"direct_bert": seed.direct_bert_sd, "translate_bert": seed.translate_bert_sd}
    if "transfer_bert_sd" in seed:
        err["transfer_bert"] = seed.transfer_bert_sd
    for i, c in enumerate(MAIN):
        y = loc.loc[order, c].values
        e = err[c].reindex(order).values if c in err else None
        ax.bar(x + (i - 1.5) * wd, y, wd, color=F.PALETTE[c], label=NAME[c],
               yerr=e, error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": F.INK})
    for k, l in enumerate(order):
        if np.isnan(loc.loc[l, "translate_bert"]):
            ax.text(x[k] + 0.5 * wd, 0.02, "no MT", rotation=90, ha="center", va="bottom", fontsize=6.5, color="#555")
    ax.axhline(1 / 3, ls=":", lw=0.9, color="#555")
    ax.annotate("chance", (1.005, 1 / 3), xycoords=("axes fraction", "data"),   # outside the plot, clear of the bars
                fontsize=7, color="#555", ha="left", va="center")
    ax.set_xticks(x, [f"{nd.LANGS[l]}\n({sim.loc[l, 'lexical_overlap']:.2f})" for l in order], fontsize=7)
    ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1)
    ax.set_xlabel("Regional language (lexical overlap with Indonesian), most to least similar", fontsize=8)
    ax.legend(ncol=4, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.grid(axis="y", lw=0.4, color="#e5e5e5"); ax.set_axisbelow(True)
    save(fig, "fig1_macro_f1_by_language")

    # Figure 2: row-normalised confusion matrices, pooled over regional languages
    conds = [c for c in MAIN if c in S["confusion"]]
    fig, axes = plt.subplots(1, len(conds), figsize=(2.35 * len(conds), 2.6))
    for ax, c in zip(np.atleast_1d(axes), conds):
        cm = np.array(S["confusion"][c], float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.55 else F.INK)
        ax.set_xticks(range(3), ["neg", "neu", "pos"], fontsize=7.5)
        ax.set_yticks(range(3), ["neg", "neu", "pos"], fontsize=7.5)
        ax.set_xlabel("predicted", fontsize=7.5)
        n = S["per_class"][c]["langs"]
        ax.set_title(f"{NAME[c]}\n({n} languages)", fontsize=8, color=F.PALETTE[c], fontweight="bold")
        for s in ax.spines.values():
            s.set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("true", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig2_confusion_matrices")

    # Figure 3: critical-difference diagram, three strategies over all ten languages
    F.NICE.update({c: NAME[c] for c in MAIN})
    fig = F.fig_cd(loc, ["direct_bert", "transfer_bert", "zeroshot"],
                   "Average ranks over ten regional languages")
    fig = fig if fig is not None else plt.gcf()
    save(fig, "fig3_critical_difference")

    # Figure 4: translate-test minus direct fine-tuning, seed means with seed SD
    both = seed.dropna()
    d = (both.translate_bert_mean - both.direct_bert_mean).sort_values()
    sd = np.sqrt(both.direct_bert_sd ** 2 + both.translate_bert_sd ** 2).reindex(d.index)
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.barh([nd.LANGS[l] for l in d.index], d.values, xerr=sd.values,
            color=F.PALETTE["translate_bert"], edgecolor=F.INK, lw=0.4,
            error_kw={"elinewidth": 0.7, "capsize": 2, "ecolor": F.INK})
    ax.axvline(0, color=F.INK, lw=0.8)
    for k, v in enumerate(d.values):   # values in a column of their own, clear of the error bars
        ax.text(0.098, k, f"{v:+.3f}", va="center", ha="right", fontsize=7.5)
    ax.set_xlabel("macro-F1 difference (translate-test minus direct fine-tuning)", fontsize=8)
    ax.set_xlim(-0.085, 0.1)
    ax.grid(axis="x", lw=0.4, color="#e5e5e5"); ax.set_axisbelow(True)
    save(fig, "fig4_translate_minus_direct")

    # Figure 5: prompting, share of "neutral" answers against macro-F1
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for l, r in z.iterrows():
        regional = l in nd.LOCAL
        ax.scatter(r.neutral_share, r.macro_f1, s=36, zorder=3,
                   color=F.PALETTE["zeroshot"] if regional else "white",
                   edgecolor=F.PALETTE["zeroshot"], lw=1.2)
        off = {"english": (-7, -3), "madurese": (6, 1), "acehnese": (-6, -3),   # Acehnese, Madurese
               "ngaju": (7, -3)}.get(l, (5, 3))                                  # and Ngaju sit close together
        ax.annotate(nd.LANGS[l], (r.neutral_share, r.macro_f1), textcoords="offset points",
                    xytext=off, fontsize=7, ha="right" if off[0] < 0 else "left")
    ax.axvline(0.24, ls="--", lw=0.8, color="#555")
    ax.text(0.245, 0.95, "true share of neutral (24%)", fontsize=7, color="#555", va="top")
    ax.axhline(1 / 3, ls=":", lw=0.9, color="#555")
    ax.text(0.99, 1 / 3 + 0.012, "chance", fontsize=7, color="#555", ha="right")
    ax.set_xlim(0.05, 1.0); ax.set_ylim(0.15, 1.0)   # room for the English label
    ax.set_xlabel('share of test sentences answered "neutral"'); ax.set_ylabel("macro-F1")
    ax.grid(lw=0.4, color="#e5e5e5"); ax.set_axisbelow(True)
    save(fig, "fig5_prompting_abstention")


if __name__ == "__main__":
    main()
