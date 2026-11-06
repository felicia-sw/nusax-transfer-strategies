"""Multi-seed direct_bert on the languages the headline comparison needs.

Only the machine-translation-covered languages matter here: the claim under
test is translate-test versus direct training, and the other three languages
have no translate-test score to compare against. Results are appended per
(language, seed) so an interrupted run resumes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

import bert
import nusax_data as nd

OUT = Path(__file__).resolve().parent.parent / "outputs" / "seed_runs.csv"
SEEDS = [42, 7, 1234]
# MT-covered local languages, from the measured NLLB coverage.
# All ten local languages. The three without MT coverage still need seed
# averaging so the direct_bert column is uniform across the table.
LANGS = list(nd.LOCAL)

# Already measured in the earlier variance check; no need to burn GPU again.
KNOWN = {
    ("javanese", 42): 0.7940, ("javanese", 7): 0.8139, ("javanese", 1234): 0.8070,
    ("buginese", 42): 0.6978, ("buginese", 7): 0.7440, ("buginese", 1234): 0.7281,
}


def done() -> set[tuple[str, int]]:
    if not OUT.exists():
        return set()
    d = pd.read_csv(OUT)
    return {(r.lang, int(r.seed)) for r in d.itertuples()}


def main() -> int:
    have = done()
    rows = []

    # Seed 42 for the remaining languages is already in the results matrix.
    matrix = pd.read_csv(OUT.parent / "macro_f1_matrix.csv", index_col=0)
    for lang in LANGS:
        if (lang, 42) not in have and (lang, 42) not in KNOWN:
            rows.append({"lang": lang, "seed": 42,
                         "macro_f1": float(matrix.loc[lang, "direct_bert"])})
    for (lang, seed), v in KNOWN.items():
        if (lang, seed) not in have:
            rows.append({"lang": lang, "seed": seed, "macro_f1": v})
    if rows:
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        have |= {(r["lang"], r["seed"]) for r in rows}

    todo = [(l, s) for l in LANGS for s in SEEDS if (l, s) not in have]
    print(f"{len(todo)} runs to do\n", flush=True)

    for lang, seed in todo:
        tr, te = nd.load(lang, "train"), nd.load(lang, "test")
        model, tok = bert.finetune(tr, epochs=3, seed=seed)
        pred = bert.predict(model, tok, te.text)
        bert.release(model)
        f1 = f1_score(te.label, pred, average="macro")
        pd.DataFrame([{"lang": lang, "seed": seed, "macro_f1": f1}]).to_csv(
            OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"  {lang:13s} seed {seed:4d}  macro-F1 {f1:.4f}", flush=True)

    d = pd.read_csv(OUT).drop_duplicates(["lang", "seed"], keep="last")
    g = d.groupby("lang").macro_f1.agg(["mean", "std", "count"])
    print("\nseed-averaged direct_bert\n")
    print(g.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
