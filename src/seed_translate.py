"""Seed-average the translate-test condition too.

direct_bert is now averaged over three seeds while translate_bert rests on one
Indonesian model, which makes the comparison asymmetric in the opposite
direction. The Indonesian model is shared across every target language, so
three seeds of it cost three training runs, not twenty-one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

import bert
import nusax_data as nd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "seed_translate.csv"
CACHE = ROOT / "data" / "translated"
SEEDS = [42, 7, 1234]
LANGS = ["acehnese", "balinese", "banjarese", "buginese",
         "javanese", "minangkabau", "sundanese"]


def main() -> int:
    rows = []
    for seed in SEEDS:
        model, tok = bert.finetune(nd.load("indonesian", "train"),
                                   epochs=3, seed=seed)
        for lang in LANGS:
            tx = pd.read_csv(CACHE / f"{lang}.csv")
            f1 = f1_score(tx.label, bert.predict(model, tok, tx.text),
                          average="macro")
            rows.append({"lang": lang, "seed": seed, "macro_f1": f1})
            print(f"  seed {seed:4d}  {lang:13s} {f1:.4f}", flush=True)
        bert.release(model)

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    print("\nseed-averaged translate_bert\n")
    print(d.groupby("lang").macro_f1.agg(["mean", "std"]).round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
