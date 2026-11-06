"""Re-run the seed-42 IndoBERT conditions and keep every prediction.

The original runs stored scores only. Per-class F1 and confusion matrices need
the predicted labels, so this repeats the seed-42 runs and writes them to
outputs/predictions/. Each run's macro-F1 is printed next to the score recorded
earlier for the same seed, so any drift (the MPS backend is not guaranteed to be
bit-for-bit deterministic) is visible rather than silently absorbed.

    python src/save_predictions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

import bert
import conditions as cond
import nusax_data as nd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "predictions"
SEED = 42


def save(name: str, lang: str, y, p, recorded: float | None) -> None:
    pd.DataFrame({"label": y, "pred": p}).to_csv(OUT / f"{name}_{lang}.csv", index=False)
    f1 = f1_score(y, p, average="macro")
    ref = "" if recorded is None else f"   recorded {recorded:.4f}   drift {f1 - recorded:+.4f}"
    print(f"  {name:15s} {lang:12s} macro-F1 {f1:.4f}{ref}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = pd.read_csv(ROOT / "outputs" / "seed_runs.csv")
    seeds_tx = pd.read_csv(ROOT / "outputs" / "seed_translate.csv")
    matrix = pd.read_csv(ROOT / "outputs" / "macro_f1_matrix.csv", index_col=0)

    def rec(df, lang):
        r = df[(df.lang == lang) & (df.seed == SEED)]
        return float(r.macro_f1.iloc[0]) if len(r) else None

    # classical baselines: deterministic, CPU
    for lang in nd.LOCAL:
        y, p = cond.direct(lang)
        save("direct", lang, y, p, matrix.loc[lang, "direct"])

    # shared Indonesian IndoBERT -> transfer and translate-test
    ind, tok = bert.finetune(nd.load("indonesian", "train"), seed=SEED)
    for lang in nd.LOCAL:
        te = nd.load(lang, "test")
        save("transfer_bert", lang, te.label.tolist(), bert.predict(ind, tok, te.text),
             matrix.loc[lang, "transfer_bert"])
        tx = cond.translated_test(lang) if (cond.CACHE / f"{lang}.csv").exists() else None
        if tx is not None:
            save("translate_bert", lang, tx.label.tolist(), bert.predict(ind, tok, tx.text),
                 rec(seeds_tx, lang))
    bert.release(ind)

    # direct fine-tuning, one model per language
    for lang in nd.LOCAL:
        tr, te = nd.load(lang, "train"), nd.load(lang, "test")
        model, tok = bert.finetune(tr, seed=SEED)
        save("direct_bert", lang, te.label.tolist(), bert.predict(model, tok, te.text),
             rec(seeds, lang))
        bert.release(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
