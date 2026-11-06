"""Encoder controls for the journal version: every seed, and a multilingual encoder.

Two gaps found in the original runs:

* Zero-shot transfer used only the seed-42 Indonesian model, although models for
  seeds 7 and 1234 were trained for translate-test anyway. Seed 42 turned out to
  be the strongest of the three, so the single-seed number is optimistic.
* Every strategy used IndoBERT, whose vocabulary is Indonesian-only. XLM-R-base
  is the standard multilingual encoder for zero-shot transfer.

For each encoder and seed, one Indonesian model is trained and scored on the raw
regional test sets (zero-shot transfer) and on their NLLB translations
(translate-test). XLM-R is also fine-tuned on each regional language (direct).
IndoBERT direct runs already exist in outputs/seed_runs.csv.

Rows are appended to outputs/srw/encoder_runs.csv as they finish, and finished
rows are skipped on restart, so an interrupted run resumes.

    cd src && python srw_encoders.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import bert
import nusax_data as nd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "srw" / "encoder_runs.csv"
CACHE = ROOT / "data" / "translated"
SEEDS = [42, 7, 1234]
ENCODERS = {"indobert": bert.INDOBERT, "xlmr": bert.XLMR}
COVERED = ["acehnese", "balinese", "banjarese", "buginese",
           "javanese", "minangkabau", "sundanese"]
RECORDED_S2 = ROOT / "outputs" / "all_scores.csv"   # seed-42 IndoBERT transfer, for the check


def _done() -> set[tuple]:
    if not OUT.exists():
        return set()
    d = pd.read_csv(OUT)
    return set(zip(d.encoder, d.strategy, d.seed, d.lang))


def _score(gold, pred) -> dict:
    return {"macro_f1": f1_score(gold, pred, average="macro"),
            "accuracy": accuracy_score(gold, pred),
            "neutral_share": sum(p == "neutral" for p in pred) / len(pred)}


def _append(row: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    print(f"  {row['encoder']:8s} {row['strategy']:9s} seed {row['seed']:4d} "
          f"{row['lang']:12s} {row['macro_f1']:.4f}", flush=True)


def _check_seed42(rows: list[dict]) -> None:
    """The seed-42 IndoBERT transfer scores must equal the recorded ones,
    otherwise the new seeds are not comparable with the paper's numbers."""
    rec = pd.read_csv(RECORDED_S2)
    rec = rec[rec.condition == "transfer_bert"].set_index("lang").macro_f1
    for r in rows:
        drift = r["macro_f1"] - rec[r["lang"]]
        assert abs(drift) < 1e-3, (r["lang"], r["macro_f1"], rec[r["lang"]])
    print("  seed-42 IndoBERT transfer reproduces the recorded scores", flush=True)


def main() -> int:
    done = _done()
    for enc, name in ENCODERS.items():
        for seed in SEEDS:
            todo_ind = [(enc, s, seed, l) for s, langs in
                        (("transfer", nd.LOCAL + ["english"]), ("translate", COVERED))
                        for l in langs]
            if not all(k in done for k in todo_ind):
                model, tok = bert.finetune(nd.load("indonesian", "train"), name,
                                           epochs=3, seed=seed)
                fresh = []
                for lang in nd.LOCAL + ["english"]:
                    if (enc, "transfer", seed, lang) in done:
                        continue
                    te = nd.load(lang, "test")
                    row = {"encoder": enc, "strategy": "transfer", "seed": seed, "lang": lang,
                           **_score(te.label.tolist(), bert.predict(model, tok, te.text))}
                    _append(row)
                    fresh.append(row)
                if enc == "indobert" and seed == 42 and fresh:
                    _check_seed42([r for r in fresh if r["lang"] in nd.LOCAL])
                for lang in COVERED:
                    if (enc, "translate", seed, lang) in done:
                        continue
                    tx = pd.read_csv(CACHE / f"{lang}.csv")
                    _append({"encoder": enc, "strategy": "translate", "seed": seed, "lang": lang,
                             **_score(tx.label.tolist(), bert.predict(model, tok, tx.text))})
                bert.release(model)

            if enc == "xlmr":
                for lang in nd.LOCAL:
                    if (enc, "direct", seed, lang) in done:
                        continue
                    model, tok = bert.finetune(nd.load(lang, "train"), name, epochs=3, seed=seed)
                    te = nd.load(lang, "test")
                    _append({"encoder": enc, "strategy": "direct", "seed": seed, "lang": lang,
                             **_score(te.label.tolist(), bert.predict(model, tok, te.text))})
                    bert.release(model)

    d = pd.read_csv(OUT)
    print("\nmean macro-F1 over seeds and regional languages\n")
    print(d[d.lang != "english"].groupby(["encoder", "strategy"]).macro_f1
          .agg(["mean", "std", "count"]).round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
