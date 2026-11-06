"""Run the bridge comparison across the NusaX languages."""
from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import nusax_data as nd
from conditions import REGISTRY
from stats import friedman_nemenyi

OUT = Path(__file__).resolve().parent.parent / "outputs"


def score(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}


def run(conditions: list[str], langs: list[str], epochs: int) -> dict:
    rows, skipped = [], []
    for name in conditions:
        fn = REGISTRY[name]
        takes_epochs = "epochs" in inspect.signature(fn).parameters
        for lang in langs:
            t0 = time.time()
            try:
                y, p = fn(lang, epochs=epochs) if takes_epochs else fn(lang)
            except LookupError as e:
                # A bridge that cannot reach a language is a result, not a crash.
                skipped.append({"condition": name, "lang": lang, "reason": str(e)})
                print(f"  {name:15s} {nd.LANGS[lang]:12s} SKIPPED - {e}", flush=True)
                continue
            s = score(y, p)
            rows.append({"condition": name, "lang": lang, "name": nd.LANGS[lang], **s})
            print(f"  {name:15s} {nd.LANGS[lang]:12s} macro-F1 {s['macro_f1']:.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    wide = df.pivot(index="lang", columns="condition", values="macro_f1").reindex(langs)

    # Blocked tests need every condition present in every block, so the test
    # runs on complete cases only. Which languages that drops is itself
    # reportable -- see `skipped`.
    complete = wide.dropna()
    stats = None
    if len(conditions) > 1 and len(complete) >= 3:
        stats = friedman_nemenyi({c: complete[c].tolist() for c in conditions})
        stats["languages_used"] = complete.index.tolist()
        stats["languages_dropped"] = [l for l in wide.index if l not in complete.index]

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "per_language_scores.csv", index=False)
    wide.to_csv(OUT / "macro_f1_matrix.csv")
    (OUT / "results.json").write_text(json.dumps(
        {"scores": rows, "skipped": skipped, "stats": stats,
         "epochs": epochs}, indent=2))
    return {"table": wide, "stats": stats, "skipped": skipped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+",
                    default=["direct_bert", "transfer_bert", "translate_bert"],
                    choices=list(REGISTRY))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--locals-only", action="store_true",
                    help="drop Indonesian and English (the high-resource controls)")
    a = ap.parse_args()

    langs = nd.LOCAL if a.locals_only else list(nd.LANGS)
    print(f"conditions: {', '.join(a.conditions)}")
    print(f"languages : {len(langs)}   epochs: {a.epochs}\n")
    res = run(a.conditions, langs, a.epochs)

    print("\nmacro-F1 by language\n")
    print(res["table"].round(4).to_string())
    if res["skipped"]:
        print(f"\nskipped {len(res['skipped'])} language/condition pairs "
              "(no machine-translation path)")
    if res["stats"]:
        s = res["stats"]
        print(f"\n{s['test']}: statistic={s['statistic']:.4f}  p={s['p']:.6f}  "
              f"N={s['n_blocks']} languages  CD={s['cd']:.4f}")
        if s.get("languages_dropped"):
            print(f"dropped from the test: {', '.join(s['languages_dropped'])}")
        print("average ranks:", {k: round(v, 3) for k, v in s["avg_ranks"].items()})
        for pr in s["pairs"]:
            mark = "significant" if pr["significant"] else "not significant"
            print(f"  {pr['a']} vs {pr['b']}: rank diff {pr['rank_diff']:.3f} -> {mark}")


if __name__ == "__main__":
    main()
