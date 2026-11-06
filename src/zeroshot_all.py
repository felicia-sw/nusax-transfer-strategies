"""Run LLM zero-shot over every NusaX language, in its own process.

Separate process for the same reason translation is: two large models resident
on the Metal device at once segfaults the allocator. Predictions are cached per
language, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

import llm_zeroshot as lz
import nusax_data as nd

CACHE = Path(__file__).resolve().parent.parent / "data" / "zeroshot"


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [l for l in nd.LANGS if not (CACHE / f"{l}.csv").exists()]
    print(f"model: {lz.MODEL}\n{len(todo)} languages to run\n", flush=True)

    for lang in todo:
        te = nd.load(lang, "test")
        t0 = time.time()
        print(f"  {lang:14s} {len(te)} sentences ...", end=" ", flush=True)
        try:
            pred = lz.classify(te.text.tolist(), batch_size=8)
        except Exception as e:                       # noqa: BLE001
            print(f"FAILED ({type(e).__name__}: {str(e)[:80]})", flush=True)
            continue
        pd.DataFrame({"pred": pred, "label": te.label.tolist()}).to_csv(
            CACHE / f"{lang}.csv", index=False)
        print(f"done ({time.time() - t0:.0f}s)", flush=True)

    done = sorted(p.stem for p in CACHE.glob("*.csv"))
    print(f"\ncached: {len(done)}/{len(nd.LANGS)} -> {', '.join(done)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
