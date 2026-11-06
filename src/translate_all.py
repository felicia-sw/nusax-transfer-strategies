"""Pre-translate every NLLB-reachable language into Indonesian, once.

Run as its own process before the classification pipeline. Holding IndoBERT and
NLLB-600M on the Metal device at the same time segfaults the allocator, so the
two phases are deliberately kept apart rather than ordered inside one run.
Results are cached per language, so this is resumable: re-running skips
whatever is already on disk.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

import nllb
import nusax_data as nd

CACHE = Path(__file__).resolve().parent.parent / "data" / "translated"


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    cov = nllb.coverage()
    todo = [l for l, c in cov.items() if c and not (CACHE / f"{l}.csv").exists()]

    have = [l for l, c in cov.items() if c]
    print(f"NLLB reaches {len(have)}/{len(cov)} languages; "
          f"{len(todo)} still to translate\n", flush=True)

    for lang in todo:
        te = nd.load(lang, "test")
        t0 = time.time()
        print(f"  {lang:14s} {len(te)} sentences ...", end=" ", flush=True)
        try:
            txt = nllb.translate(te.text.tolist(), cov[lang])
        except Exception as e:                      # noqa: BLE001
            print(f"FAILED ({type(e).__name__}: {str(e)[:80]})", flush=True)
            continue
        pd.DataFrame({"text": txt, "label": te.label.tolist()}).to_csv(
            CACHE / f"{lang}.csv", index=False)
        print(f"done ({time.time() - t0:.0f}s)", flush=True)

    done = sorted(p.stem for p in CACHE.glob("*.csv"))
    print(f"\ncached: {len(done)} languages -> {', '.join(done)}")
    missing = [l for l, c in cov.items() if not c]
    if missing:
        print(f"no NLLB path (excluded from translate-test): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
