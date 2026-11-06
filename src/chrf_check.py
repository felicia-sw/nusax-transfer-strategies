"""How much of translate-test is the Indonesian source coming back?

NusaX regional sentences are human translations of Indonesian sentences, and
translate-test machine-translates them back into Indonesian. If NLLB largely
recovers the original wording, translate-test is helped by an artefact of the
corpus rather than by translation as such. The parallel design makes this
measurable: score each NLLB output against the Indonesian original with chrF.

As a reference point, the untranslated regional sentence is scored against the
same original: the gain from the first number to the second is what translation
adds. Both are corpus-level chrF2 (character 6-grams, beta = 2, whitespace
ignored, precision and recall averaged over orders), Popovic (2015), implemented
here so the check needs no extra package.

    cd src && python chrf_check.py   ->  outputs/srw/chrf.csv
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

import nusax_data as nd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "translated"
OUT = ROOT / "outputs" / "srw" / "chrf.csv"
COVERED = ["acehnese", "balinese", "banjarese", "buginese",
           "javanese", "minangkabau", "sundanese"]
N, BETA = 6, 2.0


def _ngrams(s: str, n: int) -> Counter:
    s = "".join(s.split())
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hyps: list[str], refs: list[str]) -> float:
    match = [0] * N
    hyp_tot = [0] * N
    ref_tot = [0] * N
    for h, r in zip(hyps, refs):
        for n in range(1, N + 1):
            hc, rc = _ngrams(h, n), _ngrams(r, n)
            match[n - 1] += sum((hc & rc).values())
            hyp_tot[n - 1] += sum(hc.values())
            ref_tot[n - 1] += sum(rc.values())
    p = sum(m / t for m, t in zip(match, hyp_tot) if t) / N
    r = sum(m / t for m, t in zip(match, ref_tot) if t) / N
    return 0.0 if p + r == 0 else 100 * (1 + BETA**2) * p * r / (BETA**2 * p + r)


def main() -> int:
    ind = nd.load("indonesian", "test").text.tolist()
    seeds = pd.read_csv(ROOT / "outputs" / "seed_summary.csv").set_index("lang")
    rows = []
    for lang in COVERED:
        mt = pd.read_csv(CACHE / f"{lang}.csv").text.tolist()
        raw = nd.load(lang, "test").text.tolist()
        assert len(mt) == len(raw) == len(ind) == 400
        exact = sum(a.strip().lower() == b.strip().lower() for a, b in zip(mt, ind))
        rows.append({"lang": lang,
                     "chrf_raw_vs_ind": chrf(raw, ind),
                     "chrf_mt_vs_ind": chrf(mt, ind),
                     "exact_match": exact / 400,
                     "gap_translate_minus_direct":
                         seeds.loc[lang, "translate_bert_mean"] - seeds.loc[lang, "direct_bert_mean"]})
    d = pd.DataFrame(rows)
    d["chrf_gain"] = d.chrf_mt_vs_ind - d.chrf_raw_vs_ind
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(d.round(3).to_string(index=False))
    rho, p = spearmanr(d.chrf_mt_vs_ind, d.gap_translate_minus_direct)
    print(f"\nSpearman chrF(MT vs Indonesian) ~ translate-minus-direct gap: rho = {rho:.3f}, p = {p:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
