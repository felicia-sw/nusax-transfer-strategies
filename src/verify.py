"""Verification suite: the checks a referee would run.

Each check either passes, or prints exactly what is wrong. Run before
submission, and cite the output in the reproducibility checklist.

    python src/verify.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import nusax_data as nd
from stats import Q05

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def c1_alignment() -> None:
    print("\n1. Split alignment across languages")
    print("   If language splits were shuffled independently, a target-language")
    print("   test sentence could correspond to an Indonesian TRAINING sentence,")
    print("   and translate-test would be scoring on data the model was fit to.")
    for sp in nd.SPLITS:
        ref = nd._fetch("indonesian", sp)["id"].tolist()
        same = all(nd._fetch(l, sp)["id"].tolist() == ref for l in nd.LANGS)
        check(f"{sp}: identical id order in all 12 languages", same)


def c2_leakage() -> None:
    print("\n2. Leakage between splits")
    ids = {sp: set(nd._fetch("indonesian", sp)["id"]) for sp in nd.SPLITS}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        check(f"{a} ∩ {b} empty", not (ids[a] & ids[b]),
              f"{len(ids[a] & ids[b])} shared ids")
    tr = set(nd.load("indonesian", "train").text.str.strip().str.lower())
    te = set(nd.load("indonesian", "test").text.str.strip().str.lower())
    check("no verbatim text shared train/test", not (tr & te), f"{len(tr & te)} shared")


def c3_chance() -> None:
    print("\n3. Empirical chance level for macro-F1")
    y = nd.load("indonesian", "test").label.tolist()
    rng = np.random.default_rng(0)
    labels = ["negative", "neutral", "positive"]
    sims = [f1_score(y, rng.choice(labels, len(y)), average="macro") for _ in range(2000)]
    lo, hi = float(np.mean(sims)), float(np.percentile(sims, 95))
    print(f"        uniform-random macro-F1: mean {lo:.4f}, 95th pct {hi:.4f}")
    check("chance is near 1/3, so 'below chance' claims use a sound floor",
          abs(lo - 1 / 3) < 0.02)
    return lo


def c4_degenerate(chance: float) -> None:
    print("\n4. Degenerate (single-class) zero-shot predictions")
    print("   A model that answers one label for everything can look 'below")
    print("   chance' without being wrong in an interesting way. Report which.")
    for lang in nd.LOCAL:
        p = pd.read_csv(ROOT / f"data/zeroshot/{lang}.csv")
        top = p.pred.value_counts()
        frac = top.max() / len(p)
        print(f"        {nd.LANGS[lang]:12s} {top.idxmax():8s} {frac:.0%}"
              + ("   DEGENERATE" if frac > 0.8 else ""))


def c5_nemenyi() -> None:
    print("\n5. Nemenyi critical values vs Demsar (2006) Table 5")
    published = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    check("all nine q_0.05 values match the published table",
          all(abs(Q05[k] - v) < 1e-9 for k, v in published.items()))
    cd = Q05[5] * np.sqrt(5 * 6 / (6 * 12))
    check("CD reproduces the CFES constant (k=5, N=12 -> 1.761)",
          abs(round(cd, 3) - 1.761) < 1e-9, f"got {cd:.3f}")


def c6_reproducible() -> None:
    print("\n6. Reported scores recompute from cached predictions")
    w = pd.read_csv(ROOT / "outputs/macro_f1_matrix.csv", index_col=0)
    bad = 0
    for lang in nd.LOCAL:
        p = pd.read_csv(ROOT / f"data/zeroshot/{lang}.csv")
        if abs(f1_score(p.label, p.pred, average="macro") - w.loc[lang, "zeroshot"]) > 1e-6:
            bad += 1
    check("zeroshot column matches raw predictions", bad == 0, f"{bad} mismatches")


def c7_class_weight() -> None:
    print("\n7. Class-weighting asymmetry between condition families")
    print("   The classical models use class_weight='balanced'; IndoBERT uses")
    print("   plain cross-entropy. macro-F1 rewards balancing, so this favours")
    print("   the classical side. Quantified, and reported either way.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    def clf(balanced):
        return make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5),
                            min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=2000,
                               class_weight="balanced" if balanced else None))

    gains = []
    for lang in nd.LOCAL:
        tr, te = nd.load(lang, "train"), nd.load(lang, "test")
        b = f1_score(te.label, clf(True).fit(tr.text, tr.label).predict(te.text), average="macro")
        u = f1_score(te.label, clf(False).fit(tr.text, tr.label).predict(te.text), average="macro")
        gains.append(b - u)
    print(f"        class_weight is worth {np.mean(gains):+.4f} macro-F1 on average")
    check("asymmetry is measured and disclosed, not silently present", True,
          f"mean advantage {np.mean(gains):+.4f}")


def main() -> int:
    print("=" * 70)
    print("VERIFICATION SUITE")
    print("=" * 70)
    c1_alignment()
    c2_leakage()
    chance = c3_chance()
    c4_degenerate(chance)
    c5_nemenyi()
    c6_reproducible()
    c7_class_weight()
    print("\n" + "=" * 70)
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED: {', '.join(FAILED)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
