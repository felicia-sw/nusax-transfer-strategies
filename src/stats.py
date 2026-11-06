"""Friedman + Nemenyi over languages.

Ported from CFES-ICTIIA/src/cfes_pipeline.py:friedman_block. The blocking
variable changes: in CFES each block was a test month, here each block is a
language. Because NusaX is parallel, the twelve languages carry identical
label sequences, so per-language scores are genuinely paired.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

# Studentised range / sqrt(2) at alpha = 0.05, indexed by number of treatments.
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
       6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def friedman_nemenyi(scores: dict[str, list[float]]) -> dict:
    """`scores` maps condition name -> one score per block (language).

    Ranks are computed so that rank 1 is best. Friedman's test asks whether the
    conditions differ at all; Nemenyi's critical difference says which pairs of
    average ranks are far enough apart to call apart.
    """
    names = list(scores)
    k = len(names)
    if k < 2:
        raise ValueError("need at least two conditions")
    M = np.column_stack([scores[n] for n in names])   # blocks x conditions
    n_blocks = M.shape[0]

    if k == 2:
        # Friedman needs three or more treatments; for a straight pair the
        # paired equivalent is Wilcoxon signed-rank over the same blocks.
        stat, p = wilcoxon(M[:, 0], M[:, 1])
        test = "wilcoxon"
    else:
        stat, p = friedmanchisquare(*[M[:, j] for j in range(k)])
        test = "friedman"
    # Negate so higher macro-F1 earns the better (lower) rank.
    avg_rank = rankdata(-M, axis=1).mean(axis=0)
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * n_blocks))

    ranks = dict(zip(names, map(float, avg_rank)))
    pairs = [
        {"a": a, "b": b,
         "rank_diff": round(abs(ranks[a] - ranks[b]), 4),
         "significant": bool(abs(ranks[a] - ranks[b]) > cd)}
        for i, a in enumerate(names) for b in names[i + 1:]
    ]
    return {"test": test, "statistic": float(stat), "p": float(p), "n_blocks": int(n_blocks),
            "k": k, "cd": float(cd), "avg_ranks": ranks, "pairs": pairs}


def selfcheck() -> None:
    """Reproduce the CFES constant: k=5, N=12 must give CD = 1.761."""
    got = Q05[5] * np.sqrt(5 * 6 / (6 * 12))
    assert abs(round(got, 3) - 1.761) < 1e-9, got
    print(f"CD self-check (k=5, N=12): {got:.3f}  matches CFES 1.761")


if __name__ == "__main__":
    selfcheck()
