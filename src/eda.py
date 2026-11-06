"""Corpus statistics for the exploratory analysis.

These describe the data, not the results: vocabulary overlap and character
n-gram similarity are properties of the corpus and are computable before any
model is trained. Whether they explain the results is tested separately, in the
discussion, and is reported as post-hoc analysis rather than prediction.
"""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import nusax_data as nd


def all_text(lang: str) -> pd.Series:
    return pd.concat([nd.load(lang, s) for s in nd.SPLITS], ignore_index=True).text


def vocabulary(lang: str) -> set[str]:
    return {w for t in all_text(lang).str.lower() for w in t.split()}


def label_distribution() -> pd.DataFrame:
    rows = {}
    for lang in nd.LANGS:
        lab = pd.concat([nd.load(lang, s) for s in nd.SPLITS], ignore_index=True).label
        rows[nd.LANGS[lang]] = lab.value_counts(normalize=True)
    return pd.DataFrame(rows).T[["negative", "neutral", "positive"]]


def length_stats() -> pd.DataFrame:
    rows = []
    for lang in nd.LANGS:
        t = all_text(lang)
        words = t.str.split().str.len()
        rows.append({
            "language": nd.LANGS[lang],
            "mean_words": words.mean(),
            "median_words": words.median(),
            "mean_chars": t.str.len().mean(),
            "vocabulary": len(vocabulary(lang)),
            "type_token_ratio": len(vocabulary(lang)) / words.sum(),
        })
    return pd.DataFrame(rows).set_index("language")


def similarity_to_indonesian() -> pd.DataFrame:
    """Two independent measures of how close each language is to Indonesian.

    `lexical_overlap` is the share of a language's word types that also occur in
    Indonesian. `trigram_similarity` is cosine similarity over character
    trigrams -- the same signal Darnoto and Firmawan (2026) use to pick a pivot
    language, recomputed here so the two can be compared.
    """
    ind_vocab = vocabulary("indonesian")
    corpora = {l: " ".join(all_text(l)) for l in nd.LANGS}
    order = list(nd.LANGS)
    M = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3)).fit_transform(
        [corpora[l] for l in order])
    sims = (M @ M[order.index("indonesian")].T).toarray().ravel()

    rows = []
    for lang in nd.LOCAL:
        v = vocabulary(lang)
        rows.append({
            "lang": lang,
            "language": nd.LANGS[lang],
            "lexical_overlap": len(v & ind_vocab) / len(v),
            "trigram_similarity": float(sims[order.index(lang)]),
        })
    return pd.DataFrame(rows).set_index("lang").sort_values(
        "lexical_overlap", ascending=False)


if __name__ == "__main__":
    print(label_distribution().round(3).to_string(), "\n")
    print(length_stats().round(2).to_string(), "\n")
    print(similarity_to_indonesian().round(4).to_string())
