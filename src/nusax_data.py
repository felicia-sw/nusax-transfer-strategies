"""NusaX-Senti loader.

Pulls the sentiment splits straight from the IndoNLP/nusax repo as CSV, so the
pipeline does not depend on the `datasets` package. 12 languages, 1,000
sentences each, fully parallel: every language is a human translation of the
same Indonesian source sentences.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

RAW = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment"

# NusaX language codes -> human names. `ind` is the source language; `eng` is
# the other high-resource control. The remaining ten are the local languages.
# Repo directory name -> display name. `indonesian` is the source language and
# `english` the other high-resource control; the remaining ten are the locals.
LANGS = {
    "indonesian": "Indonesian",
    "english": "English",
    "acehnese": "Acehnese",
    "balinese": "Balinese",
    "banjarese": "Banjarese",
    "buginese": "Buginese",
    "javanese": "Javanese",
    "madurese": "Madurese",
    "minangkabau": "Minangkabau",
    "ngaju": "Ngaju",
    "sundanese": "Sundanese",
    "toba_batak": "Toba Batak",
}

LOCAL = [c for c in LANGS if c not in ("indonesian", "english")]
SPLITS = ("train", "valid", "test")

DATA = Path(__file__).resolve().parent.parent / "data"


def _fetch(lang: str, split: str) -> pd.DataFrame:
    dest = DATA / lang / f"{split}.csv"
    if dest.exists():
        return pd.read_csv(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW}/{lang}/{split}.csv"
    with urllib.request.urlopen(url, timeout=60) as r:
        body = r.read()
    df = pd.read_csv(io.BytesIO(body))
    df.to_csv(dest, index=False)
    return df


def load(lang: str, split: str) -> pd.DataFrame:
    """One split for one language, columns: text, label."""
    df = _fetch(lang, split)
    cols = {c.lower(): c for c in df.columns}
    return df.rename(columns={cols["text"]: "text", cols["label"]: "label"})[["text", "label"]]


def load_all(split: str = "test") -> dict[str, pd.DataFrame]:
    return {lang: load(lang, split) for lang in LANGS}


def check_parallel() -> pd.DataFrame:
    """Sanity check: parallel means identical row counts and identical labels
    across languages. If this fails, the paired statistics are not valid."""
    rows = []
    ref = None
    for lang in LANGS:
        parts = {s: load(lang, s) for s in SPLITS}
        labels = pd.concat([parts[s]["label"] for s in SPLITS], ignore_index=True)
        if ref is None:
            ref = labels
        rows.append({
            "lang": lang,
            "name": LANGS[lang],
            **{f"n_{s}": len(parts[s]) for s in SPLITS},
            "labels_match_ind": bool(labels.equals(ref)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tbl = check_parallel()
    print(tbl.to_string(index=False))
    print()
    print("label distribution (ind, all splits):")
    print(pd.concat([load("indonesian", s) for s in SPLITS])["label"].value_counts())
