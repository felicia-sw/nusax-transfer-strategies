"""The bridges being compared.

Each condition is a function (lang) -> (y_true, y_pred) on that language's test
split. The research question is which bridge carries sentiment best into a
low-resource Indonesian language:

  direct      train and test in the target language itself
  transfer    train on Indonesian, apply to the target language untouched
  translate   machine-translate target -> Indonesian, then apply the
              Indonesian model
  zeroshot    prompt an LLM in the target language

Each exists in a classical form (character n-grams + logistic regression, the
cheap floor) and an IndoBERT form (the comparable-to-prior-work version).

Two things are cached on purpose. The Indonesian-trained model is trained once
and reused for every target language, because retraining an identical model
twelve times would only add noise and an hour. Translations are cached to disk
because they are deterministic and slow.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import nusax_data as nd

CACHE = Path(__file__).resolve().parent.parent / "data" / "translated"

_ind_model = None   # (model, tokenizer) trained on Indonesian, lazily built


# ------------------------------------------------------------ classical floor

def _clf():
    """Character n-grams, not words: related Indonesian languages share far more
    sub-word material than vocabulary, so this is the fair classical baseline.

    Deliberately NOT class_weight="balanced". macro-F1 rewards balanced
    predictions, so weighting here and not in the transformer conditions was
    worth about +0.036 macro-F1 to the classical side -- an advantage of the
    setup rather than of the method. Both families now train unweighted, which
    is what makes cross-family numbers comparable.
    """
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000),
    )


def direct(lang: str):
    tr, te = nd.load(lang, "train"), nd.load(lang, "test")
    return te.label.tolist(), _clf().fit(tr.text, tr.label).predict(te.text).tolist()


def transfer(lang: str, source: str = "indonesian"):
    tr, te = nd.load(source, "train"), nd.load(lang, "test")
    return te.label.tolist(), _clf().fit(tr.text, tr.label).predict(te.text).tolist()


# ----------------------------------------------------------------- IndoBERT

def _indonesian_model(epochs: int):
    global _ind_model
    if _ind_model is None:
        import bert
        print("    [training the shared Indonesian IndoBERT once]", flush=True)
        _ind_model = bert.finetune(nd.load("indonesian", "train"), epochs=epochs)
    return _ind_model


def direct_bert(lang: str, epochs: int = 3):
    import bert
    tr, te = nd.load(lang, "train"), nd.load(lang, "test")
    model, tok = bert.finetune(tr, epochs=epochs)
    pred = bert.predict(model, tok, te.text)
    bert.release(model)
    return te.label.tolist(), pred


def transfer_bert(lang: str, epochs: int = 3):
    import bert
    te = nd.load(lang, "test")
    model, tok = _indonesian_model(epochs)
    return te.label.tolist(), bert.predict(model, tok, te.text)


# ---------------------------------------------------------------- translate

def translated_test(lang: str) -> pd.DataFrame | None:
    """Target-language test split rendered into Indonesian, or None when NLLB
    has no such language. Cached: translation is deterministic and slow."""
    import nllb
    code = nllb.CANDIDATE_CODES.get(lang)
    cov = nllb.coverage()
    if not cov.get(lang):
        return None

    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{lang}.csv"
    if dest.exists():
        return pd.read_csv(dest)

    te = nd.load(lang, "test")
    print(f"    [translating {lang} -> Indonesian, {len(te)} sentences]", flush=True)
    out = pd.DataFrame({"text": nllb.translate(te.text.tolist(), code),
                        "label": te.label.tolist()})
    out.to_csv(dest, index=False)
    return out


def translate(lang: str):
    """Classical translate-test."""
    tx = translated_test(lang)
    if tx is None:
        raise LookupError(f"NLLB-200 has no language code for {lang}")
    tr = nd.load("indonesian", "train")
    return tx.label.tolist(), _clf().fit(tr.text, tr.label).predict(tx.text).tolist()


def translate_bert(lang: str, epochs: int = 3):
    import bert
    tx = translated_test(lang)
    if tx is None:
        raise LookupError(f"NLLB-200 has no language code for {lang}")
    model, tok = _indonesian_model(epochs)
    return tx.label.tolist(), bert.predict(model, tok, tx.text)


ZS_CACHE = Path(__file__).resolve().parent.parent / "data" / "zeroshot"


def zeroshot(lang: str):
    """Reads predictions produced by `zeroshot_all.py`, which runs the LLM in
    its own process so it never shares the device with IndoBERT."""
    f = ZS_CACHE / f"{lang}.csv"
    if not f.exists():
        raise LookupError(f"no cached zero-shot predictions for {lang} "
                          "- run: python src/zeroshot_all.py")
    df = pd.read_csv(f)
    return df.label.tolist(), df.pred.tolist()


REGISTRY = {
    "direct": direct, "transfer": transfer, "translate": translate,
    "direct_bert": direct_bert, "transfer_bert": transfer_bert,
    "translate_bert": translate_bert,
    "zeroshot": zeroshot,
}
