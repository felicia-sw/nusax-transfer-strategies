"""Translate-test support: NusaX languages -> Indonesian via NLLB-200.

NLLB-200 does not cover every language NusaX does. That is not an obstacle to
work around; it is one of the paper's results. A language with no machine
translation path cannot use the translate-test strategy at all, which is a
concrete limitation of a strategy practitioners reach for by default.

Coverage is therefore *measured* against the loaded tokenizer rather than
asserted from memory -- see `coverage()`.
"""
from __future__ import annotations

# Best-guess FLORES-200 codes. Verified at runtime by `coverage()`; anything
# that fails verification is reported as uncovered rather than silently fixed.
CANDIDATE_CODES = {
    "indonesian": "ind_Latn",
    "english": "eng_Latn",
    "acehnese": "ace_Latn",
    "balinese": "ban_Latn",
    "banjarese": "bjn_Latn",
    "buginese": "bug_Latn",
    "javanese": "jav_Latn",
    "madurese": "mad_Latn",
    "minangkabau": "min_Latn",
    "ngaju": "nij_Latn",
    "sundanese": "sun_Latn",
    "toba_batak": "bbc_Latn",
}

MODEL = "facebook/nllb-200-distilled-600M"
TARGET = "ind_Latn"


def _supported(tokenizer) -> set[str]:
    """Language codes the tokenizer actually knows, across transformers versions."""
    for attr in ("additional_special_tokens", "lang_code_to_id"):
        val = getattr(tokenizer, attr, None)
        if val:
            return set(val if isinstance(val, (list, set)) else val.keys())
    vocab = tokenizer.get_vocab()
    return {t for t in vocab if len(t) == 8 and t[3] == "_"}


def coverage(tokenizer=None) -> dict[str, str | None]:
    """lang -> NLLB code, or None where NLLB has no such language."""
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ok = _supported(tokenizer)
    return {lang: (code if code in ok else None)
            for lang, code in CANDIDATE_CODES.items()}


def translate(texts: list[str], src_code: str, batch_size: int = 16) -> list[str]:
    """Translate `texts` from `src_code` into Indonesian."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, src_lang=src_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    bos = tok.convert_tokens_to_ids(TARGET)
    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i:i + batch_size], return_tensors="pt",
                    padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            gen = model.generate(**batch, forced_bos_token_id=bos, max_new_tokens=256)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


if __name__ == "__main__":
    cov = coverage()
    have = [k for k, v in cov.items() if v]
    miss = [k for k, v in cov.items() if not v]
    print(f"NLLB-200 covers {len(have)}/{len(cov)} NusaX languages\n")
    for k, v in cov.items():
        print(f"  {k:14s} {v or '-- NOT IN NLLB --'}")
    if miss:
        print(f"\nNo translate-test path for: {', '.join(miss)}")
        print("Report these as a coverage limitation, not a missing result.")
