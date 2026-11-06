"""LLM prompting as the fourth bridge.

Two deliberate choices, both about making the comparison fair and repeatable.

*An open model, run locally.* A closed API model cannot be re-run years later:
versions are retired and silently updated, so a reported number stops being
reproducible. Anyone with the repo can reproduce this one.

*Scoring, not generation.* The model gets one forward pass and we read the
logits for the three label words. There is no sampling, so no seed-dependence
and no parsing of free text that may not contain a label at all. It is also
about three times cheaper than generating.

The prompt is byte-identical for every language -- only the sentence changes.
Varying the instruction per language would reintroduce exactly the confound the
parallel corpus exists to remove.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
LABELS = ["negative", "neutral", "positive"]

PROMPT = (
    "Classify the sentiment of the following sentence as negative, neutral, "
    "or positive. Answer with one word only.\n\nSentence: {text}\nSentiment:"
)

_cached: tuple | None = None


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load():
    global _cached
    if _cached is None:
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16).to(_device()).eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"        # so the last position is the real one
        _cached = (model, tok, _label_ids(tok))
    return _cached


def _label_ids(tok) -> list[int]:
    """First-token id of each label, with a leading space as the model would
    continue after 'Sentiment:'. Asserts the three are distinguishable."""
    ids = [tok.encode(" " + l, add_special_tokens=False)[0] for l in LABELS]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"labels share a first token: {list(zip(LABELS, ids))}")
    return ids


def classify(texts: list[str], batch_size: int = 8) -> list[str]:
    model, tok, ids = load()
    dev = _device()
    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        chunk = [
            tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(text=t)}],
                tokenize=False, add_generation_prompt=True)
            for t in texts[i:i + batch_size]
        ]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=384).to(dev)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]      # next-token distribution
        pick = logits[:, ids].argmax(-1).cpu().tolist()
        out.extend(LABELS[p] for p in pick)
    return out


if __name__ == "__main__":
    import sys
    import time

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import nusax_data as nd
    from sklearn.metrics import f1_score

    lang = sys.argv[1] if len(sys.argv) > 1 else "javanese"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    te = nd.load(lang, "test").head(n)
    t0 = time.time()
    pred = classify(te.text.tolist())
    dt = time.time() - t0
    print(f"{lang}: macro-F1 {f1_score(te.label, pred, average='macro'):.4f} "
          f"on {n} sentences ({dt:.0f}s, {dt / n:.2f}s each)")
    print("predicted distribution:",
          {l: pred.count(l) for l in LABELS})
