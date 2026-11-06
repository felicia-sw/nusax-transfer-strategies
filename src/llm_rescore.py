"""Re-score LLM prompting under several label-scoring rules, plus calibration.

The original run read the first-token logit of " negative" / " neutral" / " positive"
(with a leading space) right after the chat template's assistant header, where
the model would more naturally start with "Negative" or "negative". Reviewers
asked whether the collapse onto "neutral" is a property of the model or of that
scoring rule. One forward pass per sentence gives every rule at once:

  space        " negative"  (the original rule)
  bare         "negative"
  cap          "Negative"
  marginal     log-sum-exp over all four spellings (space/bare x lower/Cap)
  calibrated   marginal, divided by the model's prior on content-free inputs
               ("N/A", "", "[MASK]"), following Zhao et al. (2021)

Writes outputs/srw/llm_scores/<lang>.csv (one row per test sentence, the three
label log-probabilities under each rule) and outputs/srw/llm_cf.csv (the
content-free priors). Finished languages are skipped on restart.

    cd src && python llm_rescore.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

import llm_zeroshot as lz
import nusax_data as nd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "srw" / "llm_scores"
CF_OUT = ROOT / "outputs" / "srw" / "llm_cf.csv"
SPELLINGS = {"space": " {}", "bare": "{}", "cap": "{}", "cap_space": " {}"}
CONTENT_FREE = ["N/A", "", "[MASK]"]


def label_ids(tok) -> dict[str, list[int]]:
    """First-token id per spelling, in LABELS order; each spelling must keep the
    three labels distinguishable."""
    ids = {}
    for name, fmt in SPELLINGS.items():
        words = [l.capitalize() if name.startswith("cap") else l for l in lz.LABELS]
        ids[name] = [tok.encode(fmt.format(w), add_special_tokens=False)[0] for w in words]
        assert len(set(ids[name])) == 3, (name, ids[name])
    return ids


def logprobs(texts: list[str], batch_size: int = 8) -> pd.DataFrame:
    model, tok, _ = lz.load()
    ids = label_ids(tok)
    rows = []
    for i in range(0, len(texts), batch_size):
        chunk = [tok.apply_chat_template(
                     [{"role": "user", "content": lz.PROMPT.format(text=t)}],
                     tokenize=False, add_generation_prompt=True)
                 for t in texts[i:i + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=384).to(lz._device())
        with torch.no_grad():
            lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), -1).cpu()
        for r in lp:
            row = {}
            for name, tids in ids.items():
                for lab, t in zip(lz.LABELS, tids):
                    row[f"{name}_{lab}"] = r[t].item()
            for k, lab in enumerate(lz.LABELS):
                uniq = sorted({ids[n][k] for n in ids})      # dedupe shared tokens
                row[f"marginal_{lab}"] = torch.logsumexp(r[uniq], 0).item()
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok, original = lz.load()
    assert label_ids(tok)["space"] == original, "space rule must equal the original rule"

    if not CF_OUT.exists():
        cf = logprobs(CONTENT_FREE)
        cf.insert(0, "input", CONTENT_FREE)
        cf.to_csv(CF_OUT, index=False)
        print("content-free priors written", flush=True)

    for lang in nd.LOCAL + ["indonesian", "english"]:
        dest = OUT / f"{lang}.csv"
        if dest.exists():
            continue
        te = nd.load(lang, "test")
        lp = logprobs(te.text.tolist())
        lp.insert(0, "label", te.label.tolist())
        lp.to_csv(dest, index=False)
        print(f"  {lang:12s} done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
