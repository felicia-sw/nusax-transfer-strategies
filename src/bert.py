"""Transformer fine-tuning for the bridge comparison.

A plain training loop rather than `Trainer`: the splits are tiny (500 train
sentences), the loop is easier to report in a methodology section, and it does
not move underneath us across transformers releases.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

INDOBERT = "indobenchmark/indobert-base-p1"
XLMR = "FacebookAI/xlm-roberta-base"

LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _encode(tok, texts, labels, max_len=128):
    enc = tok(list(texts), truncation=True, padding="max_length",
              max_length=max_len, return_tensors="pt")
    y = torch.tensor([L2I[l] for l in labels])
    return TensorDataset(enc["input_ids"], enc["attention_mask"], y)


def finetune(train_df, model_name=INDOBERT, *, epochs=3, lr=2e-5,
             batch_size=16, seed=42, max_len=128, verbose=False):
    """Fine-tune and return (model, tokenizer) so callers can reuse one model
    across many target languages instead of retraining per language."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS)).to(dev)

    tr = DataLoader(_encode(tok, train_df.text, train_df.label, max_len),
                    batch_size=batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    total = epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total, pct_start=0.1)

    model.train()
    for ep in range(epochs):
        run = 0.0
        for ids, mask, y in tr:
            opt.zero_grad()
            out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                        labels=y.to(dev))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += out.loss.item()
        if verbose:
            print(f"    epoch {ep + 1}/{epochs} loss {run / len(tr):.4f}")

    model.eval()
    return model, tok


def predict(model, tok, texts, *, batch_size=32, max_len=128):
    """Label strings for `texts`, scored exactly like the classical conditions."""
    dev = device()
    enc = tok(list(texts), truncation=True, padding="max_length",
              max_length=max_len, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
    preds: list[int] = []
    with torch.no_grad():
        for ids, mask in DataLoader(ds, batch_size=batch_size):
            logits = model(input_ids=ids.to(dev),
                           attention_mask=mask.to(dev)).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
    return [LABELS[i] for i in preds]


def release(model):
    del model
    if device() == "mps":
        torch.mps.empty_cache()


def finetune_predict(train_df, test_df, model_name=INDOBERT, **kw):
    verbose = kw.pop("verbose", False)
    model, tok = finetune(train_df, model_name, verbose=verbose, **kw)
    out = predict(model, tok, test_df.text)
    release(model)
    return out
