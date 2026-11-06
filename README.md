# Four transfer strategies for sentiment analysis in Indonesian regional languages

Code, cached translations, predictions and scores for the article *Cross-Lingual Sentiment
Analysis for Indonesian Regional Languages: A Controlled Comparison of Four Transfer
Strategies* (submitted to Media Jurnal Informatika).

The study compares four ways to build a sentiment classifier for the ten regional languages of
[NusaX-Senti](https://github.com/IndoNLP/nusax), under one protocol:

| Strategy | What it does |
|---|---|
| S1 Direct fine-tuning | Fine-tune IndoBERT-base-p1 on the 500 training sentences of the target language |
| S2 Zero-shot transfer | Fine-tune IndoBERT on Indonesian and apply it unchanged |
| S3 Translate-test | Translate the test sentences into Indonesian with NLLB-200-distilled-600M, then classify with the S2 model |
| S4 LLM prompting | Ask Qwen2.5-3B-Instruct for the label and read its log-probabilities for the three label words |

## Main results

Mean macro-F1 over the ten regional languages (S1 to S3: mean of seeds 42, 7 and 1234):

| Strategy | Languages | Mean macro-F1 [95% CI] |
|---|---|---|
| Direct fine-tuning | 10 | 0.763 [0.738, 0.789] |
| Translate-test | 7 (covered by NLLB) | 0.771 [0.731, 0.811] |
| Zero-shot transfer | 10 | 0.581 [0.486, 0.677] |
| LLM prompting | 10 | 0.456 [0.354, 0.557] |

- Direct fine-tuning ranked first, zero-shot transfer second and prompting third on every language (Friedman χ²(2) = 20.00, p < .001).
- Translate-test did not differ significantly from direct fine-tuning on the seven covered languages (Wilcoxon p = .688). Madurese, Ngaju and Toba Batak have no NLLB translation path.
- The prompted LLM answered "neutral" for 44–88% of regional-language sentences. Calibration removed the neutral answers but did not raise macro-F1.

All numbers in the article are in [`results/stats.json`](results/stats.json).

## Reproduce the analysis

The cached outputs in `outputs/` and `data/` are enough to recompute every table, test and
figure in about a minute, without a GPU:

```bash
pip install -r requirements.txt
cd src && python analysis.py ../results --three-seed-transfer
```

This rewrites `results/stats.json` and the figures; the file should be byte-identical to the
committed one.

## Re-run the experiments

These need PyTorch and Transformers and download the models from the Hugging Face Hub. Each
script caches its outputs and resumes after an interruption.

| Script | Produces |
|---|---|
| `src/nusax_data.py` | Downloads the official NusaX-Senti splits into `data/` |
| `src/translate_all.py` | NLLB translations of the regional test sets (`data/translated/`) |
| `src/pipeline.py`, `src/conditions.py` | Seed-42 runs of every strategy and the TF-IDF baseline (`outputs/all_scores.csv`, `outputs/macro_f1_matrix.csv`) |
| `src/seed_runs.py`, `src/seed_translate.py` | Direct fine-tuning and translate-test with seeds 42, 7 and 1234 |
| `src/srw_encoders.py` | Zero-shot transfer and translate-test with all three seeds, and the XLM-R check (`outputs/srw/encoder_runs.csv`) |
| `src/zeroshot_all.py`, `src/llm_zeroshot.py` | LLM prompting (`data/zeroshot/`) |
| `src/llm_rescore.py` | Label-scoring variants and calibration of the LLM (`outputs/srw/llm_scores/`) |
| `src/chrf_check.py` | chrF and exact matches of the translations against the Indonesian originals |
| `src/save_predictions.py` | Per-sentence predictions (`outputs/predictions/`) |
| `src/verify.py` | Integrity checks on the data and the recorded scores |

The S4 prompt, used for every language:

```text
Classify the sentiment of the following sentence as negative, neutral, or positive. Answer with one word only.

Sentence: <sentence>
Sentiment:
```

All runs used Python 3.13.5 on an Apple M3 laptop (16 GB) through the PyTorch MPS backend.

## Data

`data/` holds copies of the official NusaX-Senti splits, which were checked to be
byte-identical to commit `fa65c41` of [IndoNLP/nusax](https://github.com/IndoNLP/nusax). NusaX is
distributed under its authors' licence; please cite Winata et al. (2023) when using it.
