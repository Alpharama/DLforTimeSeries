# Moirai for Time Series Classification

Adapting the [Moirai](https://github.com/SalesforceAIResearch/uni2ts) foundation model (Salesforce AI Research) for time series classification on the [UCR LSST dataset](https://www.timeseriesclassification.com/description.php?Dataset=LSST).

The project systematically explores strategies ranging from a fully frozen encoder to full fine-tuning, including parameter-efficient methods (LoRA/AdaLoRA).

---

## Structure

```
moirai-classification/
├── encoder.py       # MoiraiEncoder: extracts token-level embeddings from Moirai
├── heads.py         # Classification head architectures (pooling, attention, hierarchical)
├── models.py        # Wrappers combining encoder + head for each training strategy
├── utils.py         # Data loading, training loop, grid search, metrics
│
├── 01_basic_pooling.ipynb              # Frozen encoder + classical ML (Ridge, RF)
├── 02_heads_on_frozen_encoder.ipynb    # Frozen encoder + learnable classification heads
├── 03_mask_finetuning_with_heads.ipynb # Fine-tune mask embedding only + head
├── 04_full_finetuning_end_to_end.ipynb # Full encoder fine-tuning + head
├── 05_lora_finetuning.ipynb            # LoRA / AdaLoRA / DoRA fine-tuning + head
├── 06_heatmap_analysis.ipynb           # Results visualization (heatmaps)
├── 07_repeated_comparison.ipynb        # Statistical comparison across best methods
│
├── results_csv/     # Experiment outputs: CSV metrics + heatmap PNGs
├── pyproject.toml   # Dependencies (managed with uv)
└── uv.lock
```

---

## Setup

```bash
uv sync
source .venv/bin/activate
```

---

## Experiments

The notebooks follow a progression from cheapest to most expensive adaptation strategy:

| Notebook | Strategy | Trainable params |
|----------|----------|-----------------|
| `01` | Frozen encoder + classical ML (Ridge, RF) | 0 |
| `02` | Frozen encoder + learnable head | head only |
| `03` | Mask embedding fine-tuning + head | mask + head |
| `04` | Full encoder fine-tuning + head | all |
| `05` | LoRA / AdaLoRA / DoRA + head | low-rank adapters + head |
| `06` | Results analysis | — |
| `07` | Repeated runs & statistical comparison | — |

---

## Dataset

- **UCR LSST** — 6-variate time series, 36 timesteps
- Patch sizes tested: 8, 16, 32, 64

---

## Encoder design

`MoiraiEncoder` reuses Moirai's preprocessing and transformer layers but drops the forecasting head, returning token-level representations instead of a distribution:

```
Input:  (batch, time, variate)
Output: (batch, num_tokens, d_model)   # d_model = 384 for moirai-small
```

The original Moirai forward pass has 6 steps. The encoder keeps steps 1–4 and drops 5–6:

1. Scale observations
2. Project observations → representations
3. Replace prediction window with learnable mask
4. Apply transformer layers
5. ~~Project representations → distribution parameters~~ (dropped)
6. ~~Return distribution~~ (dropped)

---

## References

- **uni2ts** (Salesforce AI Research): https://github.com/SalesforceAIResearch/uni2ts
- **Moirai checkpoint** (Hugging Face): https://huggingface.co/Salesforce/moirai-1.1-R-small
- **Key source files**: `uni2ts/model/moirai/forecast.py`, `uni2ts/model/moirai/module.py`
