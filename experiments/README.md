# Experiments - Moirai Classification

This directory contains the **experimental notebooks used to evaluate Moirai representations for time-series classification** on the **UCR LSST dataset**.

The experiments progressively explore different ways to adapt the **pretrained Moirai encoder** for classification, ranging from frozen feature extraction to full fine-tuning.

---

# Notebook Overview

The experiments follow a **progressive adaptation pipeline**, starting from simple baselines and gradually increasing model complexity.

| Notebook                              | Description                                             |
| ------------------------------------- | ------------------------------------------------------- |
| `data_explo.ipynb`                    | Exploration of the LSST dataset                         |
| `00_baseline.ipynb`                   | Classical baselines on raw data                         |
| `01_basic_pooling.ipynb`              | Frozen encoder + classical ML on pooled representations |
| `02_heads_on_frozen_encoder.ipynb`    | Frozen encoder + learnable classification heads         |
| `03_mask_finetuning_with_heads.ipynb` | Fine-tuning mask embeddings only                        |
| `04_full_finetuning_end_to_end.ipynb` | End-to-end fine-tuning of encoder + head                |
| `05_lora_finetuning.ipynb`            | Parameter-efficient fine-tuning (LoRA / AdaLoRA / DoRA) |
| `06_heatmap_analysis.ipynb`           | Visualization and comparison of results                 |
| `07_repeated_comparison.ipynb`        | Repeated runs for statistical comparison                |
| `moirai_encoder.ipynb`                | Inspection of Moirai encoder representations            |

---

# Experiment Progression

The experiments gradually increase the **number of trainable parameters**.

| Stage                | Strategy                                              |
| -------------------- | ----------------------------------------------------- |
| Baselines            | Classical ML models on raw time series                |
| Feature extraction   | Frozen Moirai encoder                                 |
| Head training        | Train classification heads on encoder representations |
| Partial adaptation   | Fine-tune mask embeddings                             |
| Full adaptation      | End-to-end encoder fine-tuning                        |
| Efficient adaptation | LoRA / AdaLoRA / DoRA                                 |

This setup allows systematic evaluation of:

* pooling strategies
* attention-based aggregation heads
* patch resolutions
* fine-tuning regimes

---

# Dataset

Experiments are conducted on **UCR LSST**:

* 6 variables (photometric bands)
* 36 time steps
* 14 classes

Patch sizes evaluated:

```
8 / 16 / 32 / 64
```

---

# Results

All experiment outputs are saved in:

```
results_csv/
```

This directory contains:

* experiment metrics (CSV)
* aggregated results
* heatmap visualizations

---

# Reference

The experiments build on the **Moirai encoder** from
**[uni2ts](https://github.com/SalesforceAIResearch/uni2ts)** (Salesforce AI Research).

---

