# Moirai Classification

**Moirai Classification** is a research library that studies how **pretrained Moirai representations can be adapted to time-series classification tasks**. Instead of using Moirai for forecasting, this project **reuses the encoder representations** produced by the model and trains classification heads on top of them.

The library provides tools to study how different **pooling strategies, patch scales, and fine-tuning regimes** adapt foundation model representations to downstream classification tasks.

Experiments are conducted on the **LSST time-series classification dataset** to evaluate different aggregation heads and transfer-learning strategies.

The implementation is built builds on the Moirai Forecaster from **[uni2ts](https://github.com/SalesforceAIResearch/uni2ts/tree/main)**, (Salesforce AI Research).

This repository was developed as part of the **Deep Learning for Time Series** course at Institut Polytechnique de Paris (M2 Data Science) supervised by Romain Tavenard.

---

## Approach

Moirai is used as a **representation backbone**.

The forecasting decoder is removed and the encoder embeddings are aggregated by a classification head.

```text
Time Series
      │
      ▼
Moirai Encoder
      │
      ▼
Patch Representations
      │
      ▼
Pooling / Attention Head
      │
      ▼
Classifier
```

This framework allows systematic evaluation of:

- pooling strategies
- attention-based aggregation
- parameter-efficient fine-tuning
- multi-scale patch representations

---

## Key Findings

Classification is evaluated on the **LSST dataset** using pretrained Moirai encoder representations. We compare pooling strategies, patch scales, and fine-tuning regimes.

- Mask tokens remain important for downstream performance.
- Mean pooling is a strong baseline for patch aggregation.
- Multi-scale patch representations improve results.
- LoRA fine-tuning outperforms frozen encoders.

---

## Experiments

All experiments used in the project are available in the `experiments/` directory.

The notebooks reproduce the evaluation of different pooling strategies, patch scales, and fine-tuning regimes on the LSST dataset.

See **[experiments/README.md](experiments/README.md)** for details about the experimental setup and notebook structure.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Alpharama/DLforTimeSeries.git
cd DLforTimeSeries

uv sync
uv pip install -e .
source .venv/bin/activate
```

---

## Implemented Components

**Fine-tuning strategies**

* Frozen encoder
* Mask-only tuning
* LoRA adaptation
* Full fine-tuning

**Classification heads**

* Mean pooling
* Attention pooling
* Multi-Head Attention
* Hierarchical attention

**Architectures**

* Single-scale patch models
* Multi-scale hybrid models

---

## Project Structure

```
DLforTimeSeries/
│
├── src/moirai_classification
│   ├── encoder.py
│   ├── heads.py
│   ├── utils.py
│   ├── models/
│   └── trainer/
│
├── experiments/
├── tests/
├── pyproject.toml
└── README.md
```

---

## Acknowledgements

This work builds on the Moirai time-series foundation model released in uni2ts by Salesforce AI Research.
