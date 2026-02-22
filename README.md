# DLforTimeSeries

This repository presents a comparative study of machine learning, deep learning, and foundation-model-based approaches for time series classification.

## Repository Structure

```
DLforTimeSeries/
│
├── Baseline.ipynb
├── Explo.ipynb
├── moirai-classification/
├── requirements.txt
```

## Description

### Data Exploration

* **Explo.ipynb**
  Exploratory data analysis, preprocessing, and statistical inspection of the dataset.

### Classical Machine Learning and Deep Learning Baselines

* **Baseline.ipynb**
  Implementation and evaluation of traditional models (e.g., Random Forest) and neural architectures for time series classification.

### Foundation Model Approach

* **moirai-classification/**
  Representation learning using the Moirai time-series foundation model (Salesforce AI Research).
  The encoder is used in a frozen setting to extract embeddings, followed by linear probing for classification.

  See `moirai-classification/README.md` for methodological details.

---

## Objective

The objective of this project is to compare:

* Classical machine learning methods
* Deep learning architectures
* Foundation-model-based embeddings

for the task of time series classification.
