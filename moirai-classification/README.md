# Moirai Encoder (Embeddings)

## Init

```bash
uv sync
```

## Activate env

```bash
source .venv/bin/activate
```

(or just always run with `uv run ...`)

## Quick test run

```bash
uv run main.py
```

Notebook test:

* `moirai.ipynb`

---

## What I do here

Goal: **extract embeddings** (encoder representations) from Moirai, not forecasts.

I implement `MoiraiEncoder` in `encoder.py` using:

* `MoiraiModule`
* (and reuse preprocessing / packing logic from `MoiraiForecaster`)

Source reference:

* Moirai code in `uni2ts`:

  * `src/uni2ts/model/moirai/forecast.py`
  * `src/uni2ts/model/moirai/module.py`

---

## What I changed vs original Moirai forward

Original pipeline:

1. Apply scaling to observations
2. Project from observations to representations
3. Replace prediction window with learnable mask
4. Apply transformer layers
5. Project from representations to distribution parameters
6. Return distribution object

In my encoder version:

* I **keep steps 1–4**
* I **drop steps 5–6**
* I return `reprs` (or pooled embedding) instead of a distribution.

---

## Inputs

Moirai expects **processed inputs**, so I reuse `_convert` from `MoiraiForecaster`.

That means the model takes:

* `past_target`: `(batch, time, variate)`
* `past_observed_target`: `(batch, time, variate)` bool
* `past_is_pad`: `(batch, time)` bool

Example data shape used:

* `(n_individual, time, variate) = (2459, 36, 6)`

---

## Output

Encoder output (`reprs`) is token-level:

* `reprs`: `(n_individual, combine_seq, d_model)`

If you want a single embedding per individual, you need pooling (ex: mean over context tokens):

* `Z`: `(n_individual, d_model)`

---

## Downstream Classification

After extracting token-level embeddings `(n_individual, combine_seq, d_model)`,
we apply **mean pooling over tokens** to obtain a single embedding per time series:

```
(n_individual, d_model)
```
These embeddings are then used as input to a `LogisticRegression` classifier (linear probing setup).

## How to test

* Run notebook: `moirai.ipynb`
* 
---

## Links / References

### Blog + Example Implementation

* Medium article — Zero-shot forecast with Moirai MoE:
  [https://medium.com/data-science-collective/zero-shot-forecast-with-moirai-moe-f81c764bf0e2](https://medium.com/data-science-collective/zero-shot-forecast-with-moirai-moe-f81c764bf0e2)

* GitHub repository from the blog:
  [https://github.com/anamabo/medium-blogs/tree/main/moirai-moe](https://github.com/anamabo/medium-blogs/tree/main/moirai-moe)

---

### Official Repositories / Models

* `uni2ts` (Salesforce AI Research):
  [https://github.com/SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts)

* `uni2ts` examples folder:
  [https://github.com/SalesforceAIResearch/uni2ts/tree/main/example](https://github.com/SalesforceAIResearch/uni2ts/tree/main/example)

* Hugging Face model checkpoint:
  [https://huggingface.co/Salesforce/moirai-1.0-R-large](https://huggingface.co/Salesforce/moirai-1.0-R-large)

---

### Key Source Files (Moirai Implementation)

* Moirai model directory:
  [https://github.com/SalesforceAIResearch/uni2ts/tree/main/src/uni2ts/model/moirai](https://github.com/SalesforceAIResearch/uni2ts/tree/main/src/uni2ts/model/moirai)

* `forecast.py`

* `module.py`

---




