# Glucose CGM Prediction with Homomorphic Encryption

A proof-of-concept comparing three CGM (Continuous Glucose Monitoring) prediction models implemented in plaintext and under Fully Homomorphic Encryption (FHE), developed as a Bachelor's thesis at Universitat de Lleida (UdL).

FHE allows running machine learning inference directly on encrypted data: the server never sees the patient's glucose readings in plaintext. This repository evaluates the accuracy/performance trade-off of two FHE backends ([Concrete ML](https://github.com/zama-ai/concrete-ml) and [HEIR](https://github.com/google/heir)) across three model families.

## Models

| Family | Cleartext | Concrete ML | HEIR |
|---|---|---|---|
| Naive linear regressor | ✓ | ✓ | ✓ |
| MLP (3-layer) | ✓ | ✓ | ~ |
| Decision tree | ✓ | ✓ | — |

All models predict the next CGM value (5 minutes ahead) from a sliding window of the last 4 readings (20 minutes of history).

## Dataset

The project uses the **Shanghai T2DM** dataset (Zhu, 2022):

> Zhu, Jinhao (2022). *Diabetes Datasets — ShanghaiT1DM and ShanghaiT2DM*. figshare.
> https://doi.org/10.6084/m9.figshare.20444397.v3

Download the dataset and place all Excel files from `Shanghai_T2DM/` into:

```
ml_impl/dataset/shanghai_t2dm/
```

## Setup

**Requirements:** Python 3.12, [uv](https://docs.astral.sh/uv/)

```bash
cd ml_impl
uv sync
```

## Usage

### Dataset exploration

Generates statistical summaries and plots of the Shanghai T2DM dataset into `plots/dataset/`.

```bash
uv run python -m explore
```

### Benchmark suite

Evaluates all model variants and generates comparison plots into `plots/benchmark/`.

```bash
# Quick smoke test (few FHE samples, skip HEIR)
uv run python -m benchmarks --fhe-samples 5

# Full benchmark
uv run python -m benchmarks --fhe-samples 100
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--fhe-samples N` | 100 | Samples to run through full FHE execution (slow) |

### Docker

Build the image (uses CPU PyTorch, no ROCm):

```bash
docker build -f Dockerfile -t glucose-benchmark .
```

Run (mount the dataset directory):

```bash
docker run \
  -v /path/to/Shanghai_T2DM:/app/dataset/shanghai_t2dm \
  glucose-benchmark --fhe-samples 50
```

## Project structure

```
ml_impl/
├── src/
│   ├── datasets/
│   │   └── shanghai.py          # Dataset loading and preprocessing
│   └── models/
│       ├── model.py             # Abstract CGMModel base class
│       ├── naive_regressor/     # Cleartext / Concrete / HEIR
│       ├── nn/                  # Cleartext / Concrete MLP
│       └── tree/                # Cleartext / Concrete decision tree
├── explore/                     # Dataset exploration module
├── benchmarks/                  # Benchmark runner and plots
├── plots/                       # Generated output (gitignored)
└── Dockerfile
```

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE).
