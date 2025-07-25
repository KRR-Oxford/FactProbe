# FactProbe

Code repository for the paper: [Supposedly Equivalent Facts That Aren't? Entity Frequency in Pre-training Induces Asymmetry in LLMs](https://arxiv.org/abs/2503.22362) (COLM 2025)

- `factprobe`: main code for probing
- `data_index`: code for Dolma pre-training data indexing
- Download probing datasets from: https://zenodo.org/records/15092789

## Installation

To set up the project, you'll need Poetry (a modern Python package manager). If you don't have Poetry installed, install it first:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then clone and install the project:

```bash
poetry install
```

Make sure you have the necessary GPU drivers and libraries installed (e.g., CUDA).

## Usage

To run the main probing script, use the following command:

```bash
poetry run python probe.py -c path/to/config.yaml
```
