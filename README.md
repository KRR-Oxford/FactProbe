# FactProbe

Code repository for the paper: [Supposedly Equivalent Facts That Aren't? Entity Frequency in Pre-training Induces Asymmetry in LLMs](link_to_be_given)


## Installation

To set up the project, you'll need Poetry (a modern Python package manager). If you don't have Poetry installed, install it first:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then clone and install the project:

```bash
git clone https://github.com/KRR-Oxford/FactProbe.git
cd FactProbe
poetry install
```

Make sure you have the necessary GPU drivers and libraries installed (e.g., CUDA).

## Usage

To run the main probing script, use the following command:

```bash
python -m probe.py -c path/to/config.yaml
```


## Project Note

See Notion.

## Related Work

- [Large Language Models Struggle to Learn Long-Tail Knowledge](https://arxiv.org/abs/2211.08411) (ICML 2023).
- [The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"](https://arxiv.org/abs/2309.12288) (ICLR 2024).
- [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0.pdf) (Nature 2024).

## Follow Up of Reversal Curse

- [Reverse Training to Nurse the Reversal Curse](https://arxiv.org/pdf/2403.13799) (Meta)
- [Mitigating Reversal Curse in Large Language Models via Semantic-aware Permutation Training](https://arxiv.org/pdf/2403.00758) (Microsoft Research)
- [Physics of Language Models: Part 3.2, Knowledge Manipulation](https://arxiv.org/pdf/2309.14402) (Meta)
