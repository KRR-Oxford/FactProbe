# FactProbe


FactProbe is a framework designed for probing and evaluating various models on knowledge graph triples. It supports multiple configurations and allows users to run experiments with different models and settings.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [License](#license)

## Features

- Support for multiple models (e.g., OLMo, Llama).
- Configurable parameters for running experiments.
- Batch processing of data with GPU support.
- Data integrity validation after processing.

## Installation

To set up the project, clone the repository and install the required dependencies:

```
git clone https://github.com/yourusername/FactProbe.git
cd FactProbe
pip install -r requirements.txt
```

Make sure you have the necessary GPU drivers and libraries installed (e.g., CUDA).

## Usage

To run the main probing script, use the following command:

```
python -m FactProbe.run --config_file path/to/config.yaml
```

You can specify additional options such as `--test` to run in test mode or `--run_all` to ignore high-low count thresholds and process all triples.

## Configuration

The configuration file (`config.yaml`) allows you to set various parameters for your experiments. Here are some key parameters:

- `cuda_devices`: List of available GPU devices.
- `script_path`: Path to the probing script.
- `relations`: List of relations to probe.
- `modes`: Different modes of operation (e.g., `fc`, `qa`).
- `models`: Configuration for different models.
- `test_mode`: Set to `true` to run in test mode with limited samples.
- `run_all`: Set to `true` to ignore high-low count thresholds and process all triples.

### Example Configuration
```
defaults:
  - _self_

cuda_devices: [0, 1, 2, 3]
script_path: /path/to/probe.py
base_data_dir: /path/to/data

relations:
    - P47
    - P50

modes:
  - fc
  - qa

models:
  olmo2_13b:
    name: allenai/OLMo-2-1124-13B-Instruct

test_mode: false
run_all: true  # Set to true to ignore high-low count thresholds and process all triples
```

## File Structure

```
FactProbe/
├── data/
│   └── config.yaml          # Configuration file for experiments
├── scripts/
│   └── run_generation.sh     # Script to run generation tasks
├── probe/
│   ├── __init__.py
│   ├── probe.py              # Main probing logic
│   └── evaluation.py         # Evaluation functions
└── run.py                    # Entry point for running experiments
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
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
