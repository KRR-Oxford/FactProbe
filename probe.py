# Copyright 2025 Yuan He, Bailan He

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import click
import pandas as pd
from typing import Optional
from textwrap import dedent
from yacs.config import CfgNode
from deeponto.utils import save_file, load_file, create_path
from vllm import LLM, SamplingParams
from factprobe.probe import FactProbe

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--config_file", "-c", type=str, required=True, help="Path to the configuration file.")
@click.option("--model", "-m", type=str, default=None, help="Name of the model to use (overrides `config.model`).")
@click.option("--run_all", is_flag=True, help="Run on all triples, ignoring count thresholds.")
@click.option("--run_test", is_flag=True, help="Run in test mode with only 100 samples.")
def main(config_file: str, model: Optional[str], run_all: bool, run_test: bool):
    """Main function to execute the inference pipeline."""

    # Display command-line arguments
    command_msg = f"""
        config_file: {config_file}\n
        model: {model}\n
        run_all: {run_all}\n
        run_test: {run_test}\n
    """
    logger.info(dedent(command_msg))

    # 1. Load the configuration file
    config = CfgNode(load_file(config_file))
    if model:
        config.model = model

    # 2. Load and preprocess the dataset
    df = pd.read_csv(config.dataset, nrows=100 if run_test else None)

    if not run_all:
        data_dict = {
            "high2low": df[(df["subject_count"] >= config.count_high) & (df["object_count"] <= config.count_low)],
            "low2high": df[(df["subject_count"] <= config.count_low) & (df["object_count"] >= config.count_high)],
        }
    else:
        data_dict = {"all": df}

    # 3. Initialize the model and probe
    llm = LLM(model=config.model)
    probe = FactProbe(llm=llm, **config)
    sampling_params = SamplingParams(logprobs=10, top_p=0.95)  # temperature=0.0 means greedy decoding

    # 4. Run inference with batched data
    def batch_iter(df: pd.DataFrame, batch_size: int):
        """Yields batches of a DataFrame."""
        for start in range(0, len(df), batch_size):
            yield df.iloc[start : start + batch_size]

    # Set up the output directory
    base_path = os.path.join("experiments", config.relation, config.model)
    create_path(base_path)

    for freq_setting, data in data_dict.items():
        logger.info(f"Running inference: relation={config.relation}, type={config.template_type}, freq={freq_setting}")

        # Construct file name
        file_name = f"{config.relation}_{freq_setting}_{config.template_type}.pkl"
        if not run_all:
            file_name = f"{config.relation}_h={config.count_high}_l={config.count_low}_{freq_setting}_{config.template_type}.pkl"
        file_path = os.path.join(base_path, file_name)

        # Load existing results if available
        results = {"forward": {}, "backward": {}}
        if os.path.exists(file_path):
            results = load_file(file_path)

        for batch in batch_iter(data, config.batch_size):
            batch_keys = set(map(tuple, batch[["subject", "object"]].values.tolist()))

            # Skip batch if all pairs already computed
            if results["forward"] and batch_keys <= set(results["forward"].keys()):
                continue

            # Run inference and update results
            batch_results = probe.probe(batch, sampling_params)
            results["forward"].update(batch_results["forward"])
            results["backward"].update(batch_results["backward"])
            save_file(results, file_path)  # Save intermediate results

        # Save final results
        save_file(results, file_path)
        logger.info(f"Results saved: {file_path}")


if __name__ == "__main__":
    main()
