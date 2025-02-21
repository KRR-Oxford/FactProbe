# Copyright 2025 Yuan He

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import click
import pandas as pd
from vllm import LLM
from factprobe.probe import FactProbe
from deeponto.utils import save_file, load_file, create_path
from yacs.config import CfgNode


@click.command()
@click.option("--config_file", "-c", type=str, help="Path to the configuration file.")
@click.option(
    "--run_all",
    "-r",
    type=bool,
    help="Whether to ignore the high-low count threshold and run on all triples.",
    default=False,
)
def main(config_file: str, run_all: bool):
    # 0. Load the configuration file
    config = CfgNode(load_file(config_file))
    model_suffix = config.model.split("/")[-1]
    create_path(f"experiments/{config.relation}/{model_suffix}")

    # 1. Load the preprocess data
    df = pd.read_csv(config.dataset)

    # 2. Filter the data according to the setting
    data = (
        {
            "high2low": df[
                (df["subject_count"] >= config.count_high)
                & (df["object_count"] <= config.count_low)
            ],
            "low2high": df[
                (df["subject_count"] <= config.count_low)
                & (df["object_count"] >= config.count_high)
            ],
        }
        if not run_all
        else df
    )

    # 3. Initialise the LLM model
    llm = LLM(model=config.model, dtype="half")

    # 4. Probe the model
    probe = FactProbe(llm=llm, **config)

    # 5. Save the results
    if not run_all:
        for so_setting in data.keys():
            results = probe.probe(data[so_setting], so_setting)
            save_file(
                {"forward": results["forward"], "backward": results["backward"]},
                f"experiments/{config.relation}/{model_suffix}/{config.relation}_{config.count_high}_{config.count_low}_{so_setting}_{config.template_type}.pkl",
            )
    else:
        results = probe.probe(data, "all")
        save_file(
            {"forward": results["forward"], "backward": results["backward"]},
            f"experiments/{config.relation}/{model_suffix}/{config.relation}_{so_setting}_{config.template_type}.pkl",
        )


if __name__ == "__main__":
    main()
