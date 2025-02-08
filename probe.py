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
from deeponto.utils import save_file, load_file
from yacs.config import CfgNode


@click.command()
@click.option("--config_file", "-c", type=str, help="Path to the configuration file.")
def main(config_file: str):
    # 0. Load the configuration file
    config = CfgNode(load_file(config_file))

    # 1. Load the preprocess data
    relation = config.relation
    df = pd.read_csv(f"data/cleaned/{relation}_triples.csv")

    # 2. Filter the data according to the setting
    data = {
        "high2low": df[
            (df["subject_count"] >= config.count_high)
            & (df["object_count"] <= config.count_low)
        ],
        "low2high": df[
            (df["subject_count"] <= config.count_low)
            & (df["object_count"] >= config.count_high)
        ],
    }

    # 3. Initialise the LLM model
    llm = LLM(model=config.model, dtype="half")

    # 4. Probe the model
    probe = FactProbe(llm=llm, **config)

    # 5. Save the results
    for so_setting in data.keys():
        results = probe.probe(data[so_setting], so_setting)
        save_file(
            {"forward": results["forward"], "backward": results["backward"]},
            f"experiments/{config.relation}/{config.relation}_{config.count_high}_{config.count_low}_{so_setting}_{config.template_type}.pkl",
        )


if __name__ == "__main__":
    main()
