import pandas as pd
import itertools
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from factprobe.prompt import QuestionAnsweringPrompt, FactCheckingPrompt
from collections import defaultdict
from deeponto.utils import save_file, load_file
from yacs.config import CfgNode
import click


@click.command()
@click.option("--config_file", "-c", type=str, help="Path to the configuration file.")
def main(config_file: str):
    # 0. Load the configuration file
    config = CfgNode(load_file(config_file))

    # 1. Load the preprocess data
    relation = config.relation
    df = pd.read_csv(f"data/cleaned/{relation}_triples.csv")

    # 2. Filter the data according to the setting
    setting = config.setting
    data = {
        "high2low": df[
            (df["subject_count"] >= config.count_high)
            & (df["object_count"] <= config.count_low)
        ],
        "low2high": df[
            (df["subject_count"] <= config.count_low)
            & (df["object_count"] >= config.count_high)
        ],
    }[setting]

    # 3. Initialise the LLM model
    llm = LLM(model=config.model, dtype="half")

    # 4. Set up the prompt
    template_type = config.template_type
    prompt_cls = {"qa": QuestionAnsweringPrompt, "fc": FactCheckingPrompt}[
        template_type
    ]
    forward_prompt = prompt_cls(
        template=getattr(config, f"{template_type}_template_forward")
    )
    backward_prompt = prompt_cls(
        template=getattr(config, f"{template_type}_template_backward")
    )
    
    # 5. Probing the model
    print(f"[{template_type}] probing for [{setting}]:")
    inputs_forward = []
    inputs_backward = []
    keys = []
    for _, dp in tqdm(data.iterrows()):
        for s, o in itertools.product(
            eval(dp["subject_name"]), eval(dp["object_name"])
        ):
            keys.append((dp["subject"], dp["object"]))
            inputs_forward.append(
                forward_prompt.render(
                    (s, getattr(config, f"{template_type}_relation_forward"), o)
                )
            )
            inputs_backward.append(
                backward_prompt.render(
                    (s, getattr(config, f"{template_type}_relation_backward"), o)
                )
            )
    print("Example forward inputs:\n", inputs_forward[0])
    print("Example backward inputs:", inputs_backward[0])

    outputs_forward = llm.chat(
        inputs_forward, SamplingParams(logprobs=10, top_p=0.95)
    )
    result_dict_forward = defaultdict(list)
    for output, k in zip(outputs_forward, keys):
        result_dict_forward[k].append(output.outputs[0].text.lower() == "yes")
    count_forward = 0
    for _, v in result_dict_forward.items():
        count_forward += int(any(v))

    outputs_backward = llm.chat(
        inputs_backward, SamplingParams(logprobs=10, top_p=0.95)
    )
    result_dict_backward = defaultdict(list)
    for output, k in zip(outputs_backward, keys):
        result_dict_backward[k].append(output.outputs[0].text.lower() == "yes")
    count_backward = 0
    for _, v in result_dict_backward.items():
        count_backward += int(any(v))

    print(f"{count_forward}-{count_backward} / {len(data)} ({len(inputs_forward)})")

    save_file(
        {"forward": result_dict_forward, "backward": result_dict_backward},
        f"experiments/{relation}/{relation}_{config.count_high}_{config.count_low}_{setting}_{template_type}.pkl",
    )


if __name__ == "__main__":
    main()
