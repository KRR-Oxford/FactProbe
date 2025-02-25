# Copyright 2025 Zifeng Ding, Yuan He

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
import click
from vllm import LLM, SamplingParams
from collections import defaultdict as ddict
from deeponto.utils import load_file
from yacs.config import CfgNode

@click.command()
@click.option("--config_file", "-c", type=str, help="Path to the configuration file.")
def main(config_file):
    # Load config inside the function
    config = CfgNode(load_file(config_file))

    relation = config.relation
    # model = config.model
    probe_type = f"{relation}_{config.count_low}_{config.count_high}_{config.so_setting}_{config.template_type}"
    em_model = config.eval_model

    with open(f'{config.result_path}/{probe_type}.pkl', 'rb') as file:
        data = pickle.load(file)

    llm = LLM(model=em_model, dtype="half")

    def render(prompt):
        return [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ]

    def generate_prompts(test_dataset, mode):
        forward_prompts, backward_prompts = [], []
        forward_keys, backward_keys = [], []
        forward_data = test_dataset["forward"]
        backward_data = test_dataset["backward"]

        for key, val in forward_data.items():
            s2 = "Yes" if mode == "qa" else "True"
            for gen in val["text"]:
                s1 = gen
                prompt_template = f"""Are 'S1' and 'S2' equivalent? Respond with only one word—either 'Yes' if they are equivalent or 'No' if they are inequivalent. Do not include any additional text or commentary.\nS1: {s1}\nS2: {s2}"""
                forward_prompts.append(render(prompt_template))
                forward_keys.append(key)

        for key, val in backward_data.items():
            s2 = "Yes" if mode == "qa" else "True"
            for gen in val["text"]:
                s1 = gen
                prompt_template = f"""Are 'S1' and 'S2' equivalent? Respond with only one word—either 'Yes' if they are equivalent or 'No' if they are inequivalent. Do not include any additional text or commentary.\nS1: {s1}\nS2: {s2}"""
                backward_prompts.append(render(prompt_template))
                backward_keys.append(key)

        return forward_prompts, backward_prompts, forward_keys, backward_keys

    def save_sem(output, data, keys, mode="forward"):
        for i, answer in enumerate(output):
            if "answer_sem" not in data[mode][keys[i]]:
                data[mode][keys[i]]["answer_sem"] = [answer.outputs[0].text]
            else:
                data[mode][keys[i]]["answer_sem"].append(answer.outputs[0].text)
        return data

    def exact_match(data, mode="forward", method="anyone"):
        correct_cnt = 0
        for key in data[mode]:
            if method == "anyone":
                if "yes" in [string.lower().strip() for string in data[mode][key]["answer_sem"]]:
                    correct_cnt += 1
            elif method == "majority":
                hash_freq = ddict(int)
                for string in data[mode][key]["answer_sem"]:
                    hash_freq[string] += 1
                if "yes" == max(hash_freq, key=hash_freq.get):
                    correct_cnt += 1
        return correct_cnt

    forward_prompts, backward_prompts, forward_keys, backward_keys = generate_prompts(data, config.template_type)

    outputs_forward = llm.chat(forward_prompts, SamplingParams(temperature=0))
    outputs_backward = llm.chat(backward_prompts, SamplingParams(temperature=0))

    data = save_sem(outputs_forward, data, forward_keys, mode="forward")
    data = save_sem(outputs_backward, data, backward_keys, mode="backward")

    forward_cnt = len(forward_keys)
    backward_cnt = len(backward_keys)

    print(f"Forward Soft EM (anyone): {exact_match(data, mode='forward', method='anyone')/forward_cnt}")
    print(f"Backward Soft EM (anyone): {exact_match(data, mode='backward', method='anyone')/backward_cnt}")
    print(f"Forward Soft EM (majority): {exact_match(data, mode='forward', method='majority')/forward_cnt}")
    print(f"Backward Soft EM (majority): {exact_match(data, mode='backward', method='majority')/backward_cnt}")

    # Save the updated result data
    with open(f'{config.result_path}/{probe_type}_softem.pkl', "wb") as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    main()
