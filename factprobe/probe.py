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

import pandas as pd
import itertools
import random
from collections import defaultdict
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from factprobe.prompt import QuestionAnsweringPrompt, FactCheckingPrompt


class FactProbe:
    def __init__(
        self,
        llm: LLM,
        template_type: str,
        template_forward: str,
        template_backward: str,
        relation_forward: str,
        relation_backward: str,
        **kwargs,
    ):
        self.llm = llm
        self.template_type = template_type
        assert self.template_type in ["qa", "fc"], (
            f"Invalid template type: {template_type}"
        )
        self.prompt_cls = {"qa": QuestionAnsweringPrompt, "fc": FactCheckingPrompt}[
            template_type
        ]
        self.prompt_forward = self.prompt_cls(template=template_forward)
        self.prompt_backward = self.prompt_cls(template=template_backward)
        self.relation_forward = relation_forward
        self.relation_backward = relation_backward
        self.correct = {"qa": "yes", "fc": "true"}[self.template_type]

    def probe(self, data: pd.DataFrame, so_setting: str):
        print(f"[{self.template_type}] probing for [{so_setting}]:")
        inputs_forward = []
        inputs_backward = []
        keys = []
        # collect and format inputs
        for _, dp in tqdm(
            data.iterrows(), total=len(data), desc="Preprocessed triples"
        ):
            for s, o in itertools.product(
                eval(dp["subject_name"]), eval(dp["object_name"])
            ):
                keys.append((dp["subject"], dp["object"]))
                inputs_forward.append(
                    self.prompt_forward.render((s, self.relation_forward, o))
                )
                inputs_backward.append(
                    self.prompt_backward.render((s, self.relation_backward, o))
                )
        example_idx = random.randint(0, (len(inputs_forward) - 1))
        print(f"Example forward inputs [{example_idx}]:\n", inputs_forward[example_idx])
        print(f"Example backward inputs [{example_idx}]:\n", inputs_backward[example_idx])

        # compute forward outputs
        outputs_forward = self.llm.chat(
            inputs_forward, SamplingParams(logprobs=10, top_p=0.95)
        )
        results_forward = defaultdict(list)
        for output, k in zip(outputs_forward, keys):
            results_forward[k].append(output.outputs[0].text.lower() == self.correct)
        count_forward = 0
        for _, v in results_forward.items():
            count_forward += int(any(v))

        # compute backwardward outputs
        outputs_backward = self.llm.chat(
            inputs_backward, SamplingParams(logprobs=10, top_p=0.95)
        )
        results_backward = defaultdict(list)
        for output, k in zip(outputs_backward, keys):
            results_backward[k].append(output.outputs[0].text.lower() == self.correct)
        count_backward = 0
        for _, v in results_backward.items():
            count_backward += int(any(v))

        print(f"[{self.template_type}][{so_setting}] {count_forward}-{count_backward} / {len(data)} ({len(inputs_forward)})")

        return {"forward": results_forward, "backward": results_backward}
