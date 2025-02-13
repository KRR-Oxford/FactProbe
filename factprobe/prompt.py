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

from pydantic import BaseModel, field_validator
from typing import Tuple


class QuestionPrompt(BaseModel):
    instruction: str = "Please evaluate the statement or claim contained in the question. Respond with only one word—either 'Yes' if the claim is correct or 'No' if it is incorrect. Do not include any additional text or commentary."
    template: str

    @field_validator("template")
    def validate_template(cls, template: str):
        placeholders = {"{subject}", "{predicate}", "{object}", "?"}
        missing_placeholders = placeholders - set(template.split())
        if not all(ph in template for ph in placeholders):
            raise ValueError(
                f"Triple template must include {placeholders}. Missing: {missing_placeholders}"
            )
        return template

    def render(self, triplet: Tuple[str, str, str]) -> str:
        system_input = {"role": "system", "content": self.instruction}
        s, r, o = triplet
        user_input = {
            "role": "user",
            "content": self.template.format(subject=s, predicate=r, object=o),
        }
        return [system_input, user_input]


class StatementPrompt(BaseModel):
    instruction: str = "Please evaluate the statement or claim. Respond with only one word—either 'True' if the claim is correct or 'False' if it is incorrect. Do not include any additional text or commentary."
    template: str

    @field_validator("template")
    def validate_template(cls, template: str):
        placeholders = {"{subject}", "{predicate}", "{object}", "."}
        missing_placeholders = placeholders - set(template.split())
        if not all(ph in template for ph in placeholders):
            raise ValueError(
                f"Triple template must include {placeholders}. Missing: {missing_placeholders}"
            )
        return template

    def render(self, triplet: Tuple[str, str, str]) -> str:
        system_input = {"role": "system", "content": self.instruction}
        s, r, o = triplet
        user_input = {
            "role": "user",
            "content": self.template.format(subject=s, predicate=r, object=o),
        }
        return [system_input, user_input]
