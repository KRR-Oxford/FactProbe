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

import string
import re


def clean_name(name: str, remove_parenthesis: bool = False):
    # Remove bracketed disambiguation
    if remove_parenthesis:
        name = re.sub(r"\(.*?\)", "", name)
    # Remove underscores
    name = name.replace("_", " ")
    # Strip extra spaces
    name = name.strip()
    return name


def clean_names(names: list[str], remove_parenthesis: bool = False):
    cleaned = []
    seen = set()
    for name in names:
        name = clean_name(name, remove_parenthesis)
        # Filter out empty or very short synonyms if you want:
        if len(name) > 1 and name not in seen:
            cleaned.append(name)
            seen.add(name)
    return cleaned


def is_english_name(name: str) -> bool:
    """
    Return True if the name is considered an English named entity.
    """
    allowed_chars = string.ascii_letters + string.digits + " '-.,()"

    for char in name:
        if char not in allowed_chars:
            return False
    return True


def filter_nonenglish_names(names: list[str]):
    """
    Remove names that are not considered English named entities.
    """
    return [name for name in names if is_english_name(name)]


def remove_lowercased_duplicates(names: list[str]):
    """
    Remove lower-cased names if their upper-cased versions exist in the list.
    """
    uppercased_set = {name.upper() for name in names}  # Collect all names in uppercase form
    return [name for name in names if name != name.lower() or name.upper() not in uppercased_set]
