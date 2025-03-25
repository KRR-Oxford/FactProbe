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

from typing import Callable, Dict, Tuple
import pandas as pd
from .stats import mcnemar_p

# Constants
HIGH_FREQ_THRESHOLD = 100000
DIRECTION_HIGH2LOW = "high2low"
DIRECTION_LOW2HIGH = "low2high"
DIRECTION_HIGH2HIGH = "high2high"
VALID_DIRECTIONS = {DIRECTION_HIGH2LOW, DIRECTION_LOW2HIGH, DIRECTION_HIGH2HIGH}

# Type aliases
FreqDict = Dict[str, int]
TripleKey = Tuple[str, str]


def freq_dict_from_triple_df(triple_df: pd.DataFrame) -> FreqDict:
    """
    Extract a frequency dictionary from a triple DataFrame.

    Instead of loading the whole `entities.json`, this provides relation-specific frequency search
    """
    freq_dict = {}
    for _, row in triple_df.iterrows():
        if row["subject"] not in freq_dict:
            freq_dict[row["subject"]] = row["subject_count"]
        if row["object"] not in freq_dict:
            freq_dict[row["object"]] = row["object_count"]
    return freq_dict


def freq_condition(
    freq_dict: FreqDict, direction: str, low_freq_start: int, low_freq_end: int
) -> Callable[[str, str], bool]:
    """Create a function that checks if subject and object frequencies meet specified conditions.

    Args:
        freq_dict: Dictionary mapping entities to their frequencies
        direction: One of 'high2low', 'low2high', or 'high2high'
        low_freq_start: Lower bound for low frequency range
        low_freq_end: Upper bound for low frequency range

    Returns:
        A function that takes subject and object strings and returns whether they meet frequency conditions

    Raises:
        ValueError: If direction is not one of the valid options
    """

    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"Unknown direction: {direction}. Must be one of {VALID_DIRECTIONS}")

    if direction == DIRECTION_HIGH2LOW:
        return lambda s, o: (freq_dict[s] >= HIGH_FREQ_THRESHOLD and low_freq_start <= freq_dict[o] <= low_freq_end)
    elif direction == DIRECTION_LOW2HIGH:
        return lambda s, o: (low_freq_start <= freq_dict[s] <= low_freq_end and freq_dict[o] >= HIGH_FREQ_THRESHOLD)
    else:  # high2high
        return lambda s, o: (freq_dict[s] >= HIGH_FREQ_THRESHOLD and freq_dict[o] >= HIGH_FREQ_THRESHOLD)


def analyse_results_for_low_freq_range(
    results: Dict[str, Dict[TripleKey, Dict]],
    freq_dict: FreqDict,
    direction: str,
    low_freq_start: int,
    low_freq_end: int,
) -> Dict[str, float | int | str]:
    """Analyze results for a specific frequency range.

    Args:
        results: Dictionary containing forward and backward results
        freq_dict: Dictionary mapping entities to their frequencies
        direction: Direction of analysis ('high2low', 'low2high', or 'high2high')
        low_freq_start: Lower bound for low frequency range
        low_freq_end: Upper bound for low frequency range

    Returns:
        Dictionary containing analysis statistics including:
        - total: Total number of samples
        - forward_acc: Forward accuracy
        - backward_acc: Backward accuracy
        - diff_arrow: Visual indicator of performance difference
        - stat_sig: Statistical significance indicator
    """
    n10 = 0
    n01 = 0
    total = 0
    forward_correct = 0
    backward_correct = 0
    freq_cond = freq_condition(freq_dict, direction, low_freq_start, low_freq_end)

    for s, o in results["forward"].keys():
        if freq_cond(s, o):
            total += 1
            em_forward = int(any(results["forward"][(s, o)]["answer_em"]))
            em_backward = int(any(results["backward"][(s, o)]["answer_em"]))
            forward_correct += em_forward
            backward_correct += em_backward
            if em_forward == 1 and em_backward == 0:
                n10 += 1
            elif em_forward == 0 and em_backward == 1:
                n01 += 1

    if total > 0:
        em_percentage = round((forward_correct - backward_correct) * 100 / total, 2)
    else:
        em_percentage = 0

    # Compute McNemar's test p-value and format significance stars.
    p = mcnemar_p(n10, n01)
    if p < 0.001:
        stat_sig = "***"
    elif p < 0.01:
        stat_sig = "**"
    elif p < 0.05:
        stat_sig = "*"
    else:
        stat_sig = "NS"

    # Define arrow symbols.
    up = "\\textcolor{green}{\\faArrowUp}"
    down = "\\textcolor{red}{\\faArrowDown}"
    equal = "\\textcolor{gray}{=}"

    diff_arrow = up if em_percentage > 0 else equal if em_percentage == 0 else down
    forward_acc = round(forward_correct / total, 3) if total > 0 else 0
    backward_acc = round(backward_correct / total, 3) if total > 0 else 0

    return {
        "total": total,
        "forward_acc": forward_acc,
        "backward_acc": backward_acc,
        "diff_arrow": diff_arrow,
        "stat_sig": stat_sig,
    }


def analyse_results_all_freqs(
    results: Dict[str, Dict[TripleKey, Dict]], freq_dict: FreqDict, direction: str
) -> Dict[str, Dict[str, float | int | str]]:
    """Analyze results across all frequency ranges.

    Args:
        results: Dictionary containing forward and backward results
        freq_dict: Dictionary mapping entities to their frequencies
        direction: Direction of analysis ('high2low', 'low2high', or 'high2high')

    Returns:
        Dictionary mapping frequency ranges to their analysis statistics
    """
    stats = {}

    for ls, le in [(0, 1000), (1000, 10000), (10000, 100000)]:
        k = f"{ls}-{le}".replace("100000", "100K").replace("10000", "10K").replace("1000", "1K")
        stats[k] = analyse_results_for_low_freq_range(results, freq_dict, direction, ls, le)

    return stats
