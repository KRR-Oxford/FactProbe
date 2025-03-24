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
from .stats import mcnemar_p


def freq_dict_from_triple_df(triple_df: pd.DataFrame) -> dict[str, int]:
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


def freq_condition(freq_dict: dict, direction: str, low_freq_start: int, low_freq_end: int):
    """
    Return a function that checks if a subject and object are in the given frequency range.
    """
    if direction == "high2low":
        return lambda s, o: (freq_dict[s] >= 100000 and freq_dict[o] <= low_freq_end and freq_dict[o] >= low_freq_start)
    elif direction == "low2high":
        return lambda s, o: (freq_dict[s] <= low_freq_end and freq_dict[s] >= low_freq_start and freq_dict[o] >= 100000)
    elif direction == "high2high":
        return lambda s, o: (freq_dict[s] >= 100000 and freq_dict[o] >= 100000)
    else:
        raise ValueError("Unknown direction: " + direction)


def analyse_results_for_low_freq_range(
    results: dict, freq_dict: dict, direction: str, low_freq_start: int, low_freq_end: int
):
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


def analyse_results_all_freqs(results: dict, freq_dict: dict, direction: str):
    stats = {}

    for ls, le in [(0, 1000), (1000, 10000), (10000, 100000)]:
        k = f"{ls}-{le}".replace("100000", "100K").replace("10000", "10K").replace("1000", "1K")
        stats[k] = analyse_results_for_low_freq_range(results, freq_dict, direction, ls, le)

    return stats


def results_to_latex(
    results_question: dict, results_statement: dict, freq_dict: dict, direction: str, relation_name: str
):
    results_question = analyse_results_all_freqs(results_question, freq_dict, direction)
    results_statement = analyse_results_all_freqs(results_statement, freq_dict, direction)

    if direction != "high2high":
        print("\\multirow{3}{*}{" + "\\textsf{" + relation_name + "}" + "}")
    else:
        print("\\multirow{1}{*}{" + "\\textsf{" + relation_name + "}" + "}")
        
    for k in results_question:
        forward_acc_question = results_question[k]["forward_acc"]
        backward_acc_question = results_question[k]["backward_acc"]
        forward_acc_statement = results_statement[k]["forward_acc"]
        backward_acc_statement = results_statement[k]["backward_acc"]
        total_question = results_question[k]["total"]
        total_statement = results_statement[k]["total"]
        assert total_question == total_statement
        print(
            "&",
            k,
            "&",
            results_question[k]["total"],
            "&",
            f"{forward_acc_question:.3f}",
            "&",
            f"{backward_acc_question:.3f}",
            "&",
            results_question[k]["diff_arrow"],
            "&",
            results_question[k]["stat_sig"],
            "&",
            f"{forward_acc_statement:.3f}",
            "&",
            f"{backward_acc_statement:.3f}",
            "&",
            results_statement[k]["diff_arrow"],
            "&",
            results_statement[k]["stat_sig"],
            "\\\\",
        )
