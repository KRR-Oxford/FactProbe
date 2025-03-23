# Copyright 2025 Yuan He, Yuqicheng Zhu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.stats import chi2, binom, pearsonr, spearmanr


def mcnemar_p(n_tf: int, n_ft: int, continuity_correction: bool = False):
    r"""Function to compute McNemar's test p-value with continuity correction.

    Args:
        n_tf (_type_): The number of forward correct, backward incorrect triple pairs.
        n_ft (_type_): The number of forward incorrect, backward correct triple pairs.
    Returns:
        p_value: lower p means we have evidence to reject null hypothesis.
    """
    # If there are no discordant pairs, p-value is 1.
    if n_tf + n_ft == 0:
        return 1.0

    n_min, n_max = sorted([n_tf, n_ft])
    corr = int(continuity_correction)
    # We should then use exact binomial test for small n_tf, n_ft
    if n_tf + n_ft < 25:
        n_min, n_max = sorted([n_tf, n_ft])
        p_value = 2 * binom.cdf(n_min, n_min + n_max, 0.5) - binom.pmf(n_min, n_min + n_max, 0.5)
    else:
        # We use McNemar's test for sufficient n_tf + n_ft
        chi2_stat = (abs(n_min - n_max) - corr) ** 2 / (n_min + n_max)
        p_value = chi2.sf(chi2_stat, df=1)

    return p_value


def count_correlation(entities: dict):
    r"""Analyse correlation between normalised `dolma_count` and `wiki_count` for entities."""

    # Extract counts
    dolma_counts = []
    wikidata_counts = []

    for _, entity_data in entities.items():
        dolma_counts.append(float(entity_data["dolma_count"]))
        wikidata_counts.append(float(entity_data["wiki_count"]))

    # Convert to numpy arrays
    dolma_counts = np.array(dolma_counts)
    wikidata_counts = np.array(wikidata_counts)

    # Normalise the counts using log transformation to handle skewed distributions
    dolma_counts_norm = np.log1p(dolma_counts)  # log1p to handle zeros
    wikidata_counts_norm = np.log1p(wikidata_counts)

    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(dolma_counts_norm, wikidata_counts_norm)
    spearman_corr, spearman_p = spearmanr(dolma_counts_norm, wikidata_counts_norm)

    return {
        "pearson": {"correlation": pearson_corr, "p_value": pearson_p},
        "spearman": {"correlation": spearman_corr, "p_value": spearman_p},
        "sample_size": len(dolma_counts),
    }
