# FactProbe

## Data Processing Note

- Download `wikidata5m` from the official website, transform all the data files into `json`.
- Compute `wikidata_freq` for each entity by calling wikidata API (approx two weeks on 5m entities).
- Remove 9 entities that do not have wikidata frequencies (likely to be deprecated).
- Compute 121 triples, relations w.r.t. wikidata5m (algorithm: for each `(s, r)`, `o` is unique and for each `(r, o)`, `s` is unique.); rank and select relations according to their frequencies in these triples.
- 


## Related Work

- [Large Language Models Struggle to Learn Long-Tail Knowledge](https://arxiv.org/abs/2211.08411) (ICML 2023).
- [The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"](https://arxiv.org/abs/2309.12288) (ICLR 2024).
- [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0.pdf) (Nature 2024).

## Follow Up of Reversal Curse

- [Reverse Training to Nurse the Reversal Curse](https://arxiv.org/pdf/2403.13799) (Meta)
- [Mitigating Reversal Curse in Large Language Models via Semantic-aware Permutation Training](https://arxiv.org/pdf/2403.00758) (Microsoft Research)
- [Physics of Language Models: Part 3.2, Knowledge Manipulation](https://arxiv.org/pdf/2309.14402) (Meta)
