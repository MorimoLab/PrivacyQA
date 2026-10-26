# Third-party notice

The original RecipeADQA software is licensed under Apache License 2.0. Public
datasets and aggregate research artifacts are licensed under CC BY 4.0. See
`LICENSE` for the exact scope.

The RDF data model in
`code/04_graph_enrichment/recipe_handler/data.py` and parts of the graph
enrichment implementation are copied from or derived from the FoodKG project:

https://github.com/foodkg/foodkg.github.io/

Those files are distributed under the Apache License 2.0. The complete license
text is provided in `LICENSES/Apache-2.0.txt`.

Recipe1M is an external restricted dataset and is not included in this
release. References to Recipe1M in source code describe an expected external
input interface and do not embed or redistribute dataset records. Recipe1M
and non-released Recipe1M-derived record-level artifacts are not covered by
the RecipeADQA licenses, and this repository grants no rights in them.

The query-aspect pipeline was developed using the ORCAS click dataset released
with MS MARCO for non-commercial research. ORCAS query text, QIDs, clicked URLs
or slugs, model batches, responses, and query-level derivatives are not
redistributed here. The release contains only an aggregate four-stage aspect
inventory that does not expose query-level records. ORCAS and MS MARCO remain
subject to their upstream terms:

https://microsoft.github.io/msmarco/

The released model artifacts under `checkpoints/` are fine-tuned from the
following third-party base models, whose upstream terms continue to apply:

- Qwen/Qwen3-1.7B, Apache License 2.0:
  https://huggingface.co/Qwen/Qwen3-1.7B
- microsoft/deberta-v3-base, MIT License:
  https://huggingface.co/microsoft/deberta-v3-base

The Qwen base-model weight files themselves are not duplicated. The DeBERTa
fine-tuned checkpoint includes parameters initialized from the upstream MIT
model and must retain that upstream notice in addition to the RecipeADQA
research-artifact terms.
