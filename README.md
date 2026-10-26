# PrivacyQA

PrivacyQA synthesizes question-answering (QA) data for multi-constraint recommendation from user preference evidence, such as ratings and review-derived judgments, while protecting the users who contribute that evidence with user-level differential privacy (DP). This repository contains the implementation, the released datasets, and the aggregate results of the paper *User-Level Differentially Private Synthesis of Question-Answering Data for Multi-Constraint Recommendation*.

The framework counts how many users support each request of a fixed request bank, bounds the total contribution of every user, adds discrete Laplace noise to the counts and removes requests whose noisy counts fall below a threshold, and samples QA pairs in proportion to the remaining counts. Complete answer sets come from a fixed catalog, so noise decides only which requests are sampled and how often. One construction template produces three datasets:

| Domain | User evidence | Catalog | Main-bank units / variants | Variants retained by DP |
|---|---|---|---:|---:|
| Movie | MovieLens 32M ratings | 87,585 movies; MovieLens genres and decades, TMDB runtime, language and production country | 1,021 / 3,688 | 3,265 |
| Beauty | Amazon Reviews 2023 Beauty_and_Personal_Care 5-core ratings | 1,028,914 products; six cleaned aspects with 6,264 directions | 1,021 / 3,867 | 1,698 |
| Recipe | Judgments of Food.com reviews from RecipeADQA | 706,350 recipes of RecipeADKG (not distributed) | 1,024 / 4,060 | 1,390 |

Every released dataset is the output of one run of the mechanism with privacy budget ε = 4 and δ = 0, contribution bound B = 20, and quantization Q = 2^20, implemented with OpenDP 0.15.1.

## Repository layout

| Path | Contents |
|---|---|
| `src/privacyqa/` | The Python package; [docs/pipeline.md](docs/pipeline.md) describes its stages and modules |
| `tests/` | Unit tests and a run of the complete pipeline on fictional data |
| `configs/` | The configuration of the reported run, with private values and local paths as placeholders, and the rules of the Movie, Beauty and Recipe catalogs |
| `scripts/` | Model jobs, result tables and data export |
| `data/` | Released datasets, request banks, catalogs and schemas ([data/README.md](data/README.md)) |
| `results/` | Aggregate results behind the tables of the paper ([results/README.md](results/README.md)) |
| `docs/` | [Pipeline](docs/pipeline.md) and [privacy boundary](docs/privacy.md) |
| `third_party/recipeadqa/` | The RecipeADQA constraint search, under the Apache License 2.0 |
| `LICENSES/` | The CC BY 4.0 license of the data and results |

## Installation

The package and its tests were run with Python 3.13 and the versions pinned in `requirements.txt`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
```

Training and evaluating the QA models needs a separate GPU environment with the versions pinned in `requirements-qa.txt` (the reported runs used Python 3.9) and a local copy of `Qwen/Qwen3-1.7B`; the reported runs used revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

## Tests

```bash
.venv/bin/python -m pytest
```

The tests use fictional inputs only and need no data, GPU or network. One of them runs every stage of the pipeline, including the private releases and the attacks, on fictional catalogs, ratings and reviews.

## Released data

[data/README.md](data/README.md) describes every file, its fields and its terms of use. In short:

- **Beauty**: the cleaned product catalog, the request banks, the DP release, and the QA training, validation, and test sets with candidates and answers.
- **Movie**: the schema, the MovieLens-derived part of the catalog (genres and decades), and the question side of the banks, the DP release and the QA splits: questions, constraints, sampling weights and answer counts. Runtime, language and country come from TMDB, whose terms do not allow redistributing its content; these attributes, and the candidate lists and answers that depend on them, are withheld, and the pipeline rebuilds the complete catalog with your own TMDB access ([data/README.md](data/README.md)).
- **Recipe**: the aspect-direction schema and the question side of the banks, the DP release and the QA splits. Recipe identifiers, tags and ingredient lists derive from Recipe1M, which may not be redistributed, so answers and candidate lists are withheld as in the public RecipeADQA release; a Recipe1M licensee can rebuild the catalog from the RecipeADQA resources.

No user history, user identifier, support count, membership list, attack score or output of the ablations without noise is released.

## Running the pipeline

[docs/pipeline.md](docs/pipeline.md) lists the inputs, the stages and their commands. All stages read one configuration file; `configs/paper.json` is the configuration of the reported run, with placeholders for local paths and private values. Running the pipeline requires the source user data, which the `release` stage measures with the mechanism. The model jobs read their records from the `qa-data` stage of a run, not from `data/`. The released Beauty records contain the questions, candidate lists and correct codes, so they can also be used to train and evaluate QA models outside the pipeline; the Movie and Recipe records lack the withheld item-level fields.

## Privacy

[docs/privacy.md](docs/privacy.md) defines the protected unit, the guarantee of each released dataset, the accounting of the reported experiments, and the reasons each artifact is or is not released.

## License

The code is released under the Apache License 2.0 ([LICENSE](LICENSE)). The questions, constraint sets, weights and splits in `data/` and the aggregate results in `results/` are released under CC BY 4.0 ([LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt)), within the terms of the upstream sources stated in [data/README.md](data/README.md). [NOTICE.md](NOTICE.md) lists third-party code and data.

## Citation

Citation information will be added when the paper is published.
