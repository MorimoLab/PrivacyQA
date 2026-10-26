# Data

All files come from the reported run. `scripts/export_release.py` writes them from that run with the field whitelists described below.

## Layout

```
data/
├── beauty/
│   ├── catalog/   catalog.parquet, schema.json
│   ├── banks/     main.json, random.json
│   └── release/   dp_release.jsonl, train.jsonl, dev.jsonl, test.jsonl
├── movie/
│   ├── catalog/   catalog_movielens.parquet, schema.json
│   ├── banks/     main.questions.json, random.questions.json
│   ├── release/   dp_release.questions.jsonl, train.questions.jsonl, dev.questions.jsonl, test.questions.jsonl
│   └── LICENSE.txt
└── recipe/
    ├── schema/    schema.json
    ├── banks/     main.questions.json, random.questions.json
    └── release/   dp_release.questions.jsonl, train.questions.jsonl, dev.questions.jsonl, test.questions.jsonl
```

| Domain | `dp_release` | `train` | `dev` | `test` |
|---|---:|---:|---:|---:|
| Movie | 3,265 | 2,758 | 40 | 400 |
| Beauty | 1,698 | 1,463 | 40 | 400 |
| Recipe | 1,390 | 1,158 | 40 | 400 |

Files with `.questions` in their names keep the question side of each record and omit item-level fields; the reasons are given under [Terms of use](#terms-of-use).

## Records

**`dp_release`**: one record per request variant retained by the target release of the full procedure (DP) on the main bank, before training holdout. This is the released dataset of the paper.

| Field | Meaning |
|---|---|
| `id` | Variant identifier |
| `constraints` | All constraints, as `aspect::direction` strings |
| `evidence_constraints` | The preference constraint core, whose user support the release measures |
| `catalog_constraints` | Auxiliary constraints added from the catalog; they carry no user support |
| `answer_count` | Size of the complete answer set |
| `weight` | Sampling weight from the released noisy mass of the variant's unit, split equally over the unit's variants |
| `schema_version`, `kb_version` | Versions of the schema and catalog |
| `answers` | Complete answer set (Beauty only): product `parent_asin` identifiers |

**`train`**: the training records of the DP training source after training holdout, which removes requests that share a constraint set with a validation or test request and requests that share the complete answers of a validation request or of half of the test requests. Fields: `qa_key`, `constraint_id` (the `id` in `dp_release`), `question` (training template), `evidence_constraints`, `catalog_constraints`, `answer_count`, and `sampling_weight`, the release weight renormalized over the training records. Beauty records add `template_id`, `candidates` (30 items with a code and an attribute description), `selected_candidate_codes` (the correct codes), and `answers`.

**`dev`, `test`**: the validation set of 40 requests and the test set of 400 requests, drawn from the main bank independently of user support; the validation set is meant for model selection, and the reported models, trained with fixed settings, do not use it. Fields: `qa_key`, `question` (validation or test template), `evidence_constraints`, `catalog_constraints`, `answer_count`, and `evaluation_track`, which is `answer_class_holdout` when training holdout also removes training requests with the same complete answers (a test request whose answer set takes an odd position in a fixed pseudorandom order of the test answer sets) and `constraint_signature_holdout` otherwise. Beauty records add `template_id`, `candidates`, `selected_candidate_codes`, `candidate_item_ids`, `complete_answer_ids`, `schema_version`, and `kb_version`. Every candidate list contains all correct answers and distractors that violate at least one constraint.

**`banks`**: the main and random request banks, built from the catalogs before any user data are read. `cores` lists the preference constraint cores with their catalog match counts and the identifiers of their variants, which share one support count; `candidates` lists the variants with their complete answers; `rules` and `coverage` record the construction settings and outcomes. Question-side banks replace each variant's `answers` by `answer_count`.

**`catalog`, `schema`**: `schema.json` gives the versions and separates the directions that can form preference cores (`evidence_atoms`) from those that can be constraints (`catalog_atoms`). The Beauty catalog maps each of its 1,028,914 products (`parent_asin`) to its cleaned attributes in six aspects, with 6,264 directions; the brand aspect is not part of it. `python -m privacyqa build-beauty-catalog` rebuilds it from the Amazon Reviews 2023 product metadata with the rules in `configs/beauty_rules.json`. The Movie catalog lists, for each of the 87,585 MovieLens movies (`movieId`), its genre and decade directions. The Recipe schema covers 437 directions, the union of 393 review directions and 379 recipe tags; the 335 directions in both can form preference cores.

## Terms of use

| Files | Source | Terms |
|---|---|---|
| `movie/` | MovieLens 32M (GroupLens) | Redistributed under the MovieLens usage license, reproduced in `movie/LICENSE.txt`: research use only, no commercial use without permission from GroupLens, acknowledgement of MovieLens in publications, and no claim of endorsement. Any redistribution must keep these conditions. |
| `beauty/` | Amazon Reviews 2023 (McAuley Lab) | The source data are distributed for research without a stated license; product identifiers and attribute values remain subject to the terms of the source. Cite Hou et al., *Bridging Language and Items for Retrieval and Recommendation* (2024). |
| `recipe/` | RecipeADQA schema and review judgments of Food.com reviews | The schema derives from [RecipeADQA](https://github.com/MorimoLab/RecipeADQA), released under CC BY 4.0; credit RecipeADQA when reusing it. |
| Questions, constraint sets, weights and splits of all domains | This work | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([license text](../LICENSES/CC-BY-4.0.txt)), within the source terms above |

Withheld item-level fields:

- **Movie.** Runtime, original language and production country come from The Movie Database (TMDB). The TMDB API terms prohibit derivatives of TMDB content and require a separate agreement for providing data sets that contain it, so the catalog keeps only the MovieLens-derived genres and decades, and the Movie records keep only their question side: their candidate descriptions and answer sets depend on the TMDB attributes. The question side still names TMDB-derived constraint values, such as language codes, and states answer counts. With your own TMDB credentials, `python -m privacyqa fetch-tmdb` and `build-movie-catalog` build the complete catalog in the run root ([docs/pipeline.md](../docs/pipeline.md#stages)). Through the `tmdbId` of MovieLens `links.csv`, it adds to every movie `runtime::short`, `runtime::medium` or `runtime::long` for a TMDB runtime under 90, under 120, or at least 120 minutes; `language::<code>` for the TMDB original language, if it is one of the ten most frequent original languages of the catalog; and `origin::<code>` for every TMDB production country among the ten most frequent production countries of the catalog. Genres are the MovieLens genres in lower case, and decades come from the year at the end of the MovieLens title. Executing the released constraints against that catalog yields the answer sets. The reported catalog used a TMDB snapshot taken at 2026-09-24 17:01 UTC; a later snapshot may change some attributes and answers.
- **Recipe.** Recipe identifiers, recipe tags and ingredient lists derive from Recipe1M, which is available only to research institutions under its own terms and may not be redistributed. As in the public RecipeADQA release, the Recipe records keep their question side. A Recipe1M licensee can produce the RecipeADQA resources with the RecipeADQA pipeline, build the RecipeADKG catalog from them with `python -m privacyqa build-recipe-catalog` (one row per recipe: `item_id`, `attributes` from the recipe tags, `ingredients`), and execute the released constraints against it.

`--restricted-out` of `scripts/export_release.py` writes the complete Movie and Recipe records to a private directory for local use. Do not commit or publish that directory.
