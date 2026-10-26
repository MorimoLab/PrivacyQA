# Pipeline

The pipeline runs in stages. Every stage reads one configuration file, takes the outputs of earlier stages from the run root that the configuration names, and writes a new directory named after the stage; it stops with a named error when an input is missing or malformed, or when its output directory already exists. Run outputs contain private data and must stay outside the repository; the code refuses a run root inside it.

## Inputs

| Source | Files | Where to obtain them |
|---|---|---|
| MovieLens 32M | `movies.csv`, `links.csv`, `ratings.csv` | [ml-32m.zip](https://files.grouplens.org/datasets/movielens/ml-32m.zip) from [GroupLens](https://grouplens.org/datasets/movielens/32m/) |
| TMDB | The record of every movie that `links.csv` links to TMDB | The [TMDB API](https://developer.themoviedb.org/), with your own API key or read access token, through `fetch-tmdb` |
| Amazon Reviews 2023 | `Beauty_and_Personal_Care.csv.gz` (5-core rating-only benchmark) and `meta_Beauty_and_Personal_Care.jsonl.gz` (product metadata) | [Ratings](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Beauty_and_Personal_Care.csv.gz) and [metadata](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl.gz), described on the [Amazon Reviews 2023 site](https://amazon-reviews-2023.github.io/) |
| Food.com | `reviews.parquet` | [Food.com - Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) on Kaggle |
| RecipeADQA | Aspect-direction and tag schemas, review judgments, answer index, recipe URI index and tagged Recipe1M recipes | A workspace of the [RecipeADQA repository](https://github.com/MorimoLab/RecipeADQA), which contains the two schemas in `data/schema/` and the code that writes the other files. Its review-direction classifier writes the judgments from the Food.com reviews ([commands](#review-judgments)); its RecipeADKG and QA-generation stages write the answer index, the recipe URI index and the tagged recipes, and need Recipe1M. RecipeADQA does not distribute these files |

The configuration names every input:

| Input | Configuration | Format |
|---|---|---|
| MovieLens movies and links | `domains.movie.movies`, `domains.movie.links` | `movies.csv` (`movieId`, `title`, `genres`) and `links.csv` (`movieId`, `imdbId`, `tmdbId`), read only by `fetch-tmdb` and `build-movie-catalog` |
| TMDB credentials | Environment variable `TMDB_READ_ACCESS_TOKEN` or `TMDB_API_KEY` | Read only by `fetch-tmdb`, and never written to an output |
| Movie rules | `domains.movie.rules` | `configs/movie_rules.json` ([configs/README.md](../configs/README.md#movie-rules)) |
| Beauty product metadata | `domains.beauty.metadata` | `meta_Beauty_and_Personal_Care.jsonl.gz` of Amazon Reviews 2023, read only by `build-beauty-catalog` |
| Beauty cleaning rules | `domains.beauty.rules` | `configs/beauty_rules.json` ([configs/README.md](../configs/README.md#beauty-rules)) |
| RecipeADQA resources | `domains.recipe_text.sources` | `answer_index`: SQLite with the tables `answers` (`answer_id`, `recipe_uri`), `tags` (`tag_id`) and `answer_tags` (`answer_id`, `tag_id`); `recipe_uris`: TSV with `recipe_id` and `recipe_uri`; `recipes`: JSON array of tagged Recipe1M records with `id` and `ingredients` (each with `name`); `direction_schema`: TSV with `aspect` and `direction`; `tag_schema`: TSV with `tag_id`. Read only by `build-recipe-catalog` |
| Recipe rules | `domains.recipe_text.rules` | `configs/recipe_rules.json`, the versions of the Recipe schema |
| Catalogs | `domains.<domain>.catalog` | Parquet with the item identifier (`movieId`, `parent_asin` or `item_id`), `attributes` (a list of `aspect::direction` strings) and, for Recipe, `ingredients` |
| Schemas | `domains.<domain>.schema` | JSON with `schema_version`, `kb_version`, `evidence_atoms` (directions that can form preference cores) and `catalog_atoms` (directions that can be constraints) |
| Ratings | `domains.movie.ratings`, `domains.beauty.ratings` | `ratings.csv` of MovieLens 32M (`userId`, `movieId`, `rating`, `timestamp`) and `Beauty_and_Personal_Care.csv.gz` of the Amazon Reviews 2023 5-core rating-only benchmark (`user_id`, `parent_asin`, `rating`, `timestamp`) |
| Reviews | `domains.recipe_text.reviews` | Food.com `reviews.parquet`, of which the pipeline reads `ReviewId`, `AuthorId` and `RecipeId` |
| Review judgments | `domains.recipe_text.judgments` | The RecipeADQA judgment files: one row per review sentence and aspect-direction, with `review_id`, `sentence_index`, `aspect`, `direction` and `deberta_pred_label` (support, oppose or irrelevant) |
| Constraint search | `recipe_search_module` | `third_party/recipeadqa/code/common/qa/constraint_search.py` |
| Private parameters | `private` | `role_salt`, the private string of the role assignment; `knowledge_seed`, which selects the partial histories known to the attackers; and `cohort_file`, an optional JSON file with the users of every cohort group ([format](#cohort-file)) |

The Beauty catalog and the schemas of all domains are in `data/`. The Movie and Recipe catalogs contain TMDB and Recipe1M content, which may not be redistributed ([data/README.md](../data/README.md#terms-of-use)), so `build-movie-catalog` and `build-recipe-catalog` write them to the run root; set `domains.movie.catalog` and `domains.recipe_text.catalog` to the `catalog.parquet` files they write. The schemas they write equal those in `data/`.

### Review judgments

`domains.recipe_text.judgments` lists the prediction shards of the RecipeADQA review-direction classifier. From the root of a clone of the RecipeADQA repository, in an environment with pandas, pyarrow, PyTorch and Transformers, two of its steps write them:

```bash
git lfs pull  # the released classifier checkpoint
python code/02_review_direction/step_01_build_recall_candidates.py --reviews <FOOD_COM>/reviews.parquet --triggers data/schema/aspect_direction_lexicon.tsv
python code/02_review_direction/step_06_infer_full_review_candidates.py --targets data/schema/aspect_direction_schema.tsv --gpus 0,1,2,3
```

The first step recalls sentence-direction candidates from the reviews with the trigger lexicon. The second classifies every candidate as support, oppose or irrelevant with the released checkpoint and writes the shards to `data/review/full_comment_predictions_and_tag_alignment/prediction_shards/`, the paths that `configs/paper.json` lists. With the public lexicon and schema, the two steps receive the recall patterns and the classifier inputs of the reported run, which has 4,418,496 candidates in 45 shards.

### Cohort file

`private.cohort_file` fixes the cohort groups of the membership experiments; with `null`, `freeze-cohorts` draws them from the roles. The file is a JSON object `{"domains": {"movie": {...}, "beauty": {...}, "recipe_text": {...}}}`. Each domain lists the user identifiers, as strings (MovieLens `userId`, Amazon `user_id` and Food.com `AuthorId`), of six disjoint groups, each drawn from one role with the size that the `cohorts` section sets:

| Group | Role | Size | Use |
|---|---|---|---|
| `train_pool` | Attack-reference (`attack_reference`) | 2 × `members` | Members and targets of the shadow-training releases |
| `calibration_pool` | Attack-reference (`attack_reference`) | 2 × `members` | Members and targets of the shadow-calibration releases |
| `target_members` | Construction (`protected`) | `members` | Member targets of the target release |
| `target_calibration` | Nonmember (`attack_nonmember`) | `calibration_nonmembers` | Nonmembers that calibrate the attack thresholds |
| `target_nonmembers` | Nonmember (`attack_nonmember`) | `test_nonmembers` | Nonmember targets of the target release |
| `reference_users` | Reference (`utility_reference`) | `reference_users` | Held-out users of the fidelity measures and the preference-weighted test set |

`freeze-cohorts` stops when a group is missing, has another size, overlaps another group, or contains a user of another role.

## Configuration

`configs/paper.json` is the configuration of the reported run, with private values and local paths as placeholders. To run the pipeline, copy it to a private directory outside the repository and replace the placeholders. Relative paths start at the directory of the configuration file, so make them absolute in the copy: the `rules` and `schema` entries of every domain, the Beauty `catalog` and `recipe_search_module`. The filled-in file holds the private salt and seed; keep it out of the repository.

| Section | Contents |
|---|---|
| `run_root` | Private directory that receives the output of every stage |
| `roles` | Proportions of the five roles (auxiliary, construction, attack-reference, nonmember and reference users) |
| `banks` | Construction template: number of cores and core lengths, the minimum number of matching catalog items of a rating core, the target and range of answer counts, the number of variants per core, the constraint limit, and the seeds and sampling budget of the random bank |
| `test` | Size and seed of the test set |
| `cohorts` | Numbers of member targets, test and calibration nonmembers and reference users, the seed used when no cohort file is given, and the size of the partial histories (rated items or reviews) |
| `release` | Run name, numbers of shadow-training and shadow-calibration releases, targets per release, schedule seed, familywise level of the threshold, and the number of planned measurements |
| `attack`, `attack_metrics`, `attack_audit`, `mechanism_audit` | Settings of the trained attacks, of the metrics and bootstrap intervals, of the controls and likelihood-ratio attacks, and of the synthetic audit |
| `qa` | Candidate pools, validation set, model settings and budget, inference batches per domain, bootstrap and the paired comparisons |
| `preference` | Draws, seeds, variant limit and bootstrap of the preference-weighted test set |

## Stages

Run every stage from the repository root as `python -m privacyqa STAGE --config CONFIG` (with `PYTHONPATH=src` or after `pip install -e .`).

| Stage | Output | What it does |
|---|---|---|
| `fetch-tmdb` | `tmdb/` | Requests the TMDB record of every movie that `links.csv` links to TMDB and keeps its runtime, original language and production countries in `snapshot.jsonl`; movies that TMDB does not find are recorded as `not_found`, and a request that still fails after retries stops the stage. The snapshot of the reported run was taken at 2026-09-24 17:01 UTC |
| `build-movie-catalog` | `movie-catalog/` | Builds the Movie catalog and schema from `movies.csv`, `links.csv`, the TMDB snapshot and the Movie rules; its genre and decade directions equal `data/movie/catalog/catalog_movielens.parquet` |
| `build-beauty-catalog` | `beauty-catalog/` | Extracts the category path and detail fields of every Beauty product from the metadata and cleans, splits and merges them with the rules; the result equals `data/beauty/catalog` |
| `build-recipe-catalog` | `recipe-catalog/` | Builds the RecipeADKG catalog, with the tags and Recipe1M ingredients of every recipe in the RecipeADQA answer index, and the Recipe schema of review directions and recipe tags |
| `build-index` | `index/` | Loads the ratings, reviews and review judgments into one DuckDB file and lists the users of every source |
| `assign-roles` | `roles/` | Assigns every user to one of five roles from the user identifier and the private salt |
| `build-banks` | `banks/` | Builds the main and random request banks of every domain from the catalogs alone, and draws the test set from the main banks |
| `freeze-cohorts` | `cohorts/` | Fixes the cohorts of the membership experiments, the schedule of shadow releases, and the supports and partial histories of the cohort users |
| `build-supports` | `supports/` | Computes the support and bounded contribution of every construction user on both banks |
| `release` | `releases/` | Runs the mechanism for the 49 releases of every bank and domain (32 shadow-training, 16 shadow-calibration and the target release), records every measurement in `releases/ledger.sqlite`, and writes the views of DP and of the three ablations; `--dry-run` writes the plan, thresholds and populations to `release-plan/` without measuring |
| `attack` | `attacks/` | Fits the aggregate and per-unit classifiers on the shadow-training releases and scores the targets of every target-release view, with the matching score |
| `attack-metrics` | `attack-metrics/` | Extended metrics, paired bootstrap intervals and the approximate simultaneous intervals of the primary attack |
| `attack-audit` | `attack-audit/` | Checks every view against its stored vector, and runs the negative controls (placebo releases and the blind baseline) and the realistic and informed likelihood-ratio attacks |
| `mechanism-audit` | `mechanism-audit/` | Audits the mechanism with canaries on synthetic vectors; reads no user data |
| `qa-data` | `qa-data/` | Builds the training records of the eight training sources, after training holdout, and the test and validation records with their candidate pools |
| `qa-fidelity` | `qa-fidelity/` | Reads the reference users' histories and computes user coverage, mass coverage and the total variation distances of every training source |
| `qa-analysis` | `qa-analysis/` | Recomputes the scores of the 27 model runs from their outputs and computes the paired cluster bootstrap intervals |
| `preference-data` | `preference-data/` | Draws the preference-weighted test set from the reference users' support |
| `preference-analysis` | `preference-analysis/` | Scores the preference-weighted test set and checks the re-evaluations of the main DP models |

The four catalog stages read no user data and run once; the other stages read the catalogs that the configuration names.

The models run in the model environment, between `qa-fidelity` and `qa-analysis` and after `preference-data`:

```bash
python scripts/run_models.py --config CONFIG --python MODEL_PYTHON --gpus 0,1,2,3 --slots-per-gpu 2
python scripts/run_models.py --config CONFIG --python MODEL_PYTHON --gpus 0,1,2,3 --inference
```

The first command trains and evaluates the 27 models (eight LoRA models and Base per domain) in `qa-models/`; one run is `python -m privacyqa.qa.train --config CONFIG --domain DOMAIN --method SOURCE`. The second command re-evaluates the test set with the main DP model of each domain and answers the preference-weighted test set with the four main-bank models of each domain, in `preference-models/`. The reported Movie and Beauty runs used two jobs per 24 GB GPU with a memory fraction of 0.45, and the Recipe runs one job per GPU with 0.92; the memory fraction does not change the results.

Finally, `python scripts/make_results_csv.py --config CONFIG --out results` writes the result tables, and `python scripts/export_release.py --config CONFIG --out data [--restricted-out PRIVATE_DIR]` writes the released data; `--restricted-out` adds the withheld Movie and Recipe fields to a private directory for local use.

## Reproducing the reported run

`configs/paper.json` holds every public setting of the reported run. Its private inputs are not published: the role salt, the knowledge seed and the cohort file identify which users had which role, which users took part in the membership experiments, and which parts of their histories the attackers knew. Without them, the pipeline runs the same procedure on a different pseudorandom partition of the users; with `cohort_file` set to `null`, `freeze-cohorts` draws the cohort groups from the roles in a fixed pseudorandom order. The catalogs, the banks, the test and validation sets, the candidate pools and the question templates depend only on the item-side sources and are reproduced exactly from them. The TMDB snapshot of the reported run is not distributed, and a later snapshot may change some Movie attributes and, with them, the Movie banks and answers.

## Modules

| Module | Role (paper component) |
|---|---|
| `privacyqa.common`, `privacyqa.config` | Errors, files, fixed pseudorandom orders, and the run configuration |
| `privacyqa.movie_catalog` | The TMDB snapshot, and the Movie catalog from MovieLens and TMDB |
| `privacyqa.beauty_catalog` | The Beauty catalog from the product metadata and the cleaning rules |
| `privacyqa.recipe_catalog` | The Recipe catalog of RecipeADKG from the RecipeADQA resources |
| `privacyqa.catalog` | Catalogs, complete answer sets and the RecipeADQA ingredient search |
| `privacyqa.banks` | Request banks, the test set, and reconstruction of request distributions from released masses (Sampling QA Pairs) |
| `privacyqa.evidence` | Evidence index, support rules of ratings and review judgments (Evidence Adapter), and partial histories |
| `privacyqa.roles`, `privacyqa.cohorts` | User roles, cohorts and the schedule of shadow releases |
| `privacyqa.supports`, `privacyqa.mechanism` | Contribution bounding and quantization, the OpenDP discrete Laplace mechanism, the threshold and the ledger (User-Level Private Release) |
| `privacyqa.release` | Target and shadow releases of DP and of the ablations |
| `privacyqa.attacks.features`, `.trained`, `.metrics` | Attacker features, trained attacks, metrics and simultaneous intervals |
| `privacyqa.attacks.audit`, `.synthetic` | Negative controls, likelihood-ratio attacks and the synthetic audit of the mechanism |
| `privacyqa.qa.records`, `.data` | QA records, candidate pools, training holdout and fidelity measures |
| `privacyqa.qa.train`, `.infer`, `.evaluation`, `.analysis`, `.preference` | LoRA training and inference, scoring, paired intervals and the preference-weighted test set |
