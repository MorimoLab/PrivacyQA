# Configurations

`paper.json` is the configuration of the reported run, and `movie_rules.json`, `beauty_rules.json` and `recipe_rules.json` hold the rules of the catalog stages. Every setting is the one used for the paper, except the values in angle brackets:

| Placeholder | Meaning |
|---|---|
| `<RUN_ROOT>` | Private directory for the outputs of every stage, outside the repository |
| `<MOVIELENS_32M>`, `<AMAZON_REVIEWS_2023>`, `<FOOD_COM>`, `<RECIPEADQA>` | Directories of the source files listed in [docs/pipeline.md](../docs/pipeline.md#inputs); `<RECIPEADQA>` is the root of the RecipeADQA workspace, with the file paths of the reported run below it |
| `<MOVIE_CATALOG>`, `<RECIPE_CATALOG>` | The `catalog.parquet` files written by `build-movie-catalog` and `build-recipe-catalog`, which contain TMDB and Recipe1M content and stay outside the repository |
| `<QWEN3_1_7B>` | Local snapshot of `Qwen/Qwen3-1.7B` |
| `<ROLE_SALT>`, `<KNOWLEDGE_SEED>`, `<COHORT_FILE>` | Private parameters of the reported run, which are not published; use your own salt and seed, and `null` for the cohort file to draw the cohorts from the roles |

## Movie rules

`build-movie-catalog` uses `movie_rules.json` as follows:

| Key | Use |
|---|---|
| `genres` | The direction of every MovieLens genre; `(no genres listed)` gives none |
| `min_year`, `max_year` | Years whose decade becomes an `era` direction; the year is the one at the end of the MovieLens title |
| `runtime_bins` | Runtime directions with their exclusive upper bounds in minutes; the last bin has no bound |
| `top_languages`, `top_origins` | Numbers of the most frequent TMDB original languages and production countries of the catalog that become directions, with ties broken by code |
| `unknown_languages` | TMDB language codes that count as missing |
| `schema_version`, `kb_version` | Versions written to the schema |

## Beauty rules

`build-beauty-catalog` uses the groups of `beauty_rules.json` as follows:

| Group | Use |
|---|---|
| `placeholders`, `pending_patterns`, `max_value_characters`, `max_value_words`, `numeric_pending_fields` | Excluding placeholder values, and holding back suspicious values (links, markup, truncated text, sizes, long descriptions, bare numbers) |
| `aliases`, `enumerations`, `compound_fields`, `compound_vocabularies`, `quantity_units` | Field aliases and the finite vocabularies for parsing lists, applicability phrases and form-quantity phrases |
| `brand_character_replacements`, `brand_display_marks` | Brand typography |
| `screening` | Placeholders, underspecified values, field mismatches, spelling aliases and known literals of the five list fields |
| `reviewed_literals` | Reviewed literal aliases with their basis, values held back after review, and invisible display characters to remove |
| `split_fields` | Fields whose list values become one attribute per member |
| `attribute_aliases` | Reviewed whole-attribute aliases, each with its basis, notes and source links |
| `minimum_catalog_products` | Products that a catalog attribute needs |
| `excluded_aspects` | Aspects that are not extracted: `brand`. Its rules stay in the file without effect; removing it from this list adds brand to the catalog |
| `schema_version`, `kb_version` | Versions written to the schema |

## Recipe rules

`recipe_rules.json` holds the versions written to the Recipe schema. `build-recipe-catalog` applies no further rules: the catalog keeps the recipes, tags and ingredients of the RecipeADQA resources, and the schema their aspect-direction pairs and tags.

Relative paths start at this directory; make them absolute when you copy `paper.json` outside the repository to fill in its placeholders ([docs/pipeline.md](../docs/pipeline.md#configuration)). The code rejects a configuration with a missing section, a stage whose output already exists, and a run root inside the repository.
