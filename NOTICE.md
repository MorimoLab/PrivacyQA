# Notice

Copyright 2026 The PrivacyQA authors.

The code of this repository is licensed under the Apache License 2.0 (`LICENSE`). The questions, constraint sets, weights and splits in `data/` and the results in `results/` are licensed under CC BY 4.0 (`LICENSES/CC-BY-4.0.txt`), within the terms of the upstream sources stated in `data/README.md`.

## Third-party code

- `third_party/recipeadqa/` contains files of RecipeADQA, licensed under the Apache License 2.0: the constraint search and its JSON utilities in `code/common/`. Their license and notice are kept in that directory. The files are unchanged.
- `recipe_ingredient_draw` in `src/privacyqa/banks.py` adapts the ingredient sampling of RecipeADQA's KGQA-style baseline (`kgqa_style.sample_one`, Apache License 2.0).
- The mechanism uses OpenDP 0.15.1 (MIT License), installed as a dependency.

## Third-party data

| Source | Use | Terms |
|---|---|---|
| MovieLens 32M (GroupLens Research) | Ratings of the Movie domain; movie identifiers, genres and release years | MovieLens usage license, reproduced in `data/movie/LICENSE.txt` |
| The Movie Database (TMDB) | Runtime, original language and production country of the Movie catalog | TMDB API terms of use; TMDB-derived attributes are not redistributed. This product uses the TMDB API but is not endorsed or certified by TMDB. |
| Amazon Reviews 2023 (McAuley Lab) | Ratings and product metadata of the Beauty domain | Terms of the source; see `data/README.md` |
| Food.com reviews and RecipeADQA | Review judgments and the aspect-direction schema of the Recipe domain | RecipeADQA data under CC BY 4.0 |
| Recipe1M and FoodKG | Recipe catalog (RecipeADKG) of the Recipe domain | Not redistributed |
| Qwen3-1.7B (Qwen team) | Base model of the QA experiments | Apache License 2.0; not redistributed |
