"""The catalog stages with the released rules on fictional sources."""
import json

from privacyqa.__main__ import main
from privacyqa.common import iter_rows, read_json
from privacyqa.config import Config
from tests.fixtures import TMDB, make_config


def catalog(config, stage, field):
    return {r[field]: r for r in iter_rows(config.stage(stage) / 'catalog.parquet', 'parquet')}


def tmdb_request(tmdb, token, key):
    assert key == 'fictional-key' and token is None
    return (200, dict(TMDB[tmdb], id=tmdb, spoken_languages=[]), None) if tmdb in TMDB else (404, None, None)


def test_movie_catalog_from_movielens_and_tmdb(tmp_path, monkeypatch):
    path = make_config(tmp_path)
    monkeypatch.setenv('TMDB_API_KEY', 'fictional-key')
    monkeypatch.delenv('TMDB_READ_ACCESS_TOKEN', raising=False)
    monkeypatch.setattr('privacyqa.movie_catalog.tmdb_request', tmdb_request)
    assert main(['fetch-tmdb', '--config', str(path)]) == 0 and main(['build-movie-catalog', '--config', str(path)]) == 0
    config = Config.load(path)
    assert read_json(config.stage('tmdb') / 'manifest.json')['statuses'] == {'ok': 3, 'not_found': 1, 'missing_link': 1}
    assert all(set(r['data'] or {}) <= {'id', 'runtime', 'original_language', 'production_countries'}
               for r in iter_rows(config.stage('tmdb') / 'snapshot.jsonl', 'jsonl'))
    movies = {m: r['attributes'] for m, r in catalog(config, 'movie-catalog', 'movieId').items()}
    # The most frequent language (en) and production country (US) are kept; a year before 1870 has no decade.
    assert movies == {'1': ['era::1990s', 'genre::comedy', 'genre::drama', 'language::en', 'origin::US', 'runtime::short'],
                      '2': ['era::2000s', 'genre::sci-fi', 'origin::US', 'runtime::long'], '3': [], '4': ['genre::children', 'genre::imax'],
                      '5': ['era::2010s', 'genre::action', 'language::en', 'runtime::medium']}
    schema = read_json(config.stage('movie-catalog') / 'schema.json')
    assert schema['evidence_atoms'] == schema['catalog_atoms'] == sorted({a for v in movies.values() for a in v})


def test_beauty_catalog_cleans_splits_merges_and_excludes_brand(tmp_path):
    path = make_config(tmp_path)
    assert main(['build-beauty-catalog', '--config', str(path)]) == 0
    config = Config.load(path)
    products = {p: r['attributes'] for p, r in catalog(config, 'beauty-catalog', 'parent_asin').items()}
    schema = read_json(config.stage('beauty-catalog') / 'schema.json')
    assert list(products) == ['M01', 'M02', 'M03', 'M04', 'M05'] and products['M05'] == []
    assert products['M01'] == ['category::["beauty & personal care","hair care"]', 'hair_type::all', 'item_form::cream',
                               'material::3d mesh', 'scent::lavender', 'scent::vanilla']
    # Placeholders and truncated values are dropped; a scent carried by one product stays below the catalog minimum.
    assert products['M04'] == ['category::["beauty & personal care","skin care"]', 'item_form::gel', 'skin_type::all']
    assert not any(a.startswith('brand::') for a in schema['catalog_atoms'])
    assert set(schema['catalog_atoms']) - set(schema['evidence_atoms']) == {'hair_type::all', 'skin_type::all'}
    assert schema['schema_version'] == 'amazon-2023-beauty-schema-v1' and len(schema['catalog_atoms']) == 9


def test_recipe_catalog_from_recipeadqa_resources(tmp_path):
    path = make_config(tmp_path)
    assert main(['build-recipe-catalog', '--config', str(path)]) == 0
    config = Config.load(path)
    recipes = {i: (r['attributes'], r['ingredients']) for i, r in catalog(config, 'recipe-catalog', 'item_id').items()}
    # Two recipe records share one recipe URI, so their ingredients are joined.
    assert recipes == {'0': (['occasion::dessert', 'taste::sweet'], ['egg', 'milk', 'sugar']), '1': (['time::max_30m'], ['salt']), '2': ([], [])}
    schema = read_json(config.stage('recipe-catalog') / 'schema.json')
    assert schema['evidence_atoms'] == ['audience::kids', 'occasion::dessert', 'taste::sweet']
    assert schema['catalog_atoms'] == ['occasion::dessert', 'taste::sweet', 'time::max_30m'] and json.dumps(schema)
