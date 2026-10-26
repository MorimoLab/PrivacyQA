"""Fictional catalogs, ratings, reviews and judgments, and a run configuration with small plans."""
import csv
import gzip
import json
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
SEARCH = REPO / 'third_party' / 'recipeadqa' / 'code' / 'common' / 'qa' / 'constraint_search.py'
INGREDIENTS = [f'ingredient {i:02d}' for i in range(24)]


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    return path


def write_domain(root, domain, field, rows, evidence, catalog, extra=()):
    folder = root / 'catalogs' / domain
    folder.mkdir(parents=True)
    columns = {field: pa.array([r[field] for r in rows]), 'attributes': pa.array([r['attributes'] for r in rows], type=pa.list_(pa.string()))}
    for name in extra:
        columns[name] = pa.array([r[name] for r in rows], type=pa.list_(pa.string()))
    pq.write_table(pa.table(columns), folder / 'catalog.parquet')
    dump(folder / 'schema.json', dict(domain=domain, schema_version=f'fictional-{domain}-schema', kb_version=f'fictional-{domain}-catalog',
                                      evidence_atoms=sorted(evidence), catalog_atoms=sorted(catalog)))


def write_catalogs(root, rng):
    genres, eras, languages = [f'genre::g{i}' for i in range(6)], ['era::1990s', 'era::2000s', 'era::2010s'], ['language::en', 'language::fr']
    movies = [dict(movieId=str(i), attributes=sorted(set(rng.sample(genres, rng.randint(1, 3))) | {rng.choice(eras), rng.choice(languages)}))
              for i in range(140)]
    atoms = genres + eras + languages
    write_domain(root, 'movie', 'movieId', movies, atoms, atoms)
    scents, forms, skins = [f'scent::s{i}' for i in range(5)], ['item_form::cream', 'item_form::gel', 'item_form::oil'], ['skin_type::dry', 'skin_type::oily', 'skin_type::all']
    products = [dict(parent_asin=f'B{i:05d}', attributes=sorted(set(rng.sample(scents, rng.randint(0, 2))) | {rng.choice(forms), rng.choice(skins)}))
                for i in range(160)]
    products.append(dict(parent_asin='B99999', attributes=[]))  # A product without attributes.
    write_domain(root, 'beauty', 'parent_asin', products, scents + forms + skins[:2], scents + forms + skins)
    tastes, occasions = ['taste::sweet', 'taste::sour', 'taste::spicy', 'taste::salty'], ['occasion::dessert', 'occasion::dinner', 'occasion::lunch']
    recipes = [dict(item_id=str(i), attributes=sorted({rng.choice(tastes), rng.choice(occasions)} | ({'time::max_30m'} if rng.random() < .4 else set())),
                    ingredients=sorted(rng.sample(INGREDIENTS, rng.randint(3, 7)))) for i in range(160)]
    write_domain(root, 'recipe_text', 'item_id', recipes, tastes + occasions + ['audience::kids'], tastes + occasions + ['time::max_30m'],
                 extra=('ingredients',))
    return movies, products, recipes


def write_evidence(root, rng, movies, products, recipes, users=700):
    with (root / 'movie_ratings.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['userId', 'movieId', 'rating', 'timestamp'])
        for user in range(1, users + 1):
            items = rng.sample(movies, rng.randint(10, 24))
            for t, movie in enumerate(items):
                writer.writerow([user, movie['movieId'], rng.choice([0.5 * k for k in range(1, 11)]), 1000 * user + t])
            writer.writerow([user, items[0]['movieId'], 5.0, 1000 * user + 999])  # A later rating replaces the first one.
    with gzip.open(root / 'beauty_ratings.csv.gz', 'wt', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'parent_asin', 'rating', 'timestamp'])
        for user in range(users):
            for t, product in enumerate(rng.sample(products, rng.randint(6, 14))):
                writer.writerow([f'U{user:05d}', product['parent_asin'], float(rng.randint(1, 5)), 10 ** 12 + 1000 * user + t])
    reviews, judgments, review = [], [], 0
    directions = [('taste', 'sweet'), ('taste', 'sour'), ('taste', 'spicy'), ('taste', 'salty'), ('occasion', 'dessert'),
                  ('occasion', 'dinner'), ('occasion', 'lunch'), ('audience', 'kids')]
    for author in range(users):
        for _ in range(rng.randint(1, 4)):
            review += 1
            reviews.append(dict(ReviewId=review, AuthorId=10000 + author, RecipeId=int(rng.choice(recipes)['item_id'])))
            for sentence in range(rng.randint(0, 3)):
                for aspect, direction in rng.sample(directions, rng.randint(1, 3)):
                    label = rng.choices(['support', 'oppose', 'irrelevant'], weights=[6, 1, 3])[0]
                    judgments.append(dict(review_id=review, sentence_index=sentence, aspect=aspect, direction=direction, deberta_pred_label=label))
    pq.write_table(pa.Table.from_pylist(reviews), root / 'reviews.parquet')
    half = len(judgments) // 2
    for i, part in enumerate((judgments[:half], judgments[half:])):
        with (root / f'judgments_{i}.tsv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(judgments[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(part)


METADATA = [
    dict(parent_asin='M01', categories=['Beauty & Personal Care', 'Hair Care'],
         details={'Brand': 'Acme', 'Hair Type': 'All Hair Types', 'Scent': 'Lavender, Vanilla', 'Item Form': 'Creams', 'Material': '3D-Mesh'}),
    dict(parent_asin='M02', categories=['Beauty & Personal Care', 'Hair Care'],
         details={'Brand': 'Acme', 'Hair Type': 'All', 'Scent': 'Vanilla', 'Item Form': 'Cream', 'Material': '3D Mesh'}),
    dict(parent_asin='M03', categories=['Beauty & Personal Care', 'Skin Care'],
         details={'Brand': 'Other', 'Skin Type': 'All', 'Scent': 'Lavender', 'Item Form': 'Gel', 'Material': 'N/A'}),
    dict(parent_asin='M04', categories=['Beauty & Personal Care', 'Skin Care'],
         details={'Skin Type': 'All Skin Types', 'Scent': 'Rose', 'Item Form': 'gels', 'Material': 'See more...'}),
    dict(parent_asin='M05', categories=[], details={})]


def write_metadata(root):
    with gzip.open(root / 'beauty_metadata.jsonl.gz', 'wt', encoding='utf-8') as f:
        for row in METADATA:
            f.write(json.dumps(dict(row, title='Fictional product ' + row['parent_asin'])) + '\n')


MOVIELENS = [('1', 'Alpha (1995)', 'Comedy|Drama', '11'), ('2', 'Beta (2001)', 'Sci-Fi', '22'), ('3', 'Gamma', '(no genres listed)', ''),
             ('4', 'Delta (1850)', "Children's|IMAX", '44'), ('5', 'Epsilon (2010)', 'Action', '55')]
TMDB = {11: dict(runtime=85, original_language='en', production_countries=[dict(iso_3166_1='US')]),
        22: dict(runtime=130, original_language='fr', production_countries=[dict(iso_3166_1='FR'), dict(iso_3166_1='US')]),
        55: dict(runtime=100, original_language='en', production_countries=[dict(iso_3166_1='GB')])}


def write_movielens(root):
    with (root / 'movies.csv').open('w', newline='') as f:
        csv.writer(f).writerows([('movieId', 'title', 'genres')] + [m[:3] for m in MOVIELENS])
    with (root / 'links.csv').open('w', newline='') as f:
        csv.writer(f).writerows([('movieId', 'imdbId', 'tmdbId')] + [(m[0], '00' + m[0], m[3]) for m in MOVIELENS])
    rules = json.loads((REPO / 'configs' / 'movie_rules.json').read_text())
    dump(root / 'movie_rules.json', dict(rules, top_languages=1, top_origins=1))


def write_recipe_sources(root):
    """A tiny answer index, recipe identifier index, tagged recipe records and schemas in the RecipeADQA formats."""
    import sqlite3
    folder = root / 'recipeadqa'
    folder.mkdir()
    with sqlite3.connect(folder / 'answer_index.sqlite') as db:
        db.execute('CREATE TABLE answers (answer_id INTEGER PRIMARY KEY, recipe_uri TEXT)')
        db.execute('CREATE TABLE tags (tag_id TEXT PRIMARY KEY)')
        db.execute('CREATE TABLE answer_tags (answer_id INTEGER, tag_id TEXT)')
        db.executemany('INSERT INTO answers VALUES (?, ?)', [(0, 'uri:a'), (1, 'uri:b'), (2, 'uri:c')])
        db.executemany('INSERT INTO tags VALUES (?)', [('taste::sweet',), ('occasion::dessert',), ('time::max_30m',)])
        db.executemany('INSERT INTO answer_tags VALUES (?, ?)', [(0, 'taste::sweet'), (0, 'occasion::dessert'), (1, 'time::max_30m')])
    with (folder / 'recipe_uris.tsv').open('w', newline='') as f:
        csv.writer(f, delimiter='\t').writerows([('recipe_id', 'recipe_uri'), ('r1', 'uri:a'), ('r2', 'uri:b'), ('r3', 'uri:c'), ('r4', 'uri:a')])
    (folder / 'recipes.json').write_text(json.dumps([dict(id='r1', ingredients=[dict(name='sugar'), dict(name=' milk ')]),
        dict(id='r2', ingredients=[dict(name='salt'), dict(name='')]), dict(id='r3', ingredients=[]), dict(id='r4', ingredients=[dict(name='egg')])]))
    with (folder / 'directions.tsv').open('w', newline='') as f:
        csv.writer(f, delimiter='\t').writerows([('aspect', 'direction'), ('taste', 'sweet'), ('occasion', 'dessert'), ('audience', 'kids')])
    with (folder / 'tags.tsv').open('w', newline='') as f:
        csv.writer(f, delimiter='\t').writerows([('tag_id',), ('taste::sweet',), ('occasion::dessert',), ('time::max_30m',)])
    dump(root / 'recipe_rules.json', dict(schema_version='fictional-recipe-schema', kb_version='fictional-recipe-catalog'))


def make_config(root, cohort_file=None):
    rng = random.Random(20261026)
    movies, products, recipes = write_catalogs(root, rng)
    write_evidence(root, rng, movies, products, recipes)
    write_metadata(root)
    write_movielens(root)
    write_recipe_sources(root)
    domain = lambda d: dict(catalog=f'catalogs/{d}/catalog.parquet', schema=f'catalogs/{d}/schema.json')
    return dump(root / 'config.json', {
        'run_root': str(root / 'run'),
        'domains': {'movie': dict(domain('movie'), movies='movies.csv', links='links.csv', rules='movie_rules.json',
                                  ratings=dict(path='movie_ratings.csv', format='csv', user='userId', item='movieId')),
                    'beauty': dict(domain('beauty'), metadata='beauty_metadata.jsonl.gz', rules=str(REPO / 'configs' / 'beauty_rules.json'),
                                   ratings=dict(path='beauty_ratings.csv.gz', format='csv.gz', user='user_id', item='parent_asin')),
                    'recipe_text': dict(domain('recipe_text'), rules='recipe_rules.json',
                                        sources={k: f'recipeadqa/{v}' for k, v in dict(answer_index='answer_index.sqlite', recipe_uris='recipe_uris.tsv',
                                                 recipes='recipes.json', direction_schema='directions.tsv', tag_schema='tags.tsv').items()},
                                        reviews=dict(path='reviews.parquet', format='parquet'),
                                        judgments=[dict(path=f'judgments_{i}.tsv', format='tsv') for i in range(2)])},
        'recipe_search_module': str(SEARCH),
        'private': dict(role_salt='fictional-salt-for-tests-only', knowledge_seed='fictional-knowledge-seed', cohort_file=cohort_file),
        'roles': dict(proportions=dict(auxiliary=0.2, protected=0.5, attack_reference=0.1, attack_nonmember=0.1, utility_reference=0.1)),
        'banks': dict(max_cores=24, core_lengths=[1, 2, 3], public_seed=7, rating_min_core_matches=2, target_answers=20, answer_upper=25,
                      answer_lower=dict(movie=2, beauty=2, recipe_text=1), max_options=4, max_conditions=10,
                      random=dict(public_seed=8, attempts_per_core=64, max_extensions=4)),
        'test': dict(requests=12, seed=3),
        'cohorts': dict(members=8, test_nonmembers=8, calibration_nonmembers=8, reference_users=20, selection_seed=4, known_ratings=5, known_reviews=2),
        'release': dict(experiment_id='fixture', training_releases=4, calibration_releases=2, targets_per_class=4, schedule_seed=5,
                        familywise_alpha='1/20', measurements=42),
        'attack': dict(seed=6, classifier_C=1, max_iterations=500, tolerance=1e-6),
        'attack_metrics': dict(bootstrap_replicates=20, bootstrap_seed=9, familywise_alpha=0.05),
        'attack_audit': dict(prior_shape=0.5, prior_rate=0, grid_sd=12, grid_laplace_scales=12, placebo_split='calibration', blind_folds=2, blind_seed=10),
        'mechanism_audit': dict(canary_units=[1, 20], trials={'1': 400, '20': 400}, control_trials=400, chunk_trials=200, workers=2, threshold_grid=20),
        'qa': dict(seed=11, development_requests=4, candidates=30, negative_reservoir=64, posting_sample=16, model_path=str(root / 'no-model'),
                   training_token_budget=1000, max_steps=10, learning_rate=0.0002, max_length=4096, max_new_tokens=64, lora_rank=8, lora_alpha=16,
                   inference={d: dict(batch_size=4, prompt_token_limit=8192) for d in ('movie', 'beauty', 'recipe_text')},
                   bootstrap_replicates=20, paired_comparisons=[['DP', 'Non-private'], ['DP', 'Public-only'], ['DP', 'Base']]),
        'preference': dict(draws=20, draw_seed=12, variant_limit=4, bootstrap_replicates=20, bootstrap_seed=13)})
