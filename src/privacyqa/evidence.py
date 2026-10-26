"""The evidence index of user histories, and the per-user quantities computed from it.

Movie and Beauty evidence is the latest rating of each item a user rated; an item is positive when its rating exceeds
the user's mean rating, over all rated items, by at least 0.5. Recipe evidence is the judgment of every review
sentence for every aspect-direction; within one review, a direction is positive when some sentence supports it and no
sentence opposes it. A rating core is supported when at least `rating_min_core_matches` positive items lie in its
complete answer set; a Recipe core is supported when one review contains all its atoms as positive directions.
"""
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import chain, groupby
import json

import duckdb
import numpy as np
import pyarrow as pa

from privacyqa.common import RATING_DOMAINS, progress, rank, require, write_json

SCALES = {'movie': 'rating*2 = floor(rating*2) AND rating BETWEEN 0.5 AND 5',
          'beauty': 'rating = floor(rating) AND rating BETWEEN 1 AND 5'}
LABELS = ('support', 'oppose', 'irrelevant')


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def table_sql(path, format):
    if format == 'parquet':
        return f'read_parquet({literal(path)})'
    require(format in ('csv', 'csv.gz', 'tsv'), 'invalid_table_format', 'Use parquet, csv, csv.gz or tsv')
    return f"read_csv({literal(path)},header=true,delim={literal(chr(9) if format == 'tsv' else ',')},all_varchar=true)"


def batches(cursor, size=4096):
    while rows := cursor.fetchmany(size):
        yield rows


def count(db, sql):
    return db.execute(sql).fetchone()[0]


def build_index(config, output):
    """Load ratings, reviews and review judgments into one DuckDB file; no record is dropped or changed."""
    path = output / 'evidence.duckdb'
    facts = {}
    with duckdb.connect(str(path)) as db:
        path.chmod(0o600)
        for domain in RATING_DOMAINS:
            spec = config['domains'][domain]['ratings']
            source = table_sql(config.file(spec['path']), spec['format'])
            db.execute(f'''CREATE TABLE ratings_{domain} AS SELECT trim(cast("{spec['user']}" AS VARCHAR)) AS user_key,
                trim(cast("{spec['item']}" AS VARCHAR)) AS item_id, try_cast(rating AS DOUBLE) AS rating,
                try_cast(timestamp AS BIGINT) AS timestamp FROM {source}''')
            require(count(db, f"""SELECT count(*) FROM ratings_{domain} WHERE user_key IS NULL OR user_key='' OR item_id IS NULL
                OR item_id='' OR timestamp IS NULL OR timestamp<0 OR rating IS NULL OR NOT isfinite(rating) OR NOT ({SCALES[domain]})""") == 0,
                    'invalid_rating', f'Every {domain} rating has a user, an item, a timestamp and a rating on the scale')
            require(count(db, f'SELECT count(*) FROM (SELECT 1 FROM ratings_{domain} GROUP BY user_key,item_id,timestamp HAVING count(*)>1)') == 0,
                    'duplicate_rating', 'A user rates an item at most once per timestamp, so the latest rating is unique')
            db.execute(f"CREATE TABLE users_{domain} AS SELECT DISTINCT user_key FROM ratings_{domain} ORDER BY user_key")
            facts[domain] = dict(ratings=count(db, f'SELECT count(*) FROM ratings_{domain}'),
                                 users=count(db, f'SELECT count(*) FROM users_{domain}'),
                                 items=count(db, f'SELECT count(DISTINCT item_id) FROM ratings_{domain}'))
            progress(output, 'ratings_loaded', domain=domain, **facts[domain])
        spec = config['domains']['recipe_text']
        db.execute(f'''CREATE TABLE reviews AS SELECT DISTINCT cast(ReviewId AS VARCHAR) AS review_id, cast(AuthorId AS VARCHAR) AS author_id,
            cast(RecipeId AS VARCHAR) AS recipe_id FROM {table_sql(config.file(spec['reviews']['path']), spec['reviews']['format'])}''')
        require(count(db, "SELECT count(*) FROM reviews WHERE review_id IS NULL OR trim(review_id)='' OR author_id IS NULL OR trim(author_id)=''") == 0
                and count(db, 'SELECT count(*) - count(DISTINCT review_id) FROM reviews') == 0,
                'invalid_review', 'Every review has one identifier and one author')
        db.execute('CREATE TABLE judgments(review_id VARCHAR, sentence_index BIGINT, atom VARCHAR, label VARCHAR)')
        for table in spec['judgments']:
            db.execute(f'''INSERT INTO judgments SELECT cast(review_id AS VARCHAR), try_cast(sentence_index AS BIGINT),
                aspect || '::' || direction, deberta_pred_label FROM {table_sql(config.file(table['path']), table['format'])}''')
        require(count(db, f"SELECT count(*) FROM judgments WHERE sentence_index IS NULL OR atom IS NULL OR label IS NULL "
                          f"OR label NOT IN {tuple(LABELS)}") == 0
                and count(db, 'SELECT count(*) FROM judgments j ANTI JOIN reviews r USING(review_id)') == 0,
                'invalid_judgment', 'Every judgment has a sentence, a direction, a known label and a review')
        db.execute('CREATE TABLE users_recipe_text AS SELECT DISTINCT author_id AS user_key FROM reviews ORDER BY user_key')
        facts['recipe_text'] = dict(reviews=count(db, 'SELECT count(*) FROM reviews'), judgments=count(db, 'SELECT count(*) FROM judgments'),
                                    users=count(db, 'SELECT count(*) FROM users_recipe_text'),
                                    labels=dict(db.execute('SELECT label, count(*) FROM judgments GROUP BY label ORDER BY label').fetchall()))
        progress(output, 'reviews_loaded', **facts['recipe_text'])
    manifest = dict(status='index_built', domains=facts)
    write_json(output / 'manifest.json', manifest)
    return manifest


def index_users(config, domain):
    with duckdb.connect(str(config.stage('index') / 'evidence.duckdb'), read_only=True) as db:
        return [row[0] for row in db.execute(f'SELECT user_key FROM users_{domain} ORDER BY user_key').fetchall()]


def histories(config, domain, users):
    """(user, records) for every user of a sorted list, in order, including users without records.

    Rating records are (user, item, latest rating); Recipe records are (user, review, direction, label), with one
    record whose direction is None for a review without judgments.
    """
    require(list(users) == sorted(set(users)), 'unsorted_users', 'Histories are read for a sorted list of distinct users')
    if domain in RATING_DOMAINS:
        query = f'''SELECT user_key, item_id, rating FROM ratings_{domain} JOIN chosen USING(user_key)
            QUALIFY row_number() OVER (PARTITION BY user_key, item_id ORDER BY timestamp DESC) = 1 ORDER BY user_key, item_id'''
    else:
        query = '''SELECT r.author_id, r.review_id, j.atom, j.label FROM reviews r JOIN chosen c ON r.author_id = c.user_key
            LEFT JOIN judgments j USING(review_id) ORDER BY r.author_id, r.review_id, j.atom, j.label'''
    with duckdb.connect(str(config.stage('index') / 'evidence.duckdb'), read_only=True) as db:
        db.register('chosen', pa.table({'user_key': pa.array(list(users), type=pa.string())}))
        groups = groupby(chain.from_iterable(batches(db.execute(query))), key=lambda row: row[0])
        current = next(groups, None)
        for user in users:
            require(current is None or current[0] >= user, 'history_order', 'The history scan follows the user order')
            if current is not None and current[0] == user:
                records = list(current[1])
                current = next(groups, None)
            else:
                records = []
            yield user, records
        require(current is None, 'history_order', 'Every history belongs to a listed user')


def positive_evidence(domain, records, catalog):
    """Positive catalog items (ratings) or the positive directions of every review (Recipe)."""
    if domain in RATING_DOMAINS:
        mean = sum(row[2] for row in records) / len(records) if records else None
        return {item for _, item, rating in records if rating - mean >= .5 and item in catalog.items and catalog.items[item][0]}
    return [positive for _, positive in review_contexts(records)]


def review_contexts(records):
    contexts = []
    for review, rows in groupby(records, key=lambda row: row[1]):
        labels = defaultdict(set)
        for _, _, atom, label in rows:
            if atom is not None:
                labels[atom].add(label)
        contexts.append((review, {atom for atom, values in labels.items() if 'support' in values and 'oppose' not in values}))
    return contexts


def partial_history(domain, user, records, seed, budget):
    """The evidence an attacker knows: `budget` rated items or reviews of the user, in a fixed pseudorandom order.

    Returns the positive evidence of the known part, judged from the known part alone, and its size.
    """
    if domain in RATING_DOMAINS:
        known = sorted(((item, rating) for _, item, rating in records), key=lambda r: (rank(seed, domain, user, r[0]), r[0]))[:budget]
        mean = sum((Fraction(str(rating)) for _, rating in known), Fraction(0)) / len(known) if known else Fraction(0)
        return {item for item, rating in known if Fraction(str(rating)) - mean >= Fraction(1, 2)}, len(known)
    contexts = {json.dumps(['review', review]): positive for review, positive in review_contexts(records)}
    keys = sorted(contexts, key=lambda k: (rank(seed, domain, user, k), k))[:budget]
    return [contexts[k] for k in keys], len(keys)


class SupportRule:
    """Binary support of every unit of one bank, through postings from items (ratings) or directions (Recipe) to units."""

    def __init__(self, bank):
        self.text = bank.domain == 'recipe_text'
        predicates = bank.cores if self.text else bank.core_answers
        self.required = [len(core) for core in bank.cores] if self.text else [bank.rules['rating_min_core_matches']] * len(bank.units)
        self.postings = defaultdict(list)
        for index, predicate in enumerate(predicates):
            for value in predicate:
                self.postings[value].append(index)

    def encode(self, positive):
        result = np.zeros(len(self.required), dtype=np.int64)
        for context in (positive if self.text else (positive,)):
            hits = Counter(index for value in context for index in self.postings.get(value, ()))
            for index, n in hits.items():
                if n >= self.required[index]:
                    result[index] = 1
        return result
