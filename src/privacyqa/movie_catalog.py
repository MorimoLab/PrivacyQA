"""The Movie catalog: MovieLens 32M genres and decades with the runtime, original language and production countries of TMDB.

`fetch-tmdb` reads, with your own TMDB credentials (environment variable TMDB_READ_ACCESS_TOKEN or TMDB_API_KEY), the
TMDB record of every movie that MovieLens `links.csv` links to TMDB. TMDB content may not be redistributed, so the
snapshot stays in the private run root. `build-movie-catalog` derives the attributes with the rules in
configs/movie_rules.json:

- genre: the MovieLens genres in lower case, without "(no genres listed)";
- era: the decade of the year at the end of the MovieLens title, within the declared years;
- runtime: the bin of the positive TMDB runtime in minutes;
- language: the TMDB original language, when it is one of the most frequent original languages of the catalog;
- origin: every TMDB production country that is one of the most frequent production countries of the catalog.

A later TMDB snapshot may change some attributes, and thus some answers.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pyarrow as pa

from privacyqa.common import iter_rows, progress, read_json, require, write_json, write_parquet

TMDB_URL = 'https://api.themoviedb.org/3/movie/{}'
TMDB_FIELDS = ('id', 'runtime', 'original_language', 'production_countries')  # The snapshot keeps only the fields that the catalog uses.
STATUSES = ('ok', 'not_found', 'missing_link')
SCHEMA = pa.schema([('movieId', pa.string()), ('attributes', pa.list_(pa.string()))])


def movielens_rows(config, name, fields):
    path = config.file(config['domains']['movie'][name])
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == fields, 'invalid_movielens_file', f'Use the MovieLens 32M {path.name}')
        return list(reader)


def links(config):
    rows = movielens_rows(config, 'links', ['movieId', 'imdbId', 'tmdbId'])
    result = {int(r['movieId']): int(r['tmdbId']) if r['tmdbId'] else None for r in rows}
    require(len(result) == len(rows), 'duplicate_movie', 'links.csv lists every movie once')
    return result


def tmdb_request(tmdb, token, key):
    """One TMDB movie request: (HTTP status, payload, Retry-After)."""
    query = urllib.parse.urlencode({'language': 'en-US', **({} if token else {'api_key': key})})
    request = urllib.request.Request(TMDB_URL.format(tmdb) + '?' + query, headers={'Authorization': f'Bearer {token}'} if token else {})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return response.status, json.loads(response.read()), None
    except urllib.error.HTTPError as error:
        return error.code, None, error.headers.get('Retry-After')


def fetch_tmdb(config, output, workers=8, rate=30):
    """One TMDB request per linked movie, at most `rate` requests per second, with retries after errors."""
    token, key = os.environ.get('TMDB_READ_ACCESS_TOKEN'), os.environ.get('TMDB_API_KEY')
    require(token or key, 'missing_tmdb_credentials', 'Set TMDB_READ_ACCESS_TOKEN or TMDB_API_KEY')
    linked = links(config)
    lock, next_request, stop = threading.Lock(), [0.0], threading.Event()

    def throttle():
        with lock:
            delay = next_request[0] - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            next_request[0] = time.monotonic() + 1 / rate

    def fetch(item):
        movie, tmdb = item
        if tmdb is None:
            return dict(movieId=movie, tmdbId=None, status='missing_link', data=None), None
        for attempt in range(4):
            if stop.is_set():
                return None, dict(movieId=movie, error='stopped_after_authorization_failure')
            throttle()
            try:
                status, payload, retry_after = tmdb_request(tmdb, token, key)
            except (urllib.error.URLError, OSError, ValueError):
                status, payload, retry_after = None, None, None
            if status == 200:
                require(payload.get('id') == tmdb, 'tmdb_identity_mismatch', 'TMDB returns the requested movie')
                return dict(movieId=movie, tmdbId=tmdb, status='ok', data={k: payload.get(k) for k in TMDB_FIELDS}), None
            if status == 404:
                return dict(movieId=movie, tmdbId=tmdb, status='not_found', data=None), None
            if status in (401, 403):
                stop.set()
                return None, dict(movieId=movie, http_status=status)
            if status == 429:
                with lock:
                    next_request[0] = max(next_request[0], time.monotonic() + min(120, int(retry_after) if (retry_after or '').isdigit() else 5))
            elif status is not None and status < 500:
                return None, dict(movieId=movie, http_status=status)
            time.sleep(2 ** attempt)
        return None, dict(movieId=movie, error='retries_exhausted')

    records, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (record, failure) in enumerate(pool.map(fetch, sorted(linked.items())), 1):
            (records if record else failures).append(record or failure)
            if done % 5000 == 0:
                progress(output, 'tmdb_records', done=done, total=len(linked), failures=len(failures))
    require(not failures, 'incomplete_tmdb_snapshot', f'{len(failures)} TMDB requests failed; start a new run')
    with (output / 'snapshot.jsonl').open('x', encoding='utf-8') as handle:
        os.chmod(output / 'snapshot.jsonl', 0o600)
        for record in records:
            handle.write(json.dumps(record) + '\n')
    manifest = dict(status='tmdb_fetched', movies=len(records), statuses=dict(Counter(r['status'] for r in records)),
                    fetched_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    write_json(output / 'manifest.json', manifest)
    return manifest


def tmdb_snapshot(config, linked):
    config.finished('tmdb', 'tmdb_fetched')
    snapshot = {}
    for row in iter_rows(config.stage('tmdb') / 'snapshot.jsonl', 'jsonl'):
        movie = row['movieId']
        require(movie not in snapshot and row['status'] in STATUSES and row['tmdbId'] == linked.get(movie)
                and (row['status'] != 'ok' or row['data']['id'] == row['tmdbId']),
                'invalid_tmdb_snapshot', 'The snapshot has one record of every linked movie')
        snapshot[movie] = row['data'] or {}
    require(set(snapshot) == set(linked), 'invalid_tmdb_snapshot', 'The snapshot covers every movie of links.csv')
    return snapshot


def runtime_bin(runtime, bins):
    return next(label for label, bound in bins if bound is None or runtime < bound)


def build_movie_catalog(config, output):
    rules = read_json(config.file(config['domains']['movie']['rules']))
    require(set(rules) == {'schema_version', 'kb_version', 'genres', 'min_year', 'max_year', 'runtime_bins', 'top_languages',
                           'top_origins', 'unknown_languages'} and rules['runtime_bins'][-1][1] is None,
            'invalid_movie_rules', 'Declare versions, genres, years, runtime bins and the numbers of languages and origins')
    linked = links(config)
    snapshot = tmdb_snapshot(config, linked)
    movies = movielens_rows(config, 'movies', ['movieId', 'title', 'genres'])
    require({int(r['movieId']) for r in movies} == set(linked) and len(movies) == len(linked), 'movielens_mismatch',
            'movies.csv and links.csv list the same movies')
    prepared, languages, countries = {}, Counter(), Counter()
    for row in movies:
        movie, data, attributes = int(row['movieId']), snapshot[int(row['movieId'])], []
        for genre in row['genres'].split('|'):
            if genre and genre != '(no genres listed)':
                require(genre in rules['genres'], 'unknown_genre', f'Map the MovieLens genre {genre}')
                attributes.append('genre::' + rules['genres'][genre])
        match = re.search(r'\((\d{4})\)\s*$', row['title'])
        if match and rules['min_year'] <= int(match.group(1)) <= rules['max_year']:
            attributes.append(f'era::{int(match.group(1)) // 10 * 10}s')
        runtime = data.get('runtime')
        if isinstance(runtime, (int, float)) and not isinstance(runtime, bool) and runtime > 0:
            attributes.append('runtime::' + runtime_bin(runtime, rules['runtime_bins']))
        language = data.get('original_language') or None
        language = None if language in rules['unknown_languages'] else language
        origins = sorted({c['iso_3166_1'] for c in data.get('production_countries') or [] if c.get('iso_3166_1')})
        languages.update([language] if language else [])
        countries.update(origins)
        prepared[movie] = attributes, language, origins
    top_languages = set(sorted(languages, key=lambda x: (-languages[x], x))[:rules['top_languages']])
    top_origins = set(sorted(countries, key=lambda x: (-countries[x], x))[:rules['top_origins']])
    catalog = {}
    for movie, (attributes, language, origins) in prepared.items():
        attributes = attributes + (['language::' + language] if language in top_languages else []) + ['origin::' + c for c in origins if c in top_origins]
        catalog[str(movie)] = sorted(set(attributes))
    write_parquet(output / 'catalog.parquet', (dict(movieId=m, attributes=catalog[m]) for m in sorted(catalog)), SCHEMA)
    atoms = sorted({a for attributes in catalog.values() for a in attributes})
    write_json(output / 'schema.json', dict(domain='movie', schema_version=rules['schema_version'], kb_version=rules['kb_version'],
                                            evidence_atoms=atoms, catalog_atoms=atoms))
    manifest = dict(status='movie_catalog_built', movies=len(catalog), directions=len(atoms),
                    directions_by_aspect=dict(sorted(Counter(a.split('::', 1)[0] for a in atoms).items())),
                    movies_by_aspect={aspect: sum(any(a.startswith(aspect + '::') for a in attributes) for attributes in catalog.values())
                                      for aspect in ('genre', 'era', 'runtime', 'language', 'origin')},
                    top_languages=sorted(top_languages), top_origins=sorted(top_origins), user_data_read=False)
    write_json(output / 'manifest.json', manifest)
    return manifest
