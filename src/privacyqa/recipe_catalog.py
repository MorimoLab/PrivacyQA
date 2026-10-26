"""The Recipe catalog of RecipeADKG: the tags and ingredients of every recipe that RecipeADQA answers with, and the schema.

Built from RecipeADQA resources: the answer index (`constraint_answer_index.sqlite`, with the tables answers, tags and
answer_tags), the recipe identifier index (`recipe-uri-index.tsv`), the ingredient lists of the tagged Recipe1M recipes
(`recipe1m_v5_tagged_recipes.json`), and the aspect-direction and tag schemas. The recipe records derive from Recipe1M,
which only its licensees may hold, so the catalog stays in the private run root. Review directions can form preference
cores; recipe tags can be constraints.
"""
import sqlite3

import duckdb
import pyarrow as pa

from privacyqa.catalog import load_recipe_search
from privacyqa.common import atoms, iter_rows, read_json, require, write_json, write_parquet
from privacyqa.evidence import batches, literal

SCHEMA = pa.schema([('item_id', pa.string()), ('attributes', pa.list_(pa.string())), ('ingredients', pa.list_(pa.string()))])
CATALOG = '''SELECT a.item_id, coalesce(t.attributes, []::VARCHAR[]), coalesce(i.ingredients, []::VARCHAR[]) FROM answers a
    LEFT JOIN (SELECT item_id, list(DISTINCT tag_id ORDER BY tag_id) AS attributes FROM answer_tags GROUP BY item_id) t USING(item_id)
    LEFT JOIN (SELECT recipe_uri, list(DISTINCT ingredient ORDER BY ingredient) AS ingredients FROM ingredients JOIN recipe_uris USING(recipe_id)
               GROUP BY recipe_uri) i USING(recipe_uri)
    ORDER BY a.item_id'''


def count(db, sql):
    return db.execute(sql).fetchone()[0]


def build_recipe_catalog(config, output):
    spec = config['domains']['recipe_text']
    sources, rules = spec['sources'], read_json(config.file(spec['rules']))
    require(set(rules) == {'schema_version', 'kb_version'}, 'invalid_recipe_rules', 'Declare the schema and catalog versions')
    evidence = atoms([f"{row['aspect']}::{row['direction']}" for row in iter_rows(config.file(sources['direction_schema']), 'tsv')])
    tags = [row['tag_id'] for row in iter_rows(config.file(sources['tag_schema']), 'tsv')]
    require(len(set(tags)) == len(tags), 'duplicate_tag', 'Tag identifiers are unique')
    tags = atoms(tags)
    search = load_recipe_search(config.file(config['recipe_search_module']))
    index = sqlite3.connect(config.file(sources['answer_index']).as_uri() + '?mode=ro', uri=True)
    with duckdb.connect() as db:
        for table, query, columns in (('answers', 'SELECT cast(answer_id AS text), recipe_uri FROM answers', ('item_id', 'recipe_uri')),
                                      ('tags', 'SELECT tag_id FROM tags', ('tag_id',)),
                                      ('answer_tags', 'SELECT cast(answer_id AS text), tag_id FROM answer_tags', ('item_id', 'tag_id'))):
            db.execute(f"CREATE TABLE {table}({', '.join(c + ' VARCHAR' for c in columns)})")
            for rows in batches(index.execute(query)):
                db.register('batch', pa.table({c: [row[i] for row in rows] for i, c in enumerate(columns)}))
                db.execute(f'INSERT INTO {table} SELECT * FROM batch')
                db.unregister('batch')
        index.close()
        db.register('schema_tags', pa.table({'tag_id': list(tags)}))
        require(count(db, 'SELECT count(*) FROM (SELECT tag_id FROM tags EXCEPT SELECT tag_id FROM schema_tags)') == 0
                and count(db, 'SELECT count(*) FROM (SELECT tag_id FROM schema_tags EXCEPT SELECT tag_id FROM tags)') == 0
                and count(db, 'SELECT count(*) FROM answer_tags ANTI JOIN answers USING(item_id)') == 0
                and count(db, 'SELECT count(*) FROM answer_tags ANTI JOIN tags USING(tag_id)') == 0,
                'invalid_answer_index', 'The answer index uses exactly the schema tags, and every tag edge has its recipe and tag')
        db.execute(f"""CREATE TABLE recipe_uris AS SELECT cast(recipe_id AS VARCHAR) AS recipe_id, recipe_uri
            FROM read_csv({literal(config.file(sources['recipe_uris']))}, header=true, delim='\t', all_varchar=true)""")
        require(count(db, 'SELECT count(*) FROM (SELECT recipe_id FROM recipe_uris GROUP BY recipe_id HAVING count(DISTINCT recipe_uri) > 1)') == 0,
                'recipe_uri_conflict', 'Every recipe identifier has one recipe URI')
        db.execute('CREATE TABLE records(recipe_id VARCHAR)')
        db.execute('CREATE TABLE ingredients(recipe_id VARCHAR, ingredient VARCHAR)')
        records, ingredients = [], []

        def flush():
            for table, rows, columns in (('records', records, ('recipe_id',)), ('ingredients', ingredients, ('recipe_id', 'ingredient'))):
                if rows:
                    db.register('batch', pa.table({c: [r[i] for r in rows] for i, c in enumerate(columns)}))
                    db.execute(f'INSERT INTO {table} SELECT * FROM batch')
                    db.unregister('batch')
                    rows.clear()

        for recipe in search.iter_json_array(config.file(sources['recipes'])):
            require(isinstance(recipe.get('id'), str) and recipe['id'], 'invalid_recipe_record', 'Every recipe record has its Recipe1M identifier')
            records.append((recipe['id'],))
            ingredients.extend((recipe['id'], name) for name in (str(i.get('name', '')).strip() for i in recipe.get('ingredients', [])) if name)
            if len(records) >= 4096:
                flush()
        flush()
        require(count(db, 'SELECT count(*) FROM records ANTI JOIN recipe_uris USING(recipe_id)') == 0
                and count(db, 'SELECT count(*) FROM answers ANTI JOIN (SELECT DISTINCT recipe_uri FROM recipe_uris JOIN records USING(recipe_id)) USING(recipe_uri)') == 0,
                'unmatched_recipe_records', 'Every recipe record has a URI, and every answer has its recipe record')
        rows = db.execute(CATALOG).fetchall()
    write_parquet(output / 'catalog.parquet', (dict(item_id=i, attributes=a, ingredients=n) for i, a, n in rows), SCHEMA)
    write_json(output / 'schema.json', dict(domain='recipe_text', schema_version=rules['schema_version'], kb_version=rules['kb_version'],
                                            evidence_atoms=list(evidence), catalog_atoms=list(tags)))
    manifest = dict(status='recipe_catalog_built', recipes=len(rows), evidence_directions=len(evidence), catalog_tags=len(tags),
                    directions_in_both=len(set(evidence) & set(tags)), recipes_without_tags=sum(not a for _, a, _ in rows),
                    recipes_without_ingredients=sum(not n for _, _, n in rows))
    write_json(output / 'manifest.json', manifest)
    return manifest
