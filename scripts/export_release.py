"""Export the released datasets of a run into data/, keeping only whitelisted fields.

Exports the DP target release of the main bank with its training records, the common test and development sets, both
request banks, and the public catalogs and schemas. Ablations, shadow releases, user data and attack data are never
read. Item-level fields that third-party terms do not allow to redistribute go only to --restricted-out: the
TMDB-derived Movie attributes with every Movie field that depends on them, and the Recipe1M-derived Recipe answers and
candidates. See data/README.md.

Usage: python scripts/export_release.py --config CONFIG --out data [--restricted-out DIR]
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from privacyqa.common import iter_rows, read_json  # noqa: E402
from privacyqa.config import Config  # noqa: E402

NAMES = {'movie': 'movie', 'beauty': 'beauty', 'recipe_text': 'recipe'}
PUBLIC_ITEMS = {'beauty'}  # Domains whose item-level fields are released.
MOVIELENS_ASPECTS = ('genre', 'era')  # Movie aspects that come from MovieLens alone.
RELEASE_FIELDS = ('id', 'domain', 'constraints', 'evidence_constraints', 'catalog_constraints', 'answer_count', 'weight', 'schema_version', 'kb_version')
GOLD_FIELDS = ('selected_candidate_codes', 'candidate_item_ids', 'complete_answer_ids', 'evidence_constraints', 'catalog_constraints',
               'evaluation_track', 'schema_version', 'kb_version')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return len(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write('\n')


def question_side(bank):
    return dict(bank, candidates=[{**{k: v for k, v in c.items() if k != 'answers'}, 'answer_count': len(c['answers'])} for c in bank['candidates']])


def export_domain(config, domain, public, restricted):
    root = config.stage('qa-data')
    release = read_json(root / f'{domain}-sources.json')['DP']
    records = list(iter_rows(root / domain / 'DP' / 'train.jsonl', 'jsonl'))
    constraints = list(iter_rows(root / domain / 'DP' / 'constraints.jsonl', 'jsonl'))
    assert len(records) == len(constraints) and all(len(r['selected_candidate_codes']) == c['answer_count'] for r, c in zip(records, constraints))
    splits = {}
    for name in ('development', 'test'):
        inputs = list(iter_rows(root / domain / 'common' / f'{name}.inputs.jsonl', 'jsonl'))
        gold = {g['qa_key']: g for g in iter_rows(root / domain / 'common' / f'{name}.gold.jsonl', 'jsonl')}
        assert len(gold) == len(inputs) and all(i['qa_key'] in gold for i in inputs)
        splits[name] = [(i, gold[i['qa_key']]) for i in inputs]

    def release_rows(items):
        return [{**{k: v[k] for k in RELEASE_FIELDS}, **({'answers': v['answers']} if items else {})} for v in release]

    def train_rows(items):
        rows = []
        for record, constraint in zip(records, constraints):
            row = dict(qa_key=record['qa_key'], domain=record['domain'], constraint_id=constraint['id'], question=record['question'],
                       evidence_constraints=constraint['evidence_constraints'], catalog_constraints=constraint['catalog_constraints'],
                       answer_count=constraint['answer_count'], sampling_weight=record['sampling_weight'])
            if items:
                row.update(template_id=record['template_id'], candidates=record['candidates'],
                           selected_candidate_codes=record['selected_candidate_codes'], answers=constraint['answers'])
            rows.append(row)
        return rows

    def split_rows(pairs, items):
        rows = []
        for inputs, gold in pairs:
            row = dict(qa_key=inputs['qa_key'], domain=inputs['domain'], question=inputs['question'], evidence_constraints=gold['evidence_constraints'],
                       catalog_constraints=gold['catalog_constraints'], answer_count=len(gold['complete_answer_ids']), evaluation_track=gold['evaluation_track'])
            if items:
                row.update(template_id=inputs['template_id'], candidates=inputs['candidates'], **{k: gold[k] for k in GOLD_FIELDS})
            rows.append(row)
        return rows

    def write_all(base, items):
        folder, suffix = base / NAMES[domain], '' if items else '.questions'
        counts = dict(release=write_jsonl(folder / 'release' / f'dp_release{suffix}.jsonl', release_rows(items)),
                      train=write_jsonl(folder / 'release' / f'train{suffix}.jsonl', train_rows(items)),
                      dev=write_jsonl(folder / 'release' / f'dev{suffix}.jsonl', split_rows(splits['development'], items)),
                      test=write_jsonl(folder / 'release' / f'test{suffix}.jsonl', split_rows(splits['test'], items)))
        for arm in ('main', 'random'):
            bank = read_json(config.stage('banks') / f'{domain}-{arm}.json')
            write_json(folder / 'banks' / f'{arm}{suffix}.json', bank if items else question_side(bank))
        return counts

    counts = {'public': write_all(public, domain in PUBLIC_ITEMS)}
    if domain not in PUBLIC_ITEMS and restricted is not None:
        counts['restricted'] = write_all(restricted, True)
    return counts


def export_catalogs(config, public, restricted):
    spec = config['domains']
    beauty = public / 'beauty' / 'catalog'
    beauty.mkdir(parents=True, exist_ok=True)
    pq.write_table(pq.read_table(config.file(spec['beauty']['catalog']), columns=['parent_asin', 'attributes']), beauty / 'catalog.parquet')
    for domain, folder in (('beauty', beauty), ('movie', public / 'movie' / 'catalog'), ('recipe_text', public / 'recipe' / 'schema')):
        write_json(folder / 'schema.json', read_json(config.file(spec[domain]['schema'])))
    table = pq.read_table(config.file(spec['movie']['catalog']), columns=['movieId', 'attributes'])
    keep = [[a for a in attributes if a.split('::', 1)[0] in MOVIELENS_ASPECTS] for attributes in table.column('attributes').to_pylist()]
    schema = pa.schema([('movieId', pa.string()), ('attributes', pa.list_(pa.string()))])
    pq.write_table(pa.table({'movieId': table.column('movieId').to_pylist(), 'attributes': keep}, schema=schema),
                   public / 'movie' / 'catalog' / 'catalog_movielens.parquet')
    if restricted is not None:
        folder = restricted / 'movie' / 'catalog'
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.file(spec['movie']['catalog']), folder / 'catalog.parquet')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--restricted-out', type=Path)
    args = parser.parse_args()
    config = Config.load(args.config)
    export_catalogs(config, args.out, args.restricted_out)
    print(json.dumps({d: export_domain(config, d, args.out, args.restricted_out) for d in NAMES}, indent=1))


if __name__ == '__main__':
    main()
