"""The complete pipeline on fictional data, from the evidence index to the preference-weighted test."""
import csv
import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from privacyqa.__main__ import main
from privacyqa.banks import load_banks
from privacyqa.common import ARMS, DOMAINS, VIEWS, iter_rows, read_json, write_json, write_rows
from privacyqa.config import Config
from privacyqa.qa.evaluation import evaluate_predictions
from privacyqa.qa.train import SOURCES
from privacyqa.release import METHODS
from tests.fixtures import REPO, make_config

STAGES = ('build-index', 'assign-roles', 'build-banks', 'freeze-cohorts', 'build-supports', 'release', 'attack', 'attack-metrics',
          'attack-audit', 'mechanism-audit', 'qa-data', 'qa-fidelity')


def fabricate_runs(config, folder, items, manifest):
    """Model outputs without a model: each run answers with the gold codes of every other question and nothing otherwise."""
    for name, gold, extra in items:
        output = config.stage(folder) / name
        output.mkdir(parents=True)
        write_rows(output / 'gold.jsonl', gold)
        write_rows(output / 'predictions.jsonl', (dict(qa_key=g['qa_key'], output=json.dumps(
            dict(selected_candidate_codes=g['selected_candidate_codes'] if i % 2 else []))) for i, g in enumerate(gold)))
        evaluation = evaluate_predictions(output / 'gold.jsonl', output / 'predictions.jsonl', output)
        write_json(output / 'schedule.json', dict(test_keys=[g['qa_key'] for g in gold]))
        write_json(output / 'manifest.json', dict(manifest, evaluation=evaluation['overall'], evaluated_queries=len(gold), **extra))


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp('fictional')
    path = make_config(root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr('privacyqa.attacks.features.MIN_CALIBRATION', 8)  # The fixture has eight calibration nonmembers.
        for stage in STAGES:
            if stage == 'release':
                assert main(['release', '--config', str(path), '--dry-run']) == 0
            assert main([stage, '--config', str(path)]) == 0, stage
        config = Config.load(path)
        qa = config['qa']
        for domain in DOMAINS:
            gold = list(iter_rows(config.stage('qa-data') / domain / 'common' / 'test.gold.jsonl', 'jsonl'))
            fabricate_runs(config, 'qa-models', [(f'{domain}-{m}', gold, dict(
                configuration=dict(domain=domain, method=m, seed=qa['seed'], training_token_budget=0 if m == 'Base' else qa['training_token_budget']),
                training_steps=0 if m == 'Base' else 3)) for m in SOURCES],
                dict(status='model_completed', token_budget={}, seconds=1.0, training_seconds=0.5, output_token_limit_reached=0))
        assert main(['qa-analysis', '--config', str(path)]) == 0
        assert main(['preference-data', '--config', str(path)]) == 0
        for domain in DOMAINS:
            gold = list(iter_rows(config.stage('preference-data') / domain / 'test.gold.jsonl', 'jsonl'))
            fabricate_runs(config, 'preference-models', [(f'{domain}-{m}', gold, dict(mode='preference', domain=domain, method=m)) for m in METHODS],
                           dict(status='inference_completed', complete_test=True, seconds=1.0, output_token_limit_reached=0))
            replay = config.stage('preference-models') / f'replay-{domain}-DP'
            replay.mkdir()
            write_json(replay / 'manifest.json', dict(status='replay_completed', domain=domain, method='DP', compared=3, identical_outputs=3,
                                                      identical_scores=3, original_f1=0.5, replayed_f1=0.5))
        assert main(['preference-analysis', '--config', str(path)]) == 0
    return config


def test_index_roles_and_banks(run):
    index = read_json(run.stage('index') / 'manifest.json')['domains']
    roles = read_json(run.stage('roles') / 'manifest.json')['roles']
    assert all(sum(roles[d].values()) == index[d]['users'] for d in DOMAINS) and index['movie']['users'] == 700
    banks = load_banks(run)
    for domain in DOMAINS:
        for arm in ARMS:
            bank = banks[arm][domain]
            lower = run['banks']['answer_lower'][domain]
            assert 0 < len(bank.units) <= run['banks']['max_cores']
            assert all(lower <= len(c.answers) <= 25 and 1 <= len(c.evidence_constraints) <= 3 for c in bank.candidates)
        assert len(banks['random'][domain].units) <= len(banks['main'][domain].units)
        assert not any('skin_type::all' in c.evidence_constraints for c in banks['main']['beauty'].candidates)
    tests = read_json(run.stage('banks') / 'common_test.json')
    assert all(len(tests[d]) == 12 and {r['id'] for r in tests[d]} <= set(banks['main'][d].by_id) for d in DOMAINS)


def test_cohorts_and_schedule(run):
    cohorts, schedules = read_json(run.stage('cohorts') / 'cohorts.json'), read_json(run.stage('cohorts') / 'schedules.json')
    for domain in DOMAINS:
        groups = cohorts[domain]
        assert len(groups['train_pool']) == len(groups['calibration_pool']) == 16 and len(groups['reference_users']) == 20
        schedule = schedules[domain]
        assert [r['split'] for r in schedule] == ['train'] * 4 + ['calibration'] * 2 + ['test']
        for left, right in zip(schedule[0:6:2], schedule[1:6:2]):
            assert not set(left['members']) & set(right['members']) and left['target_positions'] == right['target_positions']
            assert [a + b for a, b in zip(left['labels'], right['labels'])] == [1] * 8
        assert schedule[-1]['labels'] == [1] * 8 + [0] * 8 + [-1] * 8


def test_releases_and_ledger(run):
    manifest = read_json(run.stage('releases') / 'manifest.json')
    assert manifest['ledger'] == dict(measurements=42, statuses={'complete': 42}, sequential_epsilon=168, delta=0)
    with sqlite3.connect(run.stage('releases') / 'ledger.sqlite') as db:
        assert db.execute('SELECT count(*) FROM measurements WHERE status="complete"').fetchone()[0] == 42
    banks = load_banks(run)
    plan = read_json(run.stage('release-plan') / 'plan.json')
    assert plan['thresholds'] == manifest['thresholds'] and not (run.stage('release-plan') / 'ledger.sqlite').exists()
    for arm in ARMS:
        for domain in DOMAINS:
            target = manifest['records'][arm][domain][-1]
            dp = read_json(run.stage('releases') / target['views']['DP']['directory'] / 'dp_mass.json')
            rows = read_json(run.stage('releases') / target['views']['DP']['directory'] / 'C+weight.json')
            expected = banks[arm][domain].reconstruct(dp['integer_mass'], dp['release_id'], manifest['thresholds'][arm]['threshold_integer'])
            assert rows == expected['C+weight']
            public = read_json(run.stage('releases') / target['views']['Public-only']['directory'] / 'C-only.json')
            assert len(public) == len(banks[arm][domain].candidates)
            statistics = read_json(run.stage('supports') / f'{domain}-{arm}.json')
            raw = read_json(run.stage('releases') / target['views']['Non-private']['directory'] / 'C-only.json')
            assert {r['id'] for r in raw} == {k for unit, n in zip(banks[arm][domain].units, statistics['raw_mass']) if n for k in unit}


def test_attacks_and_audits(run):
    attacks = read_json(run.stage('attacks') / 'manifest.json')
    assert attacks['attack_models'] == 2 * len(ARMS) * len(DOMAINS) * len(METHODS) * len(VIEWS)
    for result in attacks['results']:
        if result['method'] == 'Public-only':  # A release that does not depend on users gives every target one score.
            assert result['attacks']['unit_classifier']['metrics']['auc'] == 0.5
    metrics = read_json(run.stage('attack-metrics') / 'manifest.json')
    assert set(metrics['simultaneous']) <= set(DOMAINS) and metrics['results']
    audit = read_json(run.stage('attack-audit') / 'manifest.json')
    assert all(all(d['view_checks'].values()) and all(d['frozen_models_replayed'].values()) for d in audit['domains'])
    synthetic = read_json(run.stage('mechanism-audit') / 'manifest.json')
    assert synthetic['mechanism_invocations'] == 8 and synthetic['user_data_read'] is False
    assert all(r['epsilon_lower_95'] >= 0 for r in synthetic['results'])


def test_qa_data_and_analyses(run):
    data = read_json(run.stage('qa-data') / 'manifest.json')
    for domain in DOMAINS:
        summary = data['domains'][domain]
        assert summary['test_requests'] == 12 and summary['candidate_count'] == 30 and set(summary['sources']) == set(SOURCES) - {'Base'}
        inputs = list(iter_rows(run.stage('qa-data') / domain / 'common' / 'test.inputs.jsonl', 'jsonl'))
        assert all(len(r['candidates']) == 30 and set(r) == {'qa_key', 'domain', 'question', 'candidates', 'template_id'} for r in inputs)
        for source, values in summary['sources'].items():
            rows = list(iter_rows(run.stage('qa-data') / domain / source / 'train.jsonl', 'jsonl'))
            assert len(rows) == values['training_records'] and (not rows or np.isclose(sum(r['sampling_weight'] for r in rows), 1))
    fidelity = read_json(run.stage('qa-fidelity') / 'manifest.json')
    assert all(0 <= v['reference_user_coverage'] <= 1 for d in DOMAINS for v in fidelity['domains'][d]['sources'].values())
    analysis = read_json(run.stage('qa-analysis') / 'manifest.json')
    assert analysis['model_runs'] == 27 and analysis['trained_adapters'] == 24
    assert all(np.isclose(analysis['domains'][d]['methods']['DP']['evaluation']['f1'], 0.5, atol=0.1) for d in DOMAINS)
    preference = read_json(run.stage('preference-analysis') / 'manifest.json')
    assert all(preference['domains'][d]['draws'] > 0 for d in DOMAINS)


def test_stages_never_overwrite(run):
    assert main(['build-banks', '--config', str(run.path)]) == 2


def test_result_tables_and_release_export(run, tmp_path):
    results, public, restricted = tmp_path / 'results', tmp_path / 'data', tmp_path / 'restricted'
    results.mkdir()
    for script, *options in (('make_results_csv.py', '--out', results), ('export_release.py', '--out', public, '--restricted-out', restricted)):
        subprocess.run([sys.executable, REPO / 'scripts' / script, '--config', run.path, *options], check=True, capture_output=True)
    tables = {p.stem[len('results_'):]: list(csv.DictReader(p.open(newline=''))) for p in results.glob('results_*.csv')}
    assert set(tables) == {'summary', 'comparisons', 'audit', 'preference', 'synthetic_audit', 'datasets'} and len(tables['summary']) == 27
    sizes = {(r['quantity'], r['bank']): r for r in tables['datasets']}
    assert all(sizes['test_questions', 'main'][d] == '12' for d in DOMAINS)
    for name, items in (('movie', False), ('beauty', True), ('recipe', False)):  # Movie and Recipe items go only to the restricted directory.
        rows = list(iter_rows(public / name / 'release' / ('test.jsonl' if items else 'test.questions.jsonl'), 'jsonl'))
        assert len(rows) == 12 and all(('candidates' in r) == ('complete_answer_ids' in r) == items for r in rows)
        assert (restricted / name).exists() != items
