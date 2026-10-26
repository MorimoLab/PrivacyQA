"""Common-test comparison of all model runs: scores recomputed from the saved outputs, with paired intervals that
resample whole answer-set classes (bootstrap clusters), conditional on the fitted models and the single DP release."""
from collections import defaultdict
import math

import numpy as np

from privacyqa.common import DOMAINS, iter_rows, read_json, require, write_json
from privacyqa.qa.evaluation import aggregate, score_one
from privacyqa.qa.train import SOURCES

METRICS = ('f1', 'set_exact')


def cluster_bootstrap(values, classes, repetitions, seed):
    """Resample whole answer classes, keeping every question of a class together."""
    groups = defaultdict(list)
    for i, answer in enumerate(classes):
        groups[tuple(answer)].append(i)
    require(len(values) == len(classes) > 0 and repetitions > 0, 'invalid_bootstrap', 'Use aligned question outcomes')
    indices = list(groups.values())
    totals = np.stack([values[g].sum(axis=0) for g in indices])
    counts = np.asarray([len(g) for g in indices])
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(repetitions):
        selected = rng.integers(0, len(indices), size=len(indices))
        result.append(totals[selected].sum(axis=0) / counts[selected].sum())
    return np.stack(result), len(indices)


def scored_outputs(folder, gold):
    """Scores recomputed from the saved raw outputs, checked against the saved evaluation."""
    report = read_json(folder / 'manifest.json')
    predictions = list(iter_rows(folder / 'predictions.jsonl', 'jsonl'))
    pred = {p['qa_key']: p for p in predictions}
    require(len(pred) == len(predictions) == len(gold) and set(pred) == {g['qa_key'] for g in gold}, 'prediction_alignment',
            'One saved output per question')
    scored = [score_one(g, pred[g['qa_key']]) for g in gold]
    replayed, saved = aggregate(scored), report['evaluation']
    require(replayed.keys() == saved.keys() and replayed['queries'] == saved['queries']
            and all(math.isclose(v, saved[k], rel_tol=0.0, abs_tol=1e-12) for k, v in replayed.items() if k != 'queries'),
            'score_replay_mismatch', 'Recomputed scores equal the saved evaluation')
    return scored, report


def analyze_qa(config, output):
    plan = config['qa']
    data = config.finished('qa-data', 'qa_data_prepared')
    domains, adapters, steps, predictions = {}, 0, 0, 0
    for domain in DOMAINS:
        gold = list(iter_rows(config.stage('qa-data') / domain / 'common' / 'test.gold.jsonl', 'jsonl'))
        values, records = [], {}
        for method in SOURCES:
            folder = config.stage('qa-models') / f'{domain}-{method}'
            scored, report = scored_outputs(folder, gold)
            settings = report['configuration']
            require(report['status'] == 'model_completed' and report['evaluated_queries'] == len(gold)
                    and (settings['domain'], settings['method'], settings['seed']) == (domain, method, plan['seed'])
                    and settings['training_token_budget'] == (0 if method == 'Base' else plan['training_token_budget'])
                    and read_json(folder / 'schedule.json')['test_keys'] == [g['qa_key'] for g in gold],
                    'model_run_mismatch', 'Every run answers the complete test under the planned seed and budget')
            values.append([[r[m] for m in METRICS] for r in scored])
            records[method] = dict(evaluation=report['evaluation'], training_steps=report['training_steps'], token_budget=report['token_budget'],
                                   seconds=report['seconds'], training_seconds=report['training_seconds'],
                                   output_token_limit_reached=report['output_token_limit_reached'])
            adapters += report['training_steps'] > 0
            steps += report['training_steps']
            predictions += len(gold)
        values = np.transpose(np.asarray(values), (1, 0, 2))
        boot, classes = cluster_bootstrap(values, [g['complete_answer_ids'] for g in gold], plan['bootstrap_replicates'], plan['seed'])
        point = values.mean(axis=0)
        for i, method in enumerate(SOURCES):
            records[method]['intervals'] = {m: np.quantile(boot[:, i, j], [.025, .975]).tolist() for j, m in enumerate(METRICS)}
        paired = []
        for left, right in plan['paired_comparisons']:
            i, k = SOURCES.index(left), SOURCES.index(right)
            paired.append(dict(left=left, right=right, metrics={m: dict(difference=float(point[i, j] - point[k, j]),
                               interval=np.quantile(boot[:, i, j] - boot[:, k, j], [.025, .975]).tolist()) for j, m in enumerate(METRICS)}))
        domains[domain] = dict(queries=len(gold), answer_classes=classes, methods=records, paired=paired)
    report = dict(status='qa_analysis_completed', domains=domains, construction_users=data['construction_users'],
                  model_runs=len(DOMAINS) * len(SOURCES), trained_adapters=adapters, training_steps=steps, predictions=predictions,
                  bootstrap_replicates=plan['bootstrap_replicates'], bootstrap_unit='complete_answer_class')
    write_json(output / 'manifest.json', report)
    return report
