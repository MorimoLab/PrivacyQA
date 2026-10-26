"""Trained membership attacks on every view of every release procedure.

Per bank, domain, procedure and view, an aggregate classifier and a per-unit classifier are fitted on the targets of the
shadow-training releases; with an untrained matching score, they score the targets of the target release. Each attack's
threshold is calibrated to a 1% false-positive rate on the calibration nonmembers.
"""
import numpy as np
from scipy import sparse

from privacyqa.attacks.features import (ATTACKS, KnownIndex, PublishedView, calibrate_threshold, classifier_scores,
                                        fit_classifier, score_metrics, unit_features)
from privacyqa.banks import load_banks
from privacyqa.cohorts import Knowledge, load_cohorts
from privacyqa.common import ARMS, DOMAINS, VIEWS, load_arrays, progress, require, save_arrays, write_json
from privacyqa.release import METHODS, load_releases, read_view


def evaluate_scores(scores, labels, known_matches):
    """Calibrated metrics on members (1) and test nonmembers (0), overall and by whether the partial history matches a unit."""
    test = labels >= 0
    y = labels[test]
    result = {}
    for attack, values in scores.items():
        calibration = calibrate_threshold(values[labels == -1].tolist())
        cutoff = [calibration['threshold']]
        strata = {}
        for name, mask in (('known_match', known_matches[test] > 0), ('no_known_match', known_matches[test] == 0)):
            strata[name] = dict(members=int(sum((y == 1) & mask)), nonmembers=int(sum((y == 0) & mask)),
                                **score_metrics(values[test][None, mask], y[mask], cutoff))
        result[attack] = dict(metrics=score_metrics(values[test][None], y, cutoff), calibration=calibration, strata=strata)
    return result


def run_attacks(config, output):
    plan = config['attack']
    banks = load_banks(config)
    _, schedules = load_cohorts(config)
    releases = load_releases(config)
    results, models = [], 0
    for arm in ARMS:
        for domain in DOMAINS:
            index = KnownIndex(banks[arm][domain])
            known = Knowledge.load(config.stage('cohorts') / arm / f'{domain}-known.npz')
            items = {r['release_id']: r for r in schedules[domain]}
            records = releases['records'][arm][domain]
            splits = {split: [r for r in records if r['split'] == split] for split in ('train', 'calibration', 'test')}
            require(len(splits['train']) == config['release']['training_releases'] and len(splits['test']) == 1,
                    'invalid_release_records', 'Use every shadow-training release and the one target release')
            target = items[splits['test'][0]['release_id']]

            def features(record, method, view):
                entry = record['views'][method]
                published = PublishedView(index, read_view(config, entry, view), entry['release_id'], view)
                return unit_features(published, known.take(items[record['release_id']]['target_positions']))

            for method in METHODS:
                for view in VIEWS:
                    train = [features(record, method, view) for record in splits['train']]
                    x, names = sparse.vstack([t[0] for t in train], format='csr'), train[0][1]
                    truth = np.concatenate([items[r['release_id']]['labels'] for r in splits['train']])
                    widths = dict(aggregate_classifier=len(names), unit_classifier=x.shape[1])
                    fitted = {name: fit_classifier(x[:, :width], truth, plan) for name, width in widths.items()}
                    models += len(fitted)
                    folder = output / arm / domain / method / view
                    folder.mkdir(parents=True, mode=0o700)
                    for name, model in fitted.items():
                        write_json(folder / f'{name}.json', model)
                    test, _ = features(splits['test'][0], method, view)
                    labels = np.asarray(target['labels'])
                    scores = {name: classifier_scores(model, test[:, :widths[name]]) for name, model in fitted.items()}
                    scores['matching'] = test[:, names.index('matched_weight' if view == 'C+weight' else 'match_fraction')].toarray().ravel()
                    known_matches = test[:, 1].toarray().ravel()
                    save_arrays(folder / 'target_scores.npz', **scores, labels=labels, positions=np.asarray(target['target_positions']),
                                known_matches=known_matches)
                    result = dict(arm=arm, domain=domain, method=method, view=view, attacks=evaluate_scores(scores, labels, known_matches))
                    write_json(folder / 'metrics.json', result)
                    results.append(result)
                    progress(output, 'attack_evaluated', arm=arm, domain=domain, method=method, view=view)
    report = dict(status='attacks_completed', plan=plan, attacks=list(ATTACKS), results=results, attack_models=models)
    write_json(output / 'manifest.json', report)
    return report


def target_scores(config, arm, domain, method, view):
    values = load_arrays(config.stage('attacks') / arm / domain / method / view / 'target_scores.npz')
    return {k: values[k] for k in (*ATTACKS, 'labels', 'positions', 'known_matches')}
