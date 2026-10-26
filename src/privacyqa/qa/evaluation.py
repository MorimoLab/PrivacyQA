"""Scoring of selector outputs against complete answer sets; a malformed output selects nothing."""
from collections import defaultdict
import json

from privacyqa.common import iter_rows, require, write_json, write_rows

METRICS = ('format_valid', 'precision', 'recall', 'f1', 'set_exact', 'selected_count', 'outside_candidate_count',
           'selection_error', 'noise_at_5', 'noise_at_10', 'noise_at_25')


def selected_codes(value):
    """The selected codes of a JSON output, and whether the output is valid."""
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        return [], False
    valid = (isinstance(parsed, dict) and set(parsed) == {'selected_candidate_codes'} and isinstance(parsed['selected_candidate_codes'], list)
             and all(isinstance(c, str) and c for c in parsed['selected_candidate_codes']))
    if not valid:
        return [], False
    codes = parsed['selected_candidate_codes']
    return codes, len(codes) == len(set(codes))


def score_one(gold, prediction):
    target = set(gold['selected_candidate_codes'])
    mapping = gold['candidate_item_ids']
    require(target and target <= mapping.keys() and len(set(mapping.values())) == len(mapping)
            and {mapping[c] for c in target} == set(gold['complete_answer_ids']), 'invalid_gold', 'Gold selects exactly the complete answers')
    codes, valid = selected_codes(prediction['output'])
    if not valid:
        codes = []
    chosen = set(codes)
    hits = len(chosen & target)
    row = dict(qa_key=gold['qa_key'], evaluation_track=gold['evaluation_track'], answer_count=len(target),
               condition_count=len(gold['evidence_constraints']) + len(gold['catalog_constraints']), format_valid=int(valid),
               precision=hits / len(chosen) if chosen else 0.0, recall=hits / len(target), f1=2 * hits / (len(chosen) + len(target)),
               set_exact=int(valid and chosen == target), outside_candidate_count=len(chosen - mapping.keys()),
               selected_count=len(chosen), false_positive_count=len(chosen - target), selection_error=int(not valid or bool(chosen - target)))
    for cutoff in (5, 10, 25):
        row['noise_at_' + str(cutoff)] = int(bool(set(codes[:cutoff]) - target))
    return row


def aggregate(rows):
    return dict(queries=len(rows), **{name: sum(r[name] for r in rows) / len(rows) if rows else None for name in METRICS})


def evaluate_predictions(gold_path, predictions_path, output):
    gold_rows, predictions = list(iter_rows(gold_path, 'jsonl')), list(iter_rows(predictions_path, 'jsonl'))
    gold, pred = {r['qa_key']: r for r in gold_rows}, {r['qa_key']: r for r in predictions}
    require(len(gold) == len(gold_rows) > 0 and len(pred) == len(predictions) and set(gold) == set(pred)
            and all(set(r) == {'qa_key', 'output'} for r in predictions), 'prediction_alignment',
            'One raw output per question, failures included')
    rows = [score_one(g, pred[k]) for k, g in gold.items()]
    tracks, answers, complexity = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        tracks[row['evaluation_track']].append(row)
        n = row['answer_count']
        answers['1' if n == 1 else '2-5' if n <= 5 else '6-10' if n <= 10 else '11-25' if n <= 25 else '26-100'].append(row)
        complexity[str(row['condition_count'])].append(row)
    report = dict(status='predictions_evaluated', overall=aggregate(rows), tracks={k: aggregate(v) for k, v in tracks.items()},
                  answer_size={k: aggregate(v) for k, v in answers.items()}, condition_count={k: aggregate(v) for k, v in complexity.items()})
    write_rows(output / 'per_query.jsonl', rows)
    write_json(output / 'evaluation.json', report)
    return report
