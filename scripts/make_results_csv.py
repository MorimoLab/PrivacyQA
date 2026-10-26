"""Write the aggregate result tables (results/results_*.csv) from the stage manifests of a run.

Usage: python scripts/make_results_csv.py --config CONFIG --out results
Reads only aggregate manifests; writes no history, identifier, per-user value or support count.
"""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from privacyqa.common import DOMAINS, read_json  # noqa: E402
from privacyqa.config import Config  # noqa: E402

MAIN = ('Public-only', 'Non-private', 'Bounded-non-private', 'DP')
ORDER = ('Base', *MAIN, *(f'Random-{m}' for m in MAIN))
SUMMARY = ('domain,construction,method,comparison_cohort_users,privacy_scope,qa_train_records,qa_test_questions,qa_precision,qa_recall,qa_f1,'
           'qa_f1_ci95_low,qa_f1_ci95_high,qa_exact_match,qa_exact_match_ci95_low,qa_exact_match_ci95_high,qa_format_success_rate,reference_users,'
           'reference_user_coverage,reference_mass_coverage,reference_tv_raw,reference_tv_clipped,attack_view,attack_model,attack_population,'
           'attack_test_members,attack_test_nonmembers,attack_calibration_nonmembers,attack_auc,attack_ap,attack_eer,attack_pauc_0_01_standardized,'
           'attack_precision_top_1pct,attack_calibration_fpr_target,attack_true_positives,attack_false_positives,attack_tpr,attack_fpr,'
           'attack_precision,attack_ppv_prior_10pct,attack_ppv_prior_1pct,attack_advantage_pp,attack_workpoint_status,qa_train_steps,'
           'qa_input_tokens,qa_supervised_tokens,qa_distinct_sampled_records,qa_repeated_samples,qa_train_minutes,qa_train_and_eval_minutes,'
           'qa_training_seed,qa_fit_count,dp_epsilon_per_call,dp_delta').split(',')
COMPARISONS = 'stage,domain,left_method,right_method,metric,difference_pp,ci_low_pp,ci_high_pp,ci_scope,multiplicity'.split(',')
AUDIT = ('analysis,arm,domain,method,view,attack,stratum,members,nonmembers,auc,tp_0_1pct,fp_0_1pct,tpr_0_1pct,fpr_0_1pct,tp_1pct,fp_1pct,'
         'tpr_1pct,fpr_1pct,auc_placebo_mean,auc_placebo_min,auc_placebo_max,individual_epsilon_max').split(',')
PREFERENCE = 'test,domain,item,metric,weighting,estimate,ci_low,ci_high,n'.split(',')
SYNTHETIC = ('design,mechanism,canary_units,contribution_l1,claimed_epsilon,true_epsilon,statistic,trials_per_hypothesis,evaluation_positive,'
             'evaluation_negative,event_positive,event_negative,epsilon_point,epsilon_lower_95').split(',')
ATTACK_PAIRS = {'full_vs_unbounded': ('DP', 'Non-private'), 'full_vs_bounded': ('DP', 'Bounded-non-private'),
                'full_vs_random_raw': ('DP', 'Random-Non-private'), 'full_vs_random_dp': ('DP', 'Random-DP')}
# Largest individual privacy loss in each group of supported units k: min(k, B) / B * epsilon at the group's largest k (B = 20, epsilon = 4).
SUPPORT_EPSILON = {'support_0': 0, 'support_1': 0.2, 'support_2-4': 0.8, 'support_5-9': 1.8, 'support_10-19': 3.8, 'support_20+': 4.0}


DATASETS = 'quantity,bank,movie,beauty,recipe_text'.split(',')


def f(x, digits):
    return '' if x is None else f'{x:.{digits}f}'


def datasets(config):
    """Dataset sizes (Table 3) and the draws of the preference-weighted test set."""
    stage = lambda name: read_json(config.stage(name) / 'manifest.json')
    banks, data, fidelity, preference = stage('banks'), stage('qa-data'), stage('qa-fidelity'), stage('preference-data')
    tests = read_json(config.stage('banks') / 'common_test.json')
    row = lambda quantity, bank, value: dict(quantity=quantity, bank=bank, **{d: value(d) for d in DOMAINS})
    rows = [row('construction_users', '', lambda d: data['construction_users'][d]), row('reference_users', '', lambda d: fidelity['domains'][d]['users'])]
    for quantity, value in (('support_units', lambda d, arm: banks['domains'][d][arm]['executable_units']),
                            ('request_variants', lambda d, arm: banks['domains'][d][arm]['variants']),
                            ('dp_retained_variants', lambda d, arm: data['domains'][d]['sources'][source(arm)]['released_records']),
                            ('dp_training_records', lambda d, arm: data['domains'][d]['sources'][source(arm)]['training_records']),
                            ('dp_distinct_constraint_sets', lambda d, arm: data['domains'][d]['sources'][source(arm)]['unique_questions'])):
        rows.extend(row(quantity, arm, lambda d: value(d, arm)) for arm in ('main', 'random'))
    sizes = lambda d, key: [len(r[key]) for r in tests[d]]
    rows += [row('test_questions', 'main', lambda d: data['domains'][d]['test_requests']),
             row('validation_questions', 'main', lambda d: data['domains'][d]['development_requests']),
             row('candidates_per_test_question', 'main', lambda d: data['domains'][d]['candidate_count']),
             row('constraints_per_test_question_min', 'main', lambda d: min(sizes(d, 'constraints'))),
             row('constraints_per_test_question_max', 'main', lambda d: max(sizes(d, 'constraints'))),
             row('answers_per_test_question_min', 'main', lambda d: min(sizes(d, 'answers'))),
             row('answers_per_test_question_max', 'main', lambda d: max(sizes(d, 'answers'))),
             row('answers_per_test_question_mean', 'main', lambda d: f(sum(sizes(d, 'answers')) / len(tests[d]), 2)),
             row('preference_draws', 'main', lambda d: preference['domains'][d]['draws']),
             row('preference_successful_draws', 'main', lambda d: preference['domains'][d]['successful_draws']),
             row('preference_failed_unit_probability', 'main', lambda d: f(preference['domains'][d]['failed_unit_preference_mass'], 4)),
             row('preference_draws_with_unit_in_dp_training', 'main', lambda d: f(preference['domains'][d]['unit_in_training_draw_share']['DP'], 4))]
    return rows


def source(arm):
    return 'DP' if arm == 'main' else 'Random-DP'


def tables(config):
    stage = lambda name: read_json(config.stage(name) / 'manifest.json')
    analysis, fidelity, metrics, audit = stage('qa-analysis'), stage('qa-fidelity'), stage('attack-metrics'), stage('attack-audit')
    data, preference, synthetic = stage('qa-data'), stage('preference-analysis'), stage('mechanism-audit')
    members = config['cohorts']['members']
    models = {(d, m): read_json(config.stage('qa-models') / f'{d}-{m}' / 'manifest.json') for d in DOMAINS for m in ORDER}

    def primary(domain, method):
        arm, name = ('random', method[7:]) if method.startswith('Random-') else ('main', method)
        return next(x for x in metrics['results'] if x['arm'] == arm and x['domain'] == domain and x['method'] == name
                    and x['view'] == 'C+weight' and x['attack'] == 'unit_classifier' and x['stratum'] == 'all')['metrics']

    summary = []
    for domain in DOMAINS:
        for method in ORDER:
            report, evaluation = models[domain, method], analysis['domains'][domain]['methods'][method]
            e, interval, budget = evaluation['evaluation'], evaluation['intervals'], report['token_budget']
            trained = method != 'Base'
            row = dict(domain=domain, construction='none' if not trained else ('random' if method.startswith('Random-') else 'main'), method=method,
                       comparison_cohort_users=data['construction_users'][domain],
                       privacy_scope='user_level_epsilon_4_delta_0_per_release' if method.endswith('DP') else '',
                       qa_train_records=data['domains'][domain]['sources'][method]['training_records'] if trained else 0, qa_test_questions=e['queries'],
                       qa_precision=f(e['precision'], 4), qa_recall=f(e['recall'], 4), qa_f1=f(e['f1'], 4), qa_f1_ci95_low=f(interval['f1'][0], 4),
                       qa_f1_ci95_high=f(interval['f1'][1], 4), qa_exact_match=f(e['set_exact'], 4), qa_exact_match_ci95_low=f(interval['set_exact'][0], 4),
                       qa_exact_match_ci95_high=f(interval['set_exact'][1], 4), qa_format_success_rate=f(e['format_valid'], 4))
            if trained:
                r, m = fidelity['domains'][domain]['sources'][method], primary(domain, method)
                alerts = round((m['tpr_1pct'] + m['fpr_1pct']) * members)
                row.update(reference_users=fidelity['domains'][domain]['users'], reference_user_coverage=f(r['reference_user_coverage'], 4),
                           reference_mass_coverage=f(r['reference_probability_mass_covered'], 4), reference_tv_raw=f(r['weight_TV_raw_reference'], 4),
                           reference_tv_clipped=f(r['weight_TV_bounded_reference'], 4), attack_view='C+weight', attack_model='per_unit_classifier',
                           attack_population='overall', attack_test_members=members, attack_test_nonmembers=config['cohorts']['test_nonmembers'],
                           attack_calibration_nonmembers=config['cohorts']['calibration_nonmembers'], attack_auc=f(m['auc'], 6),
                           attack_ap=f(m['average_precision'], 6), attack_eer=f(m['eer'], 6), attack_pauc_0_01_standardized=f(m['pauc_1pct'], 6),
                           attack_precision_top_1pct=f(m['precision_top_1pct'], 6), attack_calibration_fpr_target='0.01',
                           attack_true_positives=round(m['tpr_1pct'] * members), attack_false_positives=round(m['fpr_1pct'] * members),
                           attack_tpr=f(m['tpr_1pct'], 4), attack_fpr=f(m['fpr_1pct'], 4), attack_precision=f(m['precision_1pct'], 6),
                           attack_ppv_prior_10pct=f(m['ppv_prior_10pct_1pct'], 6), attack_ppv_prior_1pct=f(m['ppv_prior_1pct_1pct'], 6),
                           attack_advantage_pp=f(100 * m['advantage_1pct'], 4),
                           attack_workpoint_status='measured' if alerts else 'no_alerts_precision_ppv_undefined')
            else:
                row.update(attack_workpoint_status='not_applicable')
            row.update(qa_train_steps=report['training_steps'], qa_input_tokens=budget.get('actual_training_tokens', '') if trained else '',
                       qa_supervised_tokens=budget.get('supervised_tokens', '') if trained else '',
                       qa_distinct_sampled_records=budget.get('sampled_unique', '') if trained else '',
                       qa_repeated_samples=budget.get('repetitions', '') if trained else '',
                       qa_train_minutes=f(report['training_seconds'] / 60, 1) if trained else '', qa_train_and_eval_minutes=f(report['seconds'] / 60, 1),
                       qa_training_seed=report['configuration']['seed'] if trained else '', qa_fit_count=int(trained),
                       dp_epsilon_per_call=4 if method.endswith('DP') else '', dp_delta=0 if method.endswith('DP') else '')
            summary.append(row)

    comparisons = []
    for domain in DOMAINS:
        for p in analysis['domains'][domain]['paired']:
            for metric, name in (('f1', 'qa_f1'), ('set_exact', 'qa_exact_match')):
                x = p['metrics'][metric]
                comparisons.append(dict(stage='qa', domain=domain, left_method=p['left'], right_method=p['right'], metric=name,
                                        difference_pp=f'{100 * x["difference"]:+.4f}', ci_low_pp=f'{100 * x["interval"][0]:+.4f}',
                                        ci_high_pp=f'{100 * x["interval"][1]:+.4f}', ci_scope='95pct_conditional_answer_class_cluster_bootstrap',
                                        multiplicity='not_adjusted'))
    for domain in DOMAINS:
        for name, (left, right) in ATTACK_PAIRS.items():
            p = next(x for x in metrics['paired'] if x['domain'] == domain and x['comparison'] == name and x['view'] == 'C+weight'
                     and x['attack'] == 'unit_classifier' and x['stratum'] == 'all')
            for metric, key in (('auc', 'attack_auc'), ('average_precision', 'attack_ap'), ('advantage_1pct', 'attack_advantage_pp')):
                low, high = p['simultaneous'][metric]['interval']
                comparisons.append(dict(stage='attack', domain=domain, left_method=left, right_method=right, metric=key,
                                        difference_pp=f"{100 * p['differences'][metric]:+.3f}", ci_low_pp=f'{100 * low:+.3f}', ci_high_pp=f'{100 * high:+.3f}',
                                        ci_scope='approximate_simultaneous_conditional_paired_user_bootstrap',
                                        multiplicity='per_domain_7_comparisons_x_10_metrics_alpha_over_3'))

    def attack_row(analysis_name, arm, domain, method, view, attack, x, extra=None):
        return dict(analysis=analysis_name, arm=arm, domain=domain, method=method, view=view, attack=attack, stratum=x['stratum'], members=x['members'],
                    nonmembers=x['nonmembers'], auc=f(x['auc'], 6), tp_0_1pct=x['0_1pct']['tp'], fp_0_1pct=x['0_1pct']['fp'],
                    tpr_0_1pct=f(x['0_1pct']['tpr'], 6), fpr_0_1pct=f(x['0_1pct']['fpr'], 6), tp_1pct=x['1pct']['tp'], fp_1pct=x['1pct']['fp'],
                    tpr_1pct=f(x['1pct']['tpr'], 6), fpr_1pct=f(x['1pct']['fpr'], 6), **(extra or {}))

    audit_rows = []
    for d in audit['domains']:
        arm, domain = d['arm'], d['domain']
        for x in [o for o in audit['matches'] if o['arm'] == arm and o['domain'] == domain and o['view'] == 'C+weight' and o['attack'] == 'unit_classifier']:
            for stratum in ('known_match', 'no_known_match'):
                s = x['strata'][stratum]
                audit_rows.append(dict(analysis='primary_attack_strata', arm=arm, domain=domain, method=x['method'], view='C+weight', attack='unit_classifier',
                                       stratum=stratum, members=s['members'], nonmembers=s['nonmembers'], auc=f(s['auc'], 6), tp_1pct=x[f'tp_{stratum}'],
                                       fp_1pct=x[f'fp_{stratum}'], tpr_1pct=f(s['tpr_at_frozen_threshold'], 6), fpr_1pct=f(s['actual_fpr'], 6)))
        audit_rows.append(dict(analysis='history_only_cv', arm=arm, domain=domain, attack='logistic_history_only', stratum='all',
                               auc=f(d['blind']['cross_validated_auc'], 6)))
        for x in d['reference']:
            audit_rows.append(attack_row('reference_attack', arm, domain, x['method'], x['view'], 'reference_attack', x))
        for x in d['informed']:
            audit_rows.append(attack_row('informed_attack', arm, domain, x['method'], x['view'], 'informed_attack', x,
                                         dict(individual_epsilon_max=f(SUPPORT_EPSILON[x['stratum']], 4) if x['stratum'] in SUPPORT_EPSILON else '')))
        for x in d['placebo']:
            a = x['actual']
            audit_rows.append(dict(analysis='placebo', arm=arm, domain=domain, method=x['method'], view=x['view'], attack=x['attack'], stratum='all',
                                   members=a['members'], nonmembers=a['nonmembers'], auc=f(a['auc'], 6), tp_0_1pct=a['0_1pct']['tp'],
                                   fp_0_1pct=a['0_1pct']['fp'], tpr_0_1pct=f(a['0_1pct']['tpr'], 6), fpr_0_1pct=f(a['0_1pct']['fpr'], 6),
                                   tp_1pct=a['1pct']['tp'], fp_1pct=a['1pct']['fp'], tpr_1pct=f(a['1pct']['tpr'], 6), fpr_1pct=f(a['1pct']['fpr'], 6),
                                   auc_placebo_mean=f(x['auc_mean'], 6), auc_placebo_min=f(x['auc_min'], 6), auc_placebo_max=f(x['auc_max'], 6)))

    preference_rows = []
    for domain, x in preference['domains'].items():
        r = x['replay']
        preference_rows.append(dict(test='replay', domain=domain, item=r['method'], metric='identical_outputs', weighting='question',
                                    estimate=r['identical_outputs'], n=r['compared']))
        for method, m in x['methods'].items():
            for metric in ('f1', 'set_exact', 'format_valid'):
                preference_rows.append(dict(test='preference', domain=domain, item=method, metric=metric, weighting='draw', estimate=m[metric],
                                            ci_low=m['intervals'][metric][0], ci_high=m['intervals'][metric][1], n=x['draws']))
                u = x['unweighted_unique_questions'][method]
                preference_rows.append(dict(test='preference', domain=domain, item=method, metric=metric, weighting='unique_question',
                                            estimate=u[metric], n=u['questions']))
            for track, t in m['tracks'].items():
                preference_rows.append(dict(test='preference', domain=domain, item=method, metric='f1', weighting=f'draw:{track}',
                                            estimate='' if t['f1'] is None else t['f1'], n=t['draws']))
        for p in x['paired']:
            for metric, v in p['metrics'].items():
                preference_rows.append(dict(test='preference', domain=domain, item=f"{p['left']} minus {p['right']}", metric=metric, weighting='draw',
                                            estimate=v['difference'], ci_low=v['interval'][0], ci_high=v['interval'][1], n=x['draws']))
    synthetic_rows = [{k: r[k] for k in SYNTHETIC} for r in synthetic['results']]
    return dict(summary=(SUMMARY, summary), comparisons=(COMPARISONS, comparisons), audit=(AUDIT, audit_rows), preference=(PREFERENCE, preference_rows),
                synthetic_audit=(SYNTHETIC, synthetic_rows), datasets=(DATASETS, datasets(config)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    for name, (fields, rows) in tables(Config.load(args.config)).items():
        with (args.out / f'results_{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='raise')
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, '') for k in fields})
        print(name, len(rows))


if __name__ == '__main__':
    main()
