"""Unit tests of the construction, mechanism, attack and QA building blocks on fictional inputs."""
from collections import Counter
import json
import random

import numpy as np
import pytest
from scipy import sparse

from privacyqa.attacks.features import calibrate_threshold, score_metrics
from privacyqa.banks import length_quotas, rating_variants, recipe_ingredient_draw
from privacyqa.catalog import Catalog, ingredient_atom
from privacyqa.cohorts import GROUPS, Knowledge, check_groups, select_groups, shadow_schedule
from privacyqa.common import ContractError
from privacyqa.evidence import partial_history, positive_evidence
from privacyqa.mechanism import B, Q, SCALE, Ledger, add_noise, threshold
from privacyqa.qa.evaluation import score_one
from privacyqa.qa.train import generation_batches, sample_schedule
from privacyqa.release import release_masses
from privacyqa.roles import ROLES, assign_role
from privacyqa.supports import Aggregate

RULES = dict(max_cores=1024, core_lengths=[1, 2, 3], public_seed=1, rating_min_core_matches=2, target_answers=20, answer_upper=25,
             answer_lower=dict(movie=2, beauty=2, recipe_text=1), max_options=4, max_conditions=10)


def error(code, function):
    with pytest.raises(ContractError) as caught:
        function()
    assert caught.value.code == code


def test_length_quotas_pass_unused_quota_on():
    assert length_quotas({1: 335, 2: 20034, 3: 445577}, 1024) == {1: 335, 2: 345, 3: 344}
    assert length_quotas({1: 58, 2: 1325, 3: 13000}, 1024) == {1: 58, 2: 483, 3: 483}
    assert length_quotas({1: 2, 2: 3, 3: 1}, 1024) == {1: 2, 2: 3, 3: 1}


def test_rating_variants_keep_the_options_nearest_to_the_target_within_range():
    rows = [dict(movieId=str(i), attributes=a) for i, a in enumerate(
        [['genre::a', 'origin::x']] * 3 + [['genre::a', 'origin::y']] * 30 + [['genre::a']] * 5 + [['genre::b', 'origin::x']])]
    catalog = Catalog(rows, 'movie', ['genre::a', 'genre::b', 'origin::x', 'origin::y'])
    # Options: alone 38 answers, +origin::y 30, +origin::x 3; only the last lies within 2 to 25.
    assert [(extra, len(answers)) for extra, answers in rating_variants(catalog, 'movie', ('genre::a',), RULES)] == [(('origin::x',), 3)]


def test_catalog_executes_every_constraint_and_ingredients_only_in_recipe():
    rows = [dict(item_id='1', attributes=['taste::sweet'], ingredients=['milk', 'sugar']),
            dict(item_id='2', attributes=['taste::sweet'], ingredients=['sugar']), dict(item_id='3', attributes=['taste::sour'], ingredients=[])]
    catalog = Catalog(rows, 'recipe_text', ['taste::sweet', 'taste::sour'])
    assert catalog.execute(['taste::sweet']) == ('1', '2')
    assert catalog.execute(['taste::sweet'], [ingredient_atom('exclude', 'milk')]) == ('2',)
    assert catalog.execute(['taste::sweet'], [ingredient_atom('include', 'milk')]) == ('1',)
    error('unmapped_catalog_atom', lambda: catalog.execute(['taste::bitter']))
    error('ingredient_is_catalog_only', lambda: catalog.execute([ingredient_atom('include', 'milk')]))
    error('unknown_ingredient', lambda: catalog.execute(['taste::sweet'], [ingredient_atom('include', 'salt')]))


def test_random_recipe_draw_likes_come_from_the_recipe_and_dislikes_from_the_rest():
    rng = random.Random(3)
    for _ in range(50):
        likes, dislikes = recipe_ingredient_draw(frozenset('abcdef'), frozenset('abcdefghij'), rng)
        assert set(likes) <= set('abcdef') and set(dislikes) <= set('ghij') and 1 <= len(likes) <= 5 and 1 <= len(dislikes) <= 3
    assert recipe_ingredient_draw(frozenset('ab'), frozenset('ab'), rng) is None


def test_contributions_are_bounded_per_user():
    aggregate = Aggregate(30)
    small = np.zeros(30, dtype=np.int64)
    small[:3] = 1
    large = np.ones(30, dtype=np.int64)
    assert list(aggregate.add(small)[:3]) == [Q] * 3
    bounded = aggregate.add(large)
    assert set(bounded) == {B * Q // 30} and bounded.sum() <= B * Q
    assert aggregate.metrics()['users_clipped'] == 1 and aggregate.metrics()['clipping_loss'] == 10
    error('nonbinary_support', lambda: aggregate.add(2 * small))


def test_threshold_of_the_reported_bank_dimensions():
    # Main and random banks of the reported run have 3,066 and 2,888 units in all.
    assert threshold(3066, '1/20')['threshold_integer'] == 57796783
    assert threshold(2888, '1/20')['threshold_integer'] == 57483209
    assert SCALE == 5 * Q


def test_noise_and_ledger(tmp_path):
    noisy, manifest = add_noise([Q, 0, 5 * Q])
    assert len(noisy) == 3 and all(type(v) is int for v in noisy) and manifest['epsilon'] == 4 and manifest['delta'] == 0
    ledger = Ledger(tmp_path / 'ledger.sqlite', [('r1', 'movie'), ('r2', 'movie')])
    ledger.mark('r1', 'measuring')
    ledger.mark('r1', 'complete')
    assert ledger.totals() == dict(measurements=2, statuses={'complete': 1, 'reserved': 1}, sequential_epsilon=8, delta=0)
    error('unreserved_measurement', lambda: ledger.mark('r3', 'complete'))
    ledger.close()
    error('ledger_exists', lambda: Ledger(tmp_path / 'ledger.sqlite', []))


def test_role_assignment_is_fixed_by_the_salt():
    proportions = dict(zip(ROLES, (0.2, 0.5, 0.1, 0.1, 0.1)))
    roles = [assign_role('movie', str(u), 'fictional-salt-0001', proportions) for u in range(20000)]
    shares = Counter(roles)
    assert all(abs(shares[r] / 20000 - p) < 0.02 for r, p in proportions.items())
    assert roles == [assign_role('movie', str(u), 'fictional-salt-0001', proportions) for u in range(20000)]
    assert roles != [assign_role('movie', str(u), 'fictional-salt-0002', proportions) for u in range(20000)]


def test_cohorts_from_roles_and_from_a_file():
    roles = {f'u{i}': ROLES[i % 5] for i in range(500)}
    plan = dict(members=10, test_nonmembers=10, calibration_nonmembers=10, reference_users=10, selection_seed=1)
    groups = select_groups(roles, 'movie', plan)
    assert check_groups(json.loads(json.dumps(groups)), roles, 'movie', plan) == groups
    assert len(groups['train_pool']) == 20 and all(roles[u] == 'attack_reference' for u in groups['train_pool'] + groups['calibration_pool'])
    swapped = dict(groups, target_members=groups['target_nonmembers'], target_nonmembers=groups['target_members'])
    error('cohort_role_mismatch', lambda: check_groups(swapped, roles, 'movie', plan))


def test_shadow_releases_pair_complementary_halves():
    groups = {g: [f'{g}{i}' for i in range(n)] for g, n in zip(GROUPS, (8, 8, 4, 4, 4))}
    schedule = shadow_schedule(groups, dict(training_releases=4, calibration_releases=2, targets_per_class=2, schedule_seed=9), 'movie')
    assert [r['release_id'] for r in schedule][-1] == 'movie-test' and len(schedule) == 7
    for left, right in zip(schedule[0:6:2], schedule[1:6:2]):
        assert sorted(left['members'] + right['members']) == sorted(set(left['members'] + right['members']))
        assert len(left['members']) == len(right['members']) == 4 and left['target_positions'] == right['target_positions']
        assert [a + b for a, b in zip(left['labels'], right['labels'])] == [1] * 4
    assert schedule[-1]['members'] == list(range(16, 20)) and schedule[-1]['labels'] == [1] * 4 + [0] * 4 + [-1] * 4


def test_knowledge_rows_follow_the_requested_targets():
    known = Knowledge.from_matches([np.array([0, 2]), np.array([], dtype=np.int64), np.array([1])], np.array([3, 0, 1], dtype=np.int16))
    taken = known.take([2, 0])
    assert taken.rows.tolist() == [1, 1, 0] and taken.units.tolist() == [0, 2, 1] and taken.observations.tolist() == [1, 3]


def test_partial_histories_are_judged_alone():
    records = [('u', 'a', 5.0), ('u', 'b', 1.0), ('u', 'c', 4.0)]
    rows = [dict(movieId=x, attributes=['genre::g']) for x in 'abc']
    catalog = Catalog(rows, 'movie', ['genre::g'])
    assert positive_evidence('movie', records, catalog) == {'a', 'c'}
    known, count = partial_history('movie', 'u', records, 'seed', 2)
    assert count == 2 and known <= {'a', 'b', 'c'}
    reviews = [('u', 'r1', 'taste::sweet', 'support'), ('u', 'r1', 'taste::sour', 'support'), ('u', 'r1', 'taste::sour', 'oppose'), ('u', 'r2', None, None)]
    assert positive_evidence('recipe_text', reviews, None) == [{'taste::sweet'}, set()]
    contexts, count = partial_history('recipe_text', 'u', reviews, 'seed', 5)
    assert count == 2 and sorted(map(sorted, contexts)) == [[], ['taste::sweet']]


def test_release_masses_replace_member_targets_by_shadow_members():
    raw = sparse.csr_matrix(np.array([[1, 0], [1, 1], [0, 1], [1, 0]], dtype=np.int64))
    bounded = sparse.csr_matrix(raw.multiply(Q).astype(np.int64))
    statistics = dict(raw_mass=[7, 6], bounded_mass=[7 * Q, 6 * Q])
    schedule = [dict(release_id='s', split='train', members=[0, 1]), dict(release_id='t', split='test', members=[2, 3])]
    masses = release_masses(statistics, raw, bounded, schedule, 10)
    assert masses['t'] == statistics and masses['s'] == dict(raw_mass=[8, 6], bounded_mass=[8 * Q, 6 * Q])
    error('invalid_background', lambda: release_masses(dict(raw_mass=[0, 6], bounded_mass=[0, 6 * Q]), raw, bounded, schedule, 10))


def test_attack_metrics_count_ties_and_calibrate_on_nonmembers():
    assert score_metrics(np.array([[1., 1., 1., 1.]]), np.array([1, 0, 1, 0]), [None])['auc'] == 0.5
    result = score_metrics(np.array([[3., 2., 1., 0.]]), np.array([1, 1, 0, 0]), [1.5])
    assert result == dict(auc=1.0, actual_fpr=0.0, tpr_at_frozen_threshold=1.0, advantage=1.0)
    calibration = calibrate_threshold([float(i) for i in range(1000)])
    assert calibration['threshold'] == 989.0 and calibration['calibration_fpr'] == 0.01
    assert calibrate_threshold([1.0] * 10)['status'] == 'insufficient_calibration_nonmembers'


def test_scoring_requires_valid_json_and_complete_answers():
    gold = dict(qa_key='q', evaluation_track='test', selected_candidate_codes=['C001', 'C002'], candidate_item_ids={'C001': 'a', 'C002': 'b', 'C003': 'c'},
                complete_answer_ids=['a', 'b'], evidence_constraints=['x::y'], catalog_constraints=[])
    exact = score_one(gold, dict(output='{"selected_candidate_codes": ["C002", "C001"]}'))
    assert exact['set_exact'] == 1 and exact['f1'] == 1
    partial = score_one(gold, dict(output='{"selected_candidate_codes": ["C001", "C003"]}'))
    assert partial['precision'] == 0.5 and partial['recall'] == 0.5 and partial['set_exact'] == 0
    malformed = score_one(gold, dict(output='C001, C002'))
    assert malformed['format_valid'] == 0 and malformed['f1'] == 0


def test_training_schedule_and_inference_batches():
    sampled, tokens = sample_schedule([0.5, 0.5], [10, 30], 100, 1, 50)
    assert tokens >= 100 and tokens - 30 < 100 and set(sampled) <= {0, 1}
    error('step_limit', lambda: sample_schedule([1.0], [1], 100, 1, 5))
    batches = list(generation_batches([[0] * n for n in (5, 50, 6, 7)], 2, 60))
    assert sorted(i for b in batches for i in b) == [0, 1, 2, 3] and [1] in batches


def test_paper_configuration_plans_the_reported_measurements():
    from privacyqa.config import Config
    from tests.fixtures import REPO
    config = Config.load(REPO / 'configs' / 'paper.json')
    release = config['release']
    assert release['measurements'] == 2 * 3 * (release['training_releases'] + release['calibration_releases'] + 1) == 294
    assert sum(config['roles']['proportions'].values()) == pytest.approx(1)
    assert config['private'] == dict(role_salt='<ROLE_SALT>', knowledge_seed='<KNOWLEDGE_SEED>', cohort_file='<COHORT_FILE>')
    assert config.file(config['recipe_search_module']).is_file()
    assert all(config.file(config['domains'][d]['schema']).is_file() for d in ('movie', 'beauty', 'recipe_text'))
