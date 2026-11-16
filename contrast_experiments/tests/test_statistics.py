import json

import pytest

from stent_repro import statistics


def case(case_id, entity_id, dice=None, status='ok'):
    return {'case_id': case_id, 'entity_id': entity_id, 'status': status, 'dice': dice}


def test_repeated_acquisitions_get_one_entity_weight():
    rows = [case('repeat1', 'stent1', 0.2), case('repeat2', 'stent1', 0.4),
            case('repeat3', 'stent1', 0.6), case('other', 'stent2', 0.8)]
    summary = statistics.summarize_entities(rows, metrics=('dice',))

    assert summary['case_count'] == 4
    assert summary['entity_count'] == 2
    assert summary['metrics']['dice']['n_entities'] == 2
    assert summary['metrics']['dice']['n_cases'] == 4
    assert summary['metrics']['dice']['mean'] == pytest.approx(0.6)


def test_pair_by_acquisition_id_and_never_compare_missing_repeat():
    reference = [case('a', 'stent1', 0.2), case('b', 'stent1', 0.1),
                 case('c', 'stent2', 0.6), case('d', 'stent3', 0.5)]
    candidate = [case('f', 'stent4', 0.7), case('c', 'stent2', 0.4),
                 case('a', 'stent1', 0.6), case('e', 'stent1', 0.9)]
    report = statistics.paired_entity_bootstrap(reference, candidate, metrics=('dice',))
    result = report['metrics']['dice']

    assert result['n_entities'] == 2
    assert result['n_cases'] == 2
    assert result['mean_difference'] == pytest.approx(0.1)
    assert result['ci_lower'] == pytest.approx(-0.2)
    assert result['ci_upper'] == pytest.approx(0.4)
    assert result['valid_case_ids'] == ['a', 'c']
    assert report['reference']['missing_case_ids'] == ['e', 'f']
    assert report['candidate']['missing_case_ids'] == ['b', 'd']
    assert report['reference']['expected_case_count'] == 6
    assert report == statistics.paired_entity_bootstrap(
        reference[::-1], candidate[::-1], metrics=('dice',))


def test_failures_accumulate_without_imputing_missing_metrics():
    reference = [case('a', 'stent1', 0.9, 'pipeline_failure'),
                 case('b', 'stent2', 0.0, 'empty_prediction')]
    candidate = [case('a', 'stent1', 0.8), case('b', 'stent2', 0.6)]
    report = statistics.paired_entity_bootstrap(reference, candidate, metrics=('dice',))
    result = report['metrics']['dice']

    assert report['reference']['failure_count'] == 2
    assert report['reference']['failure_rate'] == 1.0
    assert result['n_entities'] == 1
    assert result['valid_case_ids'] == ['b']
    assert result['mean_difference'] == 0.6
    assert result['ci_lower'] is result['ci_upper'] is None


@pytest.mark.parametrize('rows', [[], [case('a', 'stent1', None, 'pipeline_failure')]])
def test_no_valid_entities_has_no_estimate_or_interval(rows):
    report = statistics.paired_entity_bootstrap(rows, rows, metrics=('dice',))
    result = report['metrics']['dice']

    assert result['n_entities'] == result['n_cases'] == 0
    assert result['mean_difference'] is result['ci_lower'] is result['ci_upper'] is None
    assert report['reference']['failure_rate'] == (1.0 if rows else None)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(('reference', 'candidate', 'message'), [
    ([case('a', 'e1'), case('a', 'e1')], [], 'duplicate case'),
    ([case('a', 'e1')], [case('a', 'e2')], 'entity_id'),
    ([case('a', '')], [], 'entity_id'),
    ([{'case_id': 'a', 'status': 'ok'}], [], 'entity_id'),
])
def test_reject_ambiguous_pairing_identity(reference, candidate, message):
    with pytest.raises(ValueError, match=message):
        statistics.paired_entity_bootstrap(reference, candidate, metrics=('dice',))


def test_fixed_seed_average_uses_same_acquisition_support_across_all_seeds():
    reference = {
        11: [case('a', 'e1', 0.1), case('b', 'e1', 0.2), case('c', 'e2', 0.6)],
        22: [case('a', 'e1', 0.4), case('b', 'e1', 0.5), case('c', 'e2', 0.8)],
    }
    candidate = {
        11: [case('a', 'e1', 0.3), case('b', 'e1', 1.0), case('c', 'e2', 0.8)],
        22: [case('a', 'e1', 0.8), case('b', 'e1', None, 'pipeline_failure'), case('c', 'e2', 0.5)],
    }
    report = statistics.compare_training_seeds(reference, candidate, metrics=('dice',))
    fixed = report['fixed_seed_average']['metrics']['dice']

    assert report['seeds'] == [11, 22]
    assert report['per_seed']['11']['metrics']['dice']['mean_difference'] == pytest.approx(0.35)
    assert report['per_seed']['22']['metrics']['dice']['mean_difference'] == pytest.approx(0.05)
    assert report['per_seed']['22']['candidate']['failure_count'] == 1
    assert fixed['valid_case_ids'] == ['a', 'c']
    assert fixed['n_entities'] == fixed['n_cases'] == 2
    assert fixed['mean_difference'] == pytest.approx(0.125)
    assert fixed['ci_lower'] == pytest.approx(-0.05)
    assert fixed['ci_upper'] == pytest.approx(0.30)


def test_case_entity_identity_must_agree_across_training_seeds():
    runs = {11: [case('a', 'e1')], 22: [case('a', 'e2')]}
    with pytest.raises(ValueError, match='entity_id'):
        statistics.compare_training_seeds(runs, runs)


def test_missing_run_preserves_frozen_denominator():
    rows = [case('a', 'e1', 0.7)]
    report = statistics.compare_training_seeds({11: rows, 22: rows}, {11: rows, 22: []}, metrics=('dice',))

    missing_run = report['per_seed']['22']['candidate']
    assert missing_run['expected_case_count'] == 1
    assert missing_run['missing_case_ids'] == ['a']
    assert missing_run['failure_count'] == 0
    assert missing_run['unavailable_case_count'] == 1
    assert report['fixed_seed_average']['metrics']['dice']['n_entities'] == 0


def test_cannot_silently_drop_an_unmatched_training_seed():
    with pytest.raises(ValueError, match='seed'):
        statistics.compare_training_seeds({1: [], 2: []}, {1: []})


def test_explicit_expected_roster_preserves_cases_missing_from_both_arms():
    report = statistics.paired_entity_bootstrap(
        [], [], expected_case_entities={'a': 'e1'}, metrics=('dice',))

    assert report['expected_entity_count'] == 1
    assert report['candidate']['unavailable_case_rate'] == 1.0
    assert report['reference']['missing_case_ids'] == ['a']


def test_explicit_expected_roster_rejects_unknown_cases():
    with pytest.raises(ValueError, match='expected roster'):
        statistics.paired_entity_bootstrap(
            [case('b', 'e1')], [], expected_case_entities={'a': 'e1'})
