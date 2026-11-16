"""Compare methods on matched acquisitions with entities as the resampling unit."""

from collections import Counter, defaultdict
from collections.abc import Mapping
from math import isfinite
from numbers import Real
from statistics import mean

import numpy as np

from stent_repro.evaluation import METRICS


# The evaluator can produce defined scores even with an empty mask or skeleton.
_MEASURED_STATUSES = frozenset({
    'ok', 'empty_prediction', 'empty_skeleton', 'empty_truth',
    'false_positive_negative_control',
})


def _identifier(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a nonempty string')
    return value


def _index_cases(rows):
    indexed = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError('each case row must be a mapping')
        case_id = _identifier(row.get('case_id'), 'case_id')
        _identifier(row.get('entity_id'), 'entity_id')
        _identifier(row.get('status'), 'status')
        if case_id in indexed:
            raise ValueError(f'duplicate case ID {case_id!r}')
        indexed[case_id] = row
    return indexed


def _roster(tables, expected_case_entities):
    explicit = expected_case_entities is not None
    if explicit and not isinstance(expected_case_entities, Mapping):
        raise ValueError('expected_case_entities must map case IDs to entity IDs')
    roster = {}
    if explicit:
        roster = {_identifier(case, 'case_id'): _identifier(entity, 'entity_id')
                  for case, entity in expected_case_entities.items()}
    for table in tables:
        for case_id, row in table.items():
            entity = row['entity_id']
            if explicit and case_id not in roster:
                raise ValueError(f'case {case_id!r} is absent from the expected roster')
            if roster.setdefault(case_id, entity) != entity:
                raise ValueError(f'entity_id mismatch for case {case_id!r}')
    return roster


def _metric_names(metrics):
    if isinstance(metrics, str):
        raise ValueError('metrics must be an iterable of metric names')
    names = tuple(_identifier(key, 'metric name') for key in metrics)
    if not names or len(set(names)) != len(names):
        raise ValueError('metrics must contain unique names')
    return names


def _values(table, metric):
    values = {}
    for case_id, row in table.items():
        value = row.get(metric)
        if value is None or row['status'] not in _MEASURED_STATUSES:
            continue
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f'metric {metric!r} for case {case_id!r} must be numeric')
        if isfinite(value):
            values[case_id] = float(value)
    return values


def _entity_means(values, roster):
    grouped = defaultdict(list)
    for case_id in sorted(values):
        grouped[roster[case_id]].append(values[case_id])
    return {entity: mean(grouped[entity]) for entity in sorted(grouped)}


def _summarize(table, roster, metrics):
    missing_cases = sorted(set(roster).difference(table))
    statuses = Counter(row['status'] for row in table.values())
    failures = len(table) - statuses.get('ok', 0)
    unavailable = failures + len(missing_cases)
    reported_entities = {row['entity_id'] for row in table.values()}
    failure_entities = {row['entity_id'] for row in table.values() if row['status'] != 'ok'}
    expected_entities = set(roster.values())
    report = {
        'case_count': len(table),
        'expected_case_count': len(roster),
        'entity_count': len(reported_entities),
        'expected_entity_count': len(expected_entities),
        'failure_count': failures,
        'failure_rate': failures / len(table) if table else None,
        'entity_any_failure_count': len(failure_entities),
        'status_counts': dict(sorted(statuses.items())),
        'missing_case_ids': missing_cases,
        'missing_entity_ids': sorted(expected_entities - reported_entities),
        'unavailable_case_count': unavailable,
        'unavailable_case_rate': unavailable / len(roster) if roster else None,
        'metrics': {},
    }
    for metric in metrics:
        values = _values(table, metric)
        entities = _entity_means(values, roster)
        report['metrics'][metric] = {
            'n_cases': len(values),
            'n_entities': len(entities),
            'mean': mean(entities.values()) if entities else None,
            'valid_case_ids': sorted(values),
            'entity_means': entities,
        }
    return report


def summarize_entities(rows, *, metrics=METRICS, expected_case_entities=None):
    """Average defined scores within an entity, then weight each entity equally.

    Each row needs explicit ``case_id``, ``entity_id`` and ``status`` strings.
    Non-ok statuses count as failures. Defined legacy segmentation scores can
    still be valid observations, for example zero Dice for an empty prediction.
    Pipeline failures and missing rows contribute no metric values.
    """
    table = _index_cases(rows)
    roster = _roster([table], expected_case_entities)
    return _summarize(table, roster, _metric_names(metrics))


def _new_rng(n_resamples, confidence, rng_seed):
    if type(n_resamples) is not int or n_resamples < 1:
        raise ValueError('n_resamples must be a positive integer')
    if not isinstance(confidence, Real) or not 0 < confidence < 1:
        raise ValueError('confidence must lie strictly between 0 and 1')
    if type(rng_seed) is not int or rng_seed < 0:
        raise ValueError('rng_seed must be a nonnegative integer')
    return np.random.default_rng(rng_seed)


def _bootstrap_interval(entity_values, n_resamples, confidence, rng):
    n_entities = len(entity_values)
    if n_entities < 2:
        return None, None
    values = np.asarray(list(entity_values.values()))
    estimates = np.empty(n_resamples)
    # Limit the temporary sample index array for larger held-out sets.
    batch_size = max(1, 1_000_000 // n_entities)
    for start in range(0, n_resamples, batch_size):
        end = min(start + batch_size, n_resamples)
        indices = rng.integers(n_entities, size=(end - start, n_entities))
        estimates[start:end] = values[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(estimates, [tail, 1 - tail])
    return float(lower), float(upper)


def _paired_metrics(pairs, roster, metrics, n_resamples, confidence, rng_seed):
    rng = _new_rng(n_resamples, confidence, rng_seed)
    results = {}
    for metric in metrics:
        refs = [_values(reference, metric) for reference, _ in pairs]
        candidates = [_values(candidate, metric) for _, candidate in pairs]
        common = set(roster).intersection(*refs, *candidates)
        # Using the same acquired cases in every seed makes the seed average
        # independent of changes in missing repeats between training runs.
        differences = {
            case_id: mean(candidate[case_id] - reference[case_id]
                          for reference, candidate in zip(refs, candidates))
            for case_id in common
        }
        entities = _entity_means(differences, roster)
        lower, upper = _bootstrap_interval(entities, n_resamples, confidence, rng)
        results[metric] = {
            'n_cases': len(common),
            'n_entities': len(entities),
            'mean_difference': mean(entities.values()) if entities else None,
            'ci_lower': lower,
            'ci_upper': upper,
            'valid_case_ids': sorted(common),
            'entity_differences': entities,
        }
    return {
        'expected_case_count': len(roster),
        'expected_entity_count': len(set(roster.values())),
        'difference_direction': 'candidate_minus_reference',
        'resampling_unit': 'entity',
        'confidence': float(confidence),
        'n_resamples': n_resamples,
        'rng_seed': rng_seed,
        'metrics': results,
    }


def _paired_report(reference, candidate, roster, metrics, n_resamples, confidence, rng_seed):
    report = _paired_metrics([(reference, candidate)], roster, metrics,
                             n_resamples, confidence, rng_seed)
    report['reference'] = _summarize(reference, roster, metrics)
    report['candidate'] = _summarize(candidate, roster, metrics)
    return report


def paired_entity_bootstrap(reference, candidate, *, metrics=METRICS, n_resamples=2000,
                            confidence=0.95, rng_seed=0, expected_case_entities=None):
    """Percentile interval for candidate-minus-reference on common case support.

    Pair case IDs within each entity and average their differences. Resample
    entity means with replacement. A single entity gives only a point estimate.
    An explicit roster lets the report identify cases absent from both arms.
    """
    ref_table = _index_cases(reference)
    candidate_table = _index_cases(candidate)
    roster = _roster([ref_table, candidate_table], expected_case_entities)
    return _paired_report(ref_table, candidate_table, roster, _metric_names(metrics),
                          n_resamples, confidence, rng_seed)


def compare_training_seeds(reference_by_seed, candidate_by_seed, *, metrics=METRICS,
                           n_resamples=2000, confidence=0.95, rng_seed=0,
                           expected_case_entities=None):
    """Per-seed and fixed-seed average intervals on paired independent entities.

    Both arms supply the same nonempty set of integer seed keys. Use an empty
    case list to report a missing run. The fixed-seed average uses case IDs
    with defined metric values in every seed and arm. Its confidence interval
    is conditional on the supplied seeds and resamples only entities.
    """
    for runs in (reference_by_seed, candidate_by_seed):
        if not isinstance(runs, Mapping) or any(type(k) is not int or k < 0 for k in runs):
            raise ValueError('runs must map nonnegative integer training seed IDs to case rows')
    seeds = sorted(reference_by_seed)
    if not seeds or set(seeds) != set(candidate_by_seed):
        raise ValueError('both arms must specify the same nonempty set of training seed IDs')
    pairs = [(_index_cases(reference_by_seed[seed]), _index_cases(candidate_by_seed[seed]))
             for seed in seeds]
    roster = _roster((table for pair in pairs for table in pair), expected_case_entities)
    metric_names = _metric_names(metrics)
    return {
        'seeds': seeds,
        'roster_source': 'explicit' if expected_case_entities is not None else 'observed_union',
        'per_seed': {
            str(seed): _paired_report(reference, candidate, roster, metric_names,
                                      n_resamples, confidence, rng_seed)
            for seed, (reference, candidate) in zip(seeds, pairs)
        },
        'fixed_seed_average': _paired_metrics(pairs, roster, metric_names,
                                              n_resamples, confidence, rng_seed),
    }
