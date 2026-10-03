"""Match published target references to verifier metrics before comparing scores.

This detects wrong tasks, splits and denominators. It does not prove identity of
the dataset bytes or checkpoint; operators must freeze those separately.
"""
import argparse
import json
import math
from pathlib import Path

DEFAULT_BASELINES = Path(__file__).with_name('baselines.json')


def matched_baseline(bench, metrics, baselines=None):
    if baselines is None:
        baselines = json.loads(DEFAULT_BASELINES.read_text())
    if metrics.get('error') or metrics.get('benchmark') != bench:
        raise ValueError('missing or mismatched benchmark identity')
    if baselines.get('score_split') != 'eval' or metrics.get('eval_split') != 'eval':
        raise ValueError('published baseline requires the final eval split')
    expected = baselines.get('n_eval', {}).get(bench)
    if type(expected) is not int or type(metrics.get('n')) is not int or metrics['n'] != expected:
        raise ValueError('evaluation denominator differs from the published reference')
    if bench == 'mmswe' and (
            metrics.get('dataset') != 'SWE-bench/SWE-bench_Multimodal'
            or metrics.get('dataset_split') != 'test'):
        raise ValueError('MMSWE baseline requires the official test split')
    score = metrics.get('accuracy')
    base = baselines.get('scores', {}).get(bench)
    for value in (score, base):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('invalid accuracy or baseline')
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', required=True)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--baselines', default=str(DEFAULT_BASELINES))
    parser.add_argument('--tolerance', type=float, default=0.00005,
                        help='absolute accuracy tolerance; default covers published rounding')
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or not 0 <= args.tolerance <= 1:
        parser.error('tolerance must be finite and within [0,1]')
    try:
        metrics = json.loads(Path(args.metrics).read_text())
        baseline = matched_baseline(args.bench, metrics, json.loads(Path(args.baselines).read_text()))
        difference = abs(metrics['accuracy'] - baseline)
        ok = difference <= args.tolerance + 1e-12
        print(f"ORACLE {'PASS' if ok else 'FAIL'}: bench={args.bench} "
              f"n={metrics['n']} split=eval raw={metrics['accuracy']} "
              f"baseline={baseline} difference={difference:.8f}")
        return 0 if ok else 1
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
        print(f'ORACLE FAIL: {error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
