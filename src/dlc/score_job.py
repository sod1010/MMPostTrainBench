#!/usr/bin/env python3
"""MMSWE score completion protocol. No model code is imported by the waiter.

run: capture the adapter, validate the score, atomically publish status then DONE.
wait: observe status, worker exit, heartbeat and a bounded deadline; never infer
success from DONE or an old reward alone. This is not an execution sandbox.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time


def atomic(path, text):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def score(out):
    m = json.loads((out / 'metrics.json').read_text())
    n, correct, accuracy = m.get('n'), m.get('correct'), m.get('accuracy')
    if (type(n) is not int or n <= 0 or type(correct) is not int or
            not 0 <= correct <= n or type(accuracy) not in (int, float) or
            not math.isfinite(accuracy) or abs(accuracy - correct / n) > 1e-8):
        raise ValueError('invalid score counts')
    if m.get('task') != 'swe_bench_multimodal@val':
        raise ValueError('self-evaluation must return the validation task')
    return accuracy


def stop(proc):
    if proc is not None:
        # Kill the whole group even if the leader exited but left descendants.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('status.json', 'DONE', 'reward.txt', 'metrics.json', 'diag.jsonl'):
        (out / name).unlink(missing_ok=True)
    proc, rc, reason = None, 70, 'launch_failure'
    env = dict(os.environ, EVAL_SPLIT='val', MMPTB_ROLE='agent')
    previous = {}
    def cancelled(signum, frame):
        raise KeyboardInterrupt(signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, cancelled)
    try:
        with (out / 'run.log').open('w') as log:
            command = [args.python, args.adapter, '--model-path', args.model,
                       '--limit', str(args.limit), '--json-output-file', str(out / 'metrics.json')]
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            rc = proc.wait(timeout=args.timeout)
        if rc < 0:
            rc = 128 - rc
        if rc == 0:
            atomic(out / 'reward.txt', str(score(out)) + '\n')
            reason = 'ok'
        else:
            reason = 'adapter_failed'
    except subprocess.TimeoutExpired:
        rc, reason = 124, 'timeout'
    except KeyboardInterrupt as e:
        rc, reason = 128 + (e.args[0] if e.args else signal.SIGINT), 'cancelled'
    except (OSError, ValueError, KeyError, TypeError):
        rc, reason = 70, 'invalid_or_missing_score'
    finally:
        # Do not let a repeated termination signal interrupt status publication.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        stop(proc)
        if rc != 0:
            for name in ('reward.txt', 'metrics.json', 'diag.jsonl'):
                (out / name).unlink(missing_ok=True)
        atomic(out / 'status.json', json.dumps({'returncode': rc, 'reason': reason}) + '\n')
        atomic(out / 'DONE', 'DONE\n')
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return rc


def wait(args):
    out = Path(args.out)
    deadline = time.monotonic() + args.timeout
    worker_rc = None
    if args.queue:
        if not re.fullmatch(r'qworker-cmd\.\d+\.\d+', args.job_id or ''):
            raise ValueError('invalid worker job id')
        worker_rc = Path(args.queue) / 'tasks' / (args.job_id.removeprefix('qworker-') + '.rc')
    while True:
        status = out / 'status.json'
        if status.is_file():
            data = json.loads(status.read_text())
            rc = data.get('returncode')
            if type(rc) is not int or not 0 <= rc <= 255:
                raise ValueError('invalid completion status')
            if rc == 0:
                reward = float((out / 'reward.txt').read_text())
                if not math.isfinite(reward) or abs(reward - score(out)) > 1e-8:
                    raise ValueError('reward disagrees with validated metrics')
                print(reward)
            else:
                print('score failed: ' + str(data.get('reason', 'unknown')) + f' (rc={rc})', file=sys.stderr)
            return rc
        if worker_rc and worker_rc.exists():
            # Worker publishes its exit code after the adapter wrapper has ended.
            # A successful worker exit without our status is a protocol failure.
            print('score task ended without completion status', file=sys.stderr)
            return 70
        if args.queue:
            hb = Path(args.queue) / '.heartbeat'
            if not hb.is_file() or time.time() - hb.stat().st_mtime > args.heartbeat_timeout:
                print('score worker unavailable; no score', file=sys.stderr)
                return 69
        if time.monotonic() >= deadline:
            print('score wait timed out; no score', file=sys.stderr)
            return 124
        time.sleep(min(args.poll, max(0, deadline - time.monotonic())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('run')
    r.add_argument('--out', required=True)
    r.add_argument('--model', required=True)
    r.add_argument('--adapter', required=True)
    r.add_argument('--python', default=sys.executable)
    r.add_argument('--limit', type=int, default=100)
    r.add_argument('--timeout', type=float, default=43200)
    w = sub.add_parser('wait')
    w.add_argument('--out', required=True)
    w.add_argument('--queue')
    w.add_argument('--job-id')
    w.add_argument('--timeout', type=float, default=43500)
    w.add_argument('--heartbeat-timeout', type=float, default=300)
    w.add_argument('--poll', type=float, default=2)
    args = p.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error('timeout must be positive and finite')
    if args.mode == 'run' and args.limit <= 0:
        p.error('self-evaluation limit must be positive')
    if args.mode == 'wait' and (not math.isfinite(args.poll) or args.poll <= 0):
        p.error('poll interval must be positive and finite')
    try:
        return run(args) if args.mode == 'run' else wait(args)
    except (OSError, ValueError, KeyError, TypeError):
        print('invalid score completion protocol; no score', file=sys.stderr)
        return 70


if __name__ == '__main__':
    raise SystemExit(main())
