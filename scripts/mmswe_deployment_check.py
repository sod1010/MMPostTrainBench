#!/usr/bin/env python3
"""Read-only deployment inventory. Never starts containers, jobs, or a loop.

Even successful checks do not certify browser support, egress isolation, storage
quotas, model inference, or the real service feedback cycle.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mmswe_service.artifacts import canonical, validate_model
from mmswe_service.policy import load


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def contract_inventory(path):
    """Check the existing frozen cache and count repositories, not failures."""
    data = json.loads(Path(path).read_text())
    if data['version'] != 1 or data['dataset'] != 'SWE-bench/SWE-bench_Multimodal':
        raise ValueError('unexpected evaluation contract')
    root = Path(data['source_snapshot'])
    if not root.is_absolute():
        raise ValueError('snapshot must be absolute')
    files = data['source_files']
    required = {'dataset_info.json', 'swe-bench_multimodal-dev.arrow', 'swe-bench_multimodal-test.arrow'}
    if not required.issubset(files):
        raise ValueError('missing frozen dataset files')
    for name, record in files.items():
        if Path(name).name != name or name in ('', '.', '..'):
            raise ValueError('invalid snapshot filename')
        file = root/name
        if file.stat().st_size != record['size'] or sha256(file) != record['sha256']:
            raise ValueError('frozen dataset file changed: '+name)
    splits, seen = {}, set()
    for split, count in [('dev', 100), ('test', 480)]:
        record = data['splits'][split]
        ids = record['instance_ids_in_order']
        if (record['n'] != count or len(ids) != count or
                any(not isinstance(i, str) or not i for i in ids) or
                len(set(ids)) != count or seen.intersection(ids)):
            raise ValueError('invalid or overlapping frozen split')
        if hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() != record['ids_sha256']:
            raise ValueError('frozen instance ID hash mismatch')
        seen.update(ids)
        splits[split] = {'n': count, 'ids_sha256': record['ids_sha256'],
                         'repository_counts': dict(Counter(i.rsplit('-', 1)[0] for i in ids))}
    return {'sha256': sha256(path), 'source_files_verified': len(files), 'splits': splits,
            'scope': 'frozen reference verified; deployment cache and runtime binding still require acceptance'}


def deployment_inventory(path):
    result = {'config_path': str(path), 'checks': [], 'loop_ready': False,
              'scope': 'read-only inventory; no containers or GPU tasks started'}

    def check(name, operation):
        try:
            detail = operation()
            result['checks'].append({'name': name, 'passed': True, 'detail': detail})
            return detail
        except Exception as error:
            result['checks'].append({'name': name, 'passed': False,
                                     'error': str(error)[:600]})
            return None

    config = check('operator_policy', lambda: load(path))
    if config is None:
        result['status'] = 'configuration_blocked'
        return result
    # Keep config paths/UIDs in the private operator receipt, never credentials.
    result['policy_sha256'] = hashlib.sha256(canonical(config)).hexdigest()

    def directory(path, owner, mode=None):
        p = Path(path)
        if not p.is_dir() or p.is_symlink():
            raise ValueError('missing or symlinked directory: '+str(p))
        s = p.stat()
        if s.st_uid != owner or (mode is not None and stat.S_IMODE(s.st_mode) != mode):
            raise ValueError('unexpected directory ownership or mode: '+str(p))
        return {'path': str(p), 'uid': s.st_uid, 'mode': oct(stat.S_IMODE(s.st_mode))}

    check('private_state', lambda: directory(config['state'], os.geteuid(), 0o700))
    check('socket_parent', lambda: directory(Path(config['socket']).parent, os.geteuid()))
    result['socket_present'] = Path(config['socket']).is_socket()
    for arm, run in config['runs'].items():
        check('workspace_'+arm, lambda r=run: directory(r['workspace'], r['uid']))
    result['registered_runs'] = list(config['runs'])
    result['profiles'] = config['profiles']
    result['instance_resources'] = config['instance_resources']
    check('baseline_structure', lambda: validate_model(Path(config['baseline'])))
    for kind in ('images', 'assets'):
        def manifest(k=kind):
            actual = sha256(Path(config['cache'])/k/'manifest.json')
            expected = config['image_manifest_sha256' if k == 'images' else 'asset_manifest_sha256']
            if actual != expected:
                raise ValueError('manifest hash mismatch')
            return actual
        check(kind+'_manifest', manifest)
    for key in ('train_image', 'generate_image'):
        def image(k=key):
            response = subprocess.run([config['docker'], 'image', 'inspect', config[k]],
                env={'PATH': '/usr/bin:/bin', 'HOME': str(Path(config['state'])/'docker-home')},
                capture_output=True, text=True, timeout=30)
            if response.returncode:
                raise ValueError('fixed image unavailable from the configured daemon')
            entries = json.loads(response.stdout)
            if len(entries) != 1 or config[k] not in [entries[0].get('Id'), *(entries[0].get('RepoDigests') or [])]:
                raise ValueError('daemon image does not match the pinned immutable image reference')
            return {'requested': config[k], 'image_id': entries[0]['Id']}
        check(key, image)
    result['status'] = ('static_checks_passed_runtime_unverified' if
                        all(c['passed'] for c in result['checks']) else 'configuration_blocked')
    result['runtime_evidence_required'] = [
        'same final backend: original four official cases plus su and network cases',
        'sandbox-preserving Chrome and runnable Firefox',
        'enforced egress policy and private state storage quota',
        'real train -> frozen artifact -> dev score -> next candidate cycle',
        'operator-only sealed evaluation and four arm identity/resource receipts']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='/etc/mmptb/service.json')
    parser.add_argument('--contract', help='optional existing frozen evaluation contract')
    parser.add_argument('--out', required=True, help='new private JSON receipt; overwrite refused')
    args = parser.parse_args()
    result = deployment_inventory(args.config)
    if args.contract:
        try:
            result['evaluation_contract'] = contract_inventory(args.contract)
        except Exception as error:
            result['evaluation_contract'] = {'error': str(error)[:600]}
            result['status'] = 'configuration_blocked'
    with open(args.out, 'x') as stream:
        os.chmod(args.out, 0o600)
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'status': result['status'], 'loop_ready': False, 'receipt': args.out}))
    return 2 if result['status'] == 'configuration_blocked' else 0


if __name__ == '__main__':
    raise SystemExit(main())
