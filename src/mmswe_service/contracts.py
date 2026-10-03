"""Bind generation and grading to the same operator-frozen dataset bytes/IDs."""
import hashlib
import json
from pathlib import Path

from .artifacts import Rejected, canonical, validate_model


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_baseline(pin, baseline):
    """Verify all model bytes at operator startup, before accepting any work."""
    from .policy import private_file
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise Rejected('frozen baseline manifest path and sha256 are required')
    raw = private_file(pin['path']).read_bytes()
    if len(raw) > 1024*1024 or hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise Rejected('baseline manifest bytes changed')
    data = json.loads(raw)
    root = Path(baseline)
    if data.get('version') != 1 or data.get('path') != str(root):
        raise Rejected('baseline manifest path mismatch')
    records = data['files']
    if (not isinstance(records, list) or not records or
            hashlib.sha256(canonical(records)).hexdigest() != data['sha256']):
        raise Rejected('baseline content manifest hash mismatch')
    names, total = set(), 0
    for record in records:
        name = record['path']
        if (not isinstance(name, str) or Path(name).name != name or
                name in ('', '.', '..') or name in names):
            raise Rejected('invalid or duplicate baseline filename')
        names.add(name)
        file = private_file(root/name)
        if file.stat().st_size != record['size'] or file_sha256(file) != record['sha256']:
            raise Rejected('baseline bytes changed: '+name)
        total += record['size']
    if names != {p.name for p in root.iterdir()} or total != data['bytes']:
        raise Rejected('baseline file set or size changed')
    validate_model(root)
    return data


def load_contract(pin, cache):
    # Imported here because policy.load also calls this validation.
    from .policy import private_file
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise Rejected('frozen evaluation contract path and sha256 are required')
    path = private_file(pin['path'])
    raw = path.read_bytes()
    if len(raw) > 1024*1024 or hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise Rejected('evaluation contract bytes changed')
    data = json.loads(raw)
    if data.get('version') != 1 or data.get('dataset') != 'SWE-bench/SWE-bench_Multimodal':
        raise Rejected('unexpected evaluation contract')
    source = Path(data['source_snapshot'])
    # The exact snapshot must be within the cache mounted for generation.
    try:
        relative = source.relative_to(Path(cache)/'hf')
    except ValueError as error:
        raise Rejected('frozen dataset must be inside the configured HF cache') from error
    if not source.is_absolute() or '..' in relative.parts:
        raise Rejected('invalid frozen dataset path')
    required = {'dataset_info.json', 'swe-bench_multimodal-dev.arrow', 'swe-bench_multimodal-test.arrow'}
    if not required.issubset(data['source_files']):
        raise Rejected('incomplete frozen dataset files')
    for name, record in data['source_files'].items():
        if Path(name).name != name or name in ('', '.', '..'):
            raise Rejected('invalid frozen dataset filename')
        file = private_file(source/name)
        if file.stat().st_size != record['size'] or file_sha256(file) != record['sha256']:
            raise Rejected('frozen dataset bytes changed: '+name)
    seen = set()
    for split, count in [('dev', 100), ('test', 480)]:
        entry = data['splits'][split]
        ids = entry['instance_ids_in_order']
        if (entry['n'] != count or not isinstance(ids, list) or len(ids) != count
                or any(not isinstance(i, str) or not i for i in ids)
                or len(set(ids)) != count or seen.intersection(ids)):
            raise Rejected('invalid frozen split identity')
        if hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() != entry['ids_sha256']:
            raise Rejected('frozen split ID hash changed')
        seen.update(ids)
    return data


def score_contract(config, split, limit):
    pin = config['evaluation_contract']
    data = load_contract(pin, config['cache'])
    ids = data['splits'][split]['instance_ids_in_order']
    if type(limit) is not int or not 0 < limit <= len(ids):
        raise Rejected('requested evaluation size exceeds frozen split')
    source = Path(data['source_snapshot'])
    filename = 'swe-bench_multimodal-'+split+'.arrow'
    container = Path('/input/hf')/source.relative_to(Path(config['cache'])/'hf')/filename
    return {'contract_sha256': pin['sha256'], 'instance_ids': ids[:limit],
            'host_dataset_file': str(source/filename), 'container_dataset_file': str(container),
            'dataset_file_sha256': data['source_files'][filename]['sha256']}
