"""Only the operator defines identity, resources, images and evaluation settings."""
import json
import os
from pathlib import Path
import re
import stat
from .artifacts import Rejected, relative
from .container_policy import profile_name
from .resources import instance_resources
from .contracts import load_contract, load_baseline
from .offline_resources import load_snapshot
from .runtime_guard import PROFILE as GUARD_PROFILE, load_guard

ID = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z')
IMAGE = re.compile(r'(?:[a-zA-Z0-9][a-zA-Z0-9._/:-]*@)?sha256:[a-f0-9]{64}\Z')


def pinned_image(value):
    if not isinstance(value, str) or not IMAGE.fullmatch(value) or value.endswith('0'*64):
        raise Rejected('images must use a real digest, not a template placeholder')
    return value


def identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise Rejected('invalid identifier')
    return value


def private_file(path):
    p = Path(path)
    if not p.is_absolute() or p.is_symlink():
        raise Rejected('operator file must be an absolute regular path')
    for parent in (p, *p.parents):
        s = parent.stat()
        if s.st_uid not in (0, os.geteuid()) or s.st_mode & 0o022:
            raise Rejected('operator path has an untrusted writer')
    if not p.is_file():
        raise Rejected('missing operator file')
    return p


def load(path):
    data = json.loads(private_file(path).read_text())
    expected = {'state', 'socket', 'docker', 'train_image', 'generate_image', 'grader_python',
                'cache', 'baseline', 'runs', 'profiles', 'dataset', 'dev_limit',
                'test_limit', 'image_manifest_sha256', 'asset_manifest_sha256', 'evaluation_contract',
                'baseline_manifest'}
    if set(data) - {'container_profile', 'instance_resources', 'offline_resources', 'runtime_guard'} != expected:
        raise Rejected('unknown or missing operator policy field')
    try:
        data['container_profile'] = profile_name(data.get('container_profile', 'strict-v1'))
        data['instance_resources'] = instance_resources(data.get('instance_resources'))
        if data['container_profile'] == GUARD_PROFILE:
            load_guard(data.get('runtime_guard'))
        elif data.get('runtime_guard') is not None:
            raise ValueError('runtime_guard requires entry-guard-v1')
    except ValueError as error:
        raise Rejected(str(error)) from error
    for key in ('train_image', 'generate_image'):
        pinned_image(data[key])
    for key in ('docker', 'grader_python'):
        private_file(data[key])
    for key in ('state', 'socket', 'cache', 'baseline'):
        p = Path(data[key])
        if not p.is_absolute() or '..' in p.parts or ',' in str(p):
            raise Rejected('invalid operator path')
    load_baseline(data['baseline_manifest'], data['baseline'])
    load_contract(data['evaluation_contract'], data['cache'])
    load_snapshot(data.get('offline_resources'))
    if data['dataset'] != 'SWE-bench/SWE-bench_Multimodal':
        raise Rejected('unexpected benchmark dataset')
    for key in ('dev_limit', 'test_limit'):
        if type(data[key]) is not int or data[key] <= 0:
            raise Rejected('fixed positive evaluation size required')
    for key in ('image_manifest_sha256', 'asset_manifest_sha256'):
        if not isinstance(data[key], str) or not re.fullmatch('[a-f0-9]{64}', data[key]) or data[key] == '0'*64:
            raise Rejected('frozen manifest pin required')
    uids = set()
    if not data['runs']:
        raise Rejected('no registered runs')
    for name, run in data['runs'].items():
        identifier(name)
        if set(run) != {'uid', 'workspace', 'budget_seconds'}:
            raise Rejected('invalid run policy')
        uid = run['uid']
        if type(uid) is not int or uid <= 0 or uid == os.geteuid() or uid in uids:
            raise Rejected('each agent needs a distinct unprivileged host UID')
        uids.add(uid)
        if type(run['budget_seconds']) is not int or run['budget_seconds'] <= 0:
            raise Rejected('invalid run budget')
        if not Path(run['workspace']).is_absolute():
            raise Rejected('workspace must be absolute')
    if set(data['profiles']) != {'train', 'score'}:
        raise Rejected('train and score profiles required')
    for profile in data['profiles'].values():
        if set(profile) != {'timeout', 'cpus', 'memory_gib', 'pids', 'gpus', 'artifact_bytes'}:
            raise Rejected('invalid resource profile')
        for k in ('timeout', 'cpus', 'memory_gib', 'pids', 'artifact_bytes'):
            if type(profile[k]) is not int or profile[k] <= 0:
                raise Rejected('positive resource limits required')
        if not isinstance(profile['gpus'], list) or any(type(g) is not int or g < 0 for g in profile['gpus']) or len(set(profile['gpus'])) != len(profile['gpus']):
            raise Rejected('invalid operator GPU assignment')
    return data


def validate_request(req, uid, config, operator=False):
    if not isinstance(req, dict):
        raise Rejected('request must be an object')
    op = req.get('op')
    schemas = {'train': {'op', 'run', 'request_id', 'code', 'data'},
               'score': {'op', 'run', 'request_id', 'artifact'},
               'import_model': {'op', 'run', 'request_id', 'model'},
               'status': {'op', 'run', 'job'},
               'seal': {'op', 'run', 'request_id', 'artifact'}}
    if op not in schemas or set(req) != schemas[op]:
        raise Rejected('unknown operation or fields; no shell/env/role overrides')
    run = identifier(req['run'])
    if run not in config['runs']:
        raise Rejected('unknown run')
    if operator:
        if uid != os.geteuid() or op not in ('seal', 'status'):
            raise Rejected('operator endpoint only permits sealed evaluation/status')
    elif op == 'seal' or uid != config['runs'][run]['uid']:
        raise Rejected('peer UID does not own this operation/run')
    for k in ('request_id', 'artifact', 'job'):
        if k in req:
            identifier(req[k])
    for k in ('code', 'data', 'model'):
        if k in req:
            relative(req[k])
    return req
