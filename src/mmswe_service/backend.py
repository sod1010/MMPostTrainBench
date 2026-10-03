"""Ordinary Docker containers only; no privileged/cap-add/userns fallback.

Docker access belongs to the broker, never to the research agent. Deployment
requires a platform-approved local daemon, dedicated disk quota for state, and
real isolation acceptance. CPU command tests do not certify that deployment.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import secrets

from .artifacts import Rejected
from .container_policy import security_args, inspect_security
from .resources import resource_environment
from .contracts import score_contract
from .offline_resources import load_snapshot, MANIFEST_ENV, SHA_ENV
from . import runtime_guard as guard


class BackendFailure(RuntimeError):
    pass


class BackendUnhealthy(BackendFailure):
    """Do not run further work until operator recovery cleans the backend."""


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name('.'+path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class DockerBackend:
    def __init__(self, config, service_id):
        self.config = config
        self.service_id = service_id
        self.binary = config['docker']

    def cli(self, args, timeout=30, check=True):
        # Do not inherit DOCKER_HOST, contexts, API credentials, PYTHONPATH, etc.
        env = {'PATH': '/usr/bin:/bin', 'HOME': str(Path(self.config['state'])/'docker-home')}
        p = subprocess.run([self.binary, *args], env=env, capture_output=True,
                           text=True, timeout=timeout)
        if check and p.returncode:
            raise BackendFailure('Docker operation rejected or failed')
        return p

    def cleanup(self):
        ids = self.cli(['ps', '-aq', '--filter', 'label=mmptb.service='+self.service_id]).stdout.split()
        for cid in ids:
            if not re.fullmatch('[a-f0-9]{12,64}', cid):
                raise BackendFailure('invalid Docker container identity')
            self.cli(['rm', '-f', cid])
        images = self.cli(['images', '-q', '--filter', 'label=mmptb.service='+self.service_id]).stdout.split()
        for image in set(images):
            if not re.fullmatch('[a-f0-9]{12,64}|sha256:[a-f0-9]{64}', image):
                raise BackendUnhealthy('invalid image cleanup identity')
            self.cli(['image', 'rm', '-f', image])

    def create_args(self, job, mode, mounts, output, nonce=None):
        profile = self.config['profiles']['train' if mode == 'train' else 'score']
        image = self.config['train_image' if mode == 'train' else 'generate_image']
        guarded = self.config.get('container_profile') == guard.PROFILE
        if guarded:
            guard.load_guard(self.config.get('runtime_guard'))
            guard.command('workload', mode, nonce)
        args = ['create', '--name', 'mmptb-'+self.service_id[:12]+'-'+job,
                '--label', 'mmptb.service='+self.service_id,
                '--label', 'mmptb.job='+job, '--pull', 'never',
                '--init', '--user', '0:0' if guarded else '65534:65534', '--read-only',
                '--network', 'none', '--pids-limit', str(profile['pids']),
                '--memory', str(profile['memory_gib'])+'g', '--memory-swap', str(profile['memory_gib'])+'g',
                '--cpus', str(profile['cpus']), '--shm-size', '1g',
                '--tmpfs', '/tmp:rw,nosuid,nodev,size=2g',
                '--log-driver', 'local', '--log-opt', 'max-size=10m', '--log-opt', 'max-file=2',
                '--workdir', '/output', '--env', 'HOME=/tmp',
                '--entrypoint', guard.GUARD_PATH if guarded else '/bin/bash']
        if guarded:
            args += ['--interactive']
        args += security_args('workload', self.config.get('container_profile', 'strict-v1'))
        # The allowlist of mounts is constructed here, not accepted from a client.
        for src, dst in mounts:
            if ',' in str(src) or not dst.startswith('/'):
                raise BackendFailure('invalid mount')
            args += ['--mount', f'type=bind,source={src},target={dst},readonly']
        args += ['--mount', f'type=bind,source={output},target=/output']
        if profile['gpus']:
            args += ['--gpus', '"device='+','.join(map(str, profile['gpus']))+'"']
        args += [image] + (guard.command('workload',mode,nonce) if guarded else ['/opt/mmptb-service/entry.sh',mode])
        return args

    def container(self, job, mode, mounts, output, deadline):
        output = Path(output)
        output.mkdir(mode=0o777)
        output.chmod(0o777)  # private parent is broker-only; only the child sees this bind
        name = 'mmptb-'+self.service_id[:12]+'-'+job
        cid = None
        guarded = self.config.get('container_profile') == guard.PROFILE
        nonce = secrets.token_hex(32) if guarded else None
        try:
            args = self.create_args(job, mode, mounts, output, nonce)
            # The daemon must already have the exact approved image digest.
            cid = self.cli(args, timeout=min(60, max(1, deadline-time.monotonic()))).stdout.strip()
            if not re.fullmatch('[a-f0-9]{64}', cid):
                raise BackendFailure('invalid created container')
            inspected = json.loads(self.cli(['inspect', cid]).stdout)
            if not isinstance(inspected, list) or len(inspected) != 1:
                raise BackendFailure('invalid container inspection')
            receipt = inspect_security(inspected[0], 'workload',
                                       self.config.get('container_profile', 'strict-v1'))
            atomic(output.parent/(job+'-container-security.json'), receipt)
            if not receipt['compliant']:
                raise BackendFailure('container security policy mismatch; workload not started')
            if guarded:
                image = self.config['train_image' if mode == 'train' else 'generate_image']
                guard.validate_launch(inspected[0],'workload',mode,nonce,image)
                home = Path(self.config['state'])/'docker-home'
                guard.verify_container_guard(self.binary,home,cid,self.config['runtime_guard'],
                                             timeout=min(30,max(.01,deadline-time.monotonic())))
                evidence = guard.guarded_start(self.binary,home,cid,'workload',mode,nonce,
                                              output.parent/(job+'-guard.log'),deadline)
                atomic(output.parent/(job+'-guard.json'),{**evidence,'guard_sha256':self.config['runtime_guard']['sha256']})
                if evidence['attach_returncode']:
                    raise BackendFailure('guarded workload failed; no result')
            else:
                self.cli(['start', cid], timeout=min(60, max(1, deadline-time.monotonic())))
            result = self.cli(['wait', cid], timeout=max(.01, deadline-time.monotonic()))
            if result.stdout.strip() != '0':
                raise BackendFailure('workload failed; no result')
        finally:
            # Name is known even when create timed out after daemon-side creation.
            try:
                cleanup = self.cli(['rm', '-f', cid or name], check=False)
                if cleanup.returncode and 'No such container' not in cleanup.stderr:
                    raise BackendUnhealthy('container cleanup failed; execution must stop')
            except (OSError, subprocess.TimeoutExpired) as e:
                raise BackendUnhealthy('container cleanup unavailable') from e
        output.chmod(0o700)

    def train(self, job, directory, code, data, deadline):
        self.container(job, 'train', [(code, '/input/code'), (data, '/input/data'),
                                     (self.config['baseline'], '/input/base')],
                       directory/'output', deadline)
        return directory/'output'/'model'

    def score(self, job, directory, model, sealed, deadline):
        cache = Path(self.config['cache'])
        request = {'dataset': self.config['dataset'], 'split': 'test' if sealed else 'dev',
                   'eval_split': 'eval' if sealed else 'val',
                   'limit': self.config['test_limit' if sealed else 'dev_limit'],
                   'image_manifest_sha256': self.config['image_manifest_sha256']}
        contract = score_contract(self.config, request['split'], request['limit'])
        request.update({k:v for k,v in contract.items() if k != 'host_dataset_file'})
        request_dir = directory/'request'
        request_dir.mkdir(mode=0o755)
        atomic(request_dir/'request.json', request)
        (request_dir/'request.json').chmod(0o444)
        for name in ('assets', 'images'):
            pin = self.config['asset_manifest_sha256' if name == 'assets' else 'image_manifest_sha256']
            if hashlib.sha256((cache/name/'manifest.json').read_bytes()).hexdigest() != pin:
                raise BackendFailure('operator cache manifest changed')
        self.container(job, 'generate', [(model, '/input/model'), (request_dir, '/input/request'),
                                         (cache/'images', '/input/images'), (cache/'hf', '/input/hf')],
                       directory/'generation', deadline)
        # Re-copy generation output through the no-follow reader before the trusted grader.
        from .artifacts import freeze
        frozen = directory/'predictions'
        freeze(directory, 'generation', frozen, 128*1024*1024, 10, deadline)
        preds = frozen/'predictions.jsonl'
        meta = json.loads((frozen/'meta.json').read_text())
        predictions = [json.loads(l) for l in preds.read_text().splitlines() if l.strip()]
        if not predictions or len(predictions) != request['limit']:
            raise BackendFailure('incomplete generation; no grading')
        if meta.get('split') != request['split'] or meta.get('eval_split') != request['eval_split'] or meta.get('n_generation_failed') != 0:
            raise BackendFailure('invalid generation metadata')
        if (meta.get('evaluation_contract_sha256') != contract['contract_sha256'] or
                meta.get('instance_ids') != contract['instance_ids'] or
                [p.get('instance_id') for p in predictions] != contract['instance_ids']):
            raise BackendFailure('generation differs from frozen evaluation identity')
        cmd = [self.config['grader_python'], str(Path(__file__).with_name('docker_grade.py')),
               '--preds', str(preds), '--out', str(directory/'grade'),
               '--dataset', request['dataset'], '--split', request['split'],
               '--scratch', str(directory/'scratch'), '--timeout', '1800', '--jobs', '1']
        env = {'PATH': '/usr/bin:/bin', 'HOME': str(directory), 'HF_HOME': str(cache/'hf'),
               'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
               'MMSWE_ASSET_MANIFEST': str(cache/'assets/manifest.json'),
               'MMSWE_ASSET_MANIFEST_SHA256': self.config['asset_manifest_sha256'],
               'MMSWE_BLOB_CACHE': str(cache/'blobs'), 'MMSWE_DOCKER': self.binary,
               'MMSWE_DOCKER_HOME': str(Path(self.config['state'])/'docker-home'),
               'MMSWE_CONTAINER_PROFILE': self.config.get('container_profile', 'strict-v1'),
               'MMSWE_SERVICE_ID': self.service_id, 'MMSWE_JOB_ID': job}
        env.update(MMSWE_FROZEN_DATASET_FILE=contract['host_dataset_file'],
                   MMSWE_FROZEN_DATASET_SHA256=contract['dataset_file_sha256'])
        # Comes only from the operator policy, never the agent request or parent env.
        env.update(resource_environment(self.config.get('instance_resources')))
        if self.config.get('container_profile') == guard.PROFILE:
            pin = guard.load_guard(self.config['runtime_guard'])
            env.update({guard.PIN_ENV:pin['path'],guard.SHA_ENV:pin['sha256']})
        snapshot=load_snapshot(self.config.get('offline_resources'))
        if snapshot:
            env.update({MANIFEST_ENV:snapshot['pin']['path'],SHA_ENV:snapshot['pin']['sha256']})
        # Grader subprocess can be killed; the finally block removes all job containers.
        import signal
        proc = None
        try:
            with (directory/'grader.log').open('w') as log:
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                if proc.wait(timeout=max(.01, deadline-time.monotonic())):
                    raise BackendFailure('grading failed; no score')
        finally:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            try:
                ids = self.cli(['ps', '-aq', '--filter', 'label=mmptb.service='+self.service_id,
                                '--filter', 'label=mmptb.job='+job]).stdout.split()
                for cid in ids:
                    if not re.fullmatch('[a-f0-9]{12,64}', cid):
                        raise BackendUnhealthy('invalid cleanup identity')
                    self.cli(['rm', '-f', cid])
                # A killed grading process may leave an imported instance image.
                images = self.cli(['images', '-q', '--filter', 'label=mmptb.service='+self.service_id]).stdout.split()
                for image in set(images):
                    if not re.fullmatch('[a-f0-9]{12,64}|sha256:[a-f0-9]{64}', image):
                        raise BackendUnhealthy('invalid image cleanup identity')
                    self.cli(['image', 'rm', '-f', image])
            except (BackendFailure, OSError, subprocess.TimeoutExpired) as e:
                raise BackendUnhealthy('grading cleanup failed') from e
        # Reuse the existing strict coverage/status validator, outside agent control.
        import importlib.util
        module = Path(__file__).parents[1]/'eval/tasks/mmswe/evaluate.py'
        spec = importlib.util.spec_from_file_location('mmswe_broker_validator', module)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        summary = json.loads((directory/'grade/summary.json').read_text())
        report = json.loads((directory/'grade/report.json').read_text())
        ids, resolved = adapter.validate_run(predictions, meta, summary, report,
                                             request['dataset'], request['split'], request['eval_split'])
        if any(type(p.get('n_images')) is not int or p['n_images'] < 0 or not isinstance(p.get('image_sha256'), list) or len(p['image_sha256']) != p['n_images'] or any(not isinstance(h, str) or not re.fullmatch('[a-f0-9]{64}', h) for h in p['image_sha256']) for p in predictions):
            raise BackendFailure('invalid per-item media audit')
        if meta.get('image_manifest_sha256') != request['image_manifest_sha256']:
            raise BackendFailure('media manifest binding mismatch')
        return {'accuracy': len(resolved)/len(ids), 'correct': len(resolved), 'n': len(ids),
                'split': request['split'], 'predictions_sha256': hashlib.sha256(preds.read_bytes()).hexdigest(),
                'image_manifest_sha256': request['image_manifest_sha256'],
                'asset_manifest_sha256': self.config['asset_manifest_sha256'],
                'evaluation_contract_sha256': contract['contract_sha256'],
                'offline_snapshot_sha256': snapshot['pin']['sha256'] if snapshot else None}
