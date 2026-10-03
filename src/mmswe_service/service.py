"""Linux peer-authenticated broker. Experimental until real backend acceptance.

Run with: PYTHONPATH=src python -m mmswe_service.service --config /etc/mmptb/service.json
No legacy queue is read or modified. A separate operator socket admits sealing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import socket
import socketserver
import stat
import struct
import threading
import time
import uuid

from .artifacts import Rejected, canonical, freeze, validate_model, discard_private_tree
from .backend import DockerBackend, BackendFailure, BackendUnhealthy, atomic
from .policy import load, validate_request


class Broker:
    def __init__(self, config, backend=None):
        self.config = config
        self.state = Path(config['state'])
        self.state.mkdir(mode=0o700, parents=True, exist_ok=True)
        s = self.state.stat()
        if self.state.is_symlink() or s.st_uid != os.geteuid() or stat.S_IMODE(s.st_mode) != 0o700:
            raise Rejected('state must be broker-owned mode 0700')
        # Do not place private state inside any agent workspace, or vice versa.
        for run in config['runs'].values():
            ws = Path(run['workspace']).resolve()
            if self.state.resolve().is_relative_to(ws) or ws.is_relative_to(self.state.resolve()):
                raise Rejected('state and agent workspaces overlap')
        for n in ('jobs', 'artifacts', 'docker-home'):
            (self.state/n).mkdir(mode=0o700, exist_ok=True)
        ident = self.state/'service_id'
        if not ident.exists():
            ident.write_text(uuid.uuid4().hex)
        self.service_id = ident.read_text().strip()
        if len(self.service_id) != 32 or any(x not in '0123456789abcdef' for x in self.service_id):
            raise Rejected('invalid service identity')
        self.backend = backend or DockerBackend(config, self.service_id)
        self.lock = threading.Lock()
        self.q = queue.Queue()
        self.policy_hash = hashlib.sha256(canonical(config)).hexdigest()

    def job_path(self, job):
        return self.state/'jobs'/job/'state.json'

    def save(self, job):
        atomic(self.job_path(job['id']), job)

    def jobs(self):
        return [json.loads(p.read_text()) for p in (self.state/'jobs').glob('*/state.json')]

    def recover(self):
        # Called under the process lock at startup. A restart must not silently
        # change arm identities, budgets, images or split sizes for old jobs.
        self.bind_policy()
        # Failure to clean prior containers prevents the service from starting.
        self.backend.cleanup()
        (self.state/'blocked.json').unlink(missing_ok=True)
        for job in self.jobs():
            if job['status'] in ('queued', 'running'):
                job.update(status='failed', reason='service_restart', finished=time.time(),
                           charged_seconds=job['reserved_seconds'])
                self.save(job)

    def bind_policy(self):
        pin = self.state/'policy.json'
        if pin.exists() and json.loads(pin.read_text()).get('sha256') != self.policy_hash:
            raise Rejected('service policy changed; use a new state directory for a new experiment')
        if any(j.get('policy_sha256') != self.policy_hash for j in self.jobs()):
            raise Rejected('historical jobs do not match the current service policy')
        if not pin.exists():
            atomic(pin, {'sha256': self.policy_hash, 'config': self.config})

    def public(self, job, operator):
        result = {k: job[k] for k in ('id', 'run', 'status', 'op')}
        if job['op'] == 'seal' and not operator:
            raise Rejected('sealed task is operator-only')
        for k in ('reason', 'artifact', 'elapsed_seconds', 'charged_seconds'):
            if k in job:
                result[k] = job[k]
        if job['status'] == 'succeeded' and 'score' in job:
            # Only scalar feedback, never per-item logs, predictions or test data.
            result['score'] = {k: job['score'][k] for k in ('accuracy', 'correct', 'n')}
        return result

    def submit(self, request, uid, operator=False):
        req = validate_request(request, uid, self.config, operator)
        with self.lock:
            if req['op'] == 'status':
                path = self.job_path(req['job'])
                if not path.exists():
                    raise Rejected('unknown job')
                job = json.loads(path.read_text())
                if job['run'] != req['run']:
                    raise Rejected('job belongs to another run')
                return self.public(job, operator)
            if (self.state/'blocked.json').exists():
                raise Rejected('backend blocked pending operator recovery')
            fingerprint = hashlib.sha256(canonical(req)).hexdigest()
            jobs = self.jobs()
            for old in jobs:
                if old['run'] == req['run'] and old['request']['request_id'] == req['request_id']:
                    if old['request_sha256'] != fingerprint:
                        raise Rejected('request id reused with different content')
                    return self.public(old, operator)
            if any(j['run'] == req['run'] and j['op'] == 'seal' for j in jobs):
                raise Rejected('run was sealed; no further research requests')
            if req['op'] in ('score', 'seal'):
                self.artifact(req['run'], req['artifact'])
            # Reserve the entire allowed duration before admitting work. Queue
            # waits are not charged; restart with lost timing is conservative.
            kind = 'train' if req['op'] in ('train', 'import_model') else 'score'
            seconds = self.config['profiles'][kind]['timeout']
            spent = sum(j.get('charged_seconds', j['reserved_seconds']) for j in jobs
                        if j['run'] == req['run'] and j['op'] != 'seal')
            if req['op'] != 'seal' and spent+seconds > self.config['runs'][req['run']]['budget_seconds']:
                raise Rejected('insufficient reserved run budget')
            if sum(j['status'] in ('queued', 'running') for j in jobs if j['run'] == req['run']) >= 2:
                raise Rejected('too many pending requests')
            if req['op'] == 'seal' and any(j['run'] == req['run'] and j['status'] in ('queued', 'running') for j in jobs):
                raise Rejected('finish pending research before sealing')
            jid = uuid.uuid4().hex
            (self.state/'jobs'/jid).mkdir(mode=0o700)
            job = {'id': jid, 'run': req['run'], 'op': req['op'], 'request': req,
                   'request_sha256': fingerprint, 'policy_sha256': self.policy_hash,
                   'status': 'queued', 'created': time.time(), 'reserved_seconds': seconds,
                   'peer_uid': uid, 'operator': operator}
            self.save(job)
            self.q.put(jid)
            return self.public(job, operator)

    def artifact(self, run, ident):
        directory = self.state/'artifacts'/ident
        p = directory/'manifest.json'
        if not p.is_file():
            raise Rejected('unknown artifact')
        manifest = json.loads(p.read_text())
        if manifest['run'] != run:
            raise Rejected('artifact belongs to another run')
        # Storage is private to the broker. Never accept an agent-written manifest.
        return directory/'model', manifest

    def register_model(self, run, job, root, name, maximum, deadline=None):
        ident = uuid.uuid4().hex
        directory = self.state/'artifacts'/ident
        directory.mkdir(mode=0o700)
        try:
            manifest = freeze(root, name, directory/'model', maximum, deadline=deadline)
            validate_model(directory/'model')
            manifest.update(run=run, job=job, artifact=ident, created=time.time())
            atomic(directory/'manifest.json', manifest)
        except BaseException:
            discard_private_tree(directory)
            raise
        return ident

    def execute(self, jid):
        with self.lock:
            job = json.loads(self.job_path(jid).read_text())
            job.update(status='running', started=time.time())
            self.save(job)
        started = time.monotonic()
        req = job['request']
        directory = self.state/'jobs'/jid
        run = self.config['runs'][job['run']]
        profile = self.config['profiles']['train' if job['op'] in ('train', 'import_model') else 'score']
        deadline = started+job['reserved_seconds']
        try:
            if (self.state/'blocked.json').exists():
                raise BackendUnhealthy('backend blocked')
            if job['op'] == 'import_model':
                artifact = self.register_model(job['run'], jid, run['workspace'], req['model'], profile['artifact_bytes'], deadline)
                job['artifact'] = artifact
            elif job['op'] == 'train':
                code = freeze(run['workspace'], req['code'], directory/'code', 16*1024*1024, 1000, deadline)
                data = freeze(run['workspace'], req['data'], directory/'data', profile['artifact_bytes'], deadline=deadline)
                if not (directory/'code/train.py').is_file():
                    raise Rejected('training export must contain train.py')
                job['input_sha256'] = {'code': code['sha256'], 'data': data['sha256']}
                self.save(job)
                model = self.backend.train(jid, directory, directory/'code', directory/'data', deadline)
                job['artifact'] = self.register_model(job['run'], jid, model.parent, model.name, profile['artifact_bytes'], deadline)
            else:
                model, manifest = self.artifact(job['run'], req['artifact'])
                job['artifact_sha256'] = manifest['sha256']
                self.save(job)
                job['score'] = self.backend.score(jid, directory, model, job['op']=='seal', deadline)
            if time.monotonic() > deadline:
                raise TimeoutError('job deadline exceeded')
            job['status'] = 'succeeded'
        except Exception as error:
            if isinstance(error, BackendUnhealthy):
                atomic(self.state/'blocked.json', {'job': jid, 'reason': 'cleanup_or_backend_failure'})
            job.pop('score', None)
            job.update(status='failed', reason='execution_or_validation_failed')
            # Details remain operator-private, never streamed into research context.
            atomic(directory/'failure.json', {'type': type(error).__name__, 'detail': str(error)})
        finally:
            elapsed = time.monotonic()-started
            # Wall time includes copies and CPU grading. It is not observed GPU
            # allocation or utilization; importing weights allocates no GPU.
            job.update(elapsed_seconds=elapsed, charged_seconds=elapsed, finished=time.time(),
                       profile_gpu_seconds_upper_estimate=(0 if job['op'] == 'import_model'
                                                           else elapsed*len(profile['gpus'])),
                       gpu_accounting='profile_wall_time_upper_estimate_not_measured')
            with self.lock:
                self.save(job)

    def work(self):
        while True:
            jid = self.q.get()
            try:
                self.execute(jid)
            finally:
                self.q.task_done()


class Server(socketserver.UnixStreamServer):
    allow_reuse_address = False
    request_queue_size = 8

    def __init__(self, address, broker, operator):
        self.broker, self.operator = broker, operator
        super().__init__(address, Handler)


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(5)
        uid = struct.unpack('3i', self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
        try:
            raw = self.rfile.readline(16385)
            if len(raw) > 16384 or not raw.endswith(b'\n'):
                raise Rejected('oversized or incomplete request')
            def unique(pairs):
                result = {}
                for k, v in pairs:
                    if k in result:
                        raise Rejected('duplicate JSON field')
                    result[k] = v
                return result
            req = json.loads(raw, object_pairs_hook=unique)
            response = self.server.broker.submit(req, uid, self.server.operator)
            self.wfile.write(canonical({'ok': True, 'result': response})+b'\n')
        except Exception:
            self.wfile.write(b'{"ok":false,"error":"request rejected"}\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    if not hasattr(socket, 'SO_PEERCRED') or os.name != 'posix':
        ap.error('Linux peer credentials are required')
    config = load(args.config)
    broker = Broker(config)
    # Exclusive process lock; clients cannot touch broker state.
    import fcntl
    lock = (broker.state/'service.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    broker.recover()
    sock = Path(config['socket'])
    sock.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    s = sock.parent.stat()
    if s.st_uid != os.geteuid() or s.st_mode & 0o022:
        raise Rejected('socket directory must be operator-owned and not writable by agents')
    for path in (sock, sock.with_name(sock.name+'.operator')):
        if path.exists():
            if not stat.S_ISSOCK(path.lstat().st_mode) or path.lstat().st_uid != os.geteuid():
                raise Rejected('unsafe socket path')
            path.unlink()
    public = Server(str(sock), broker, False)
    private = Server(str(sock)+'.operator', broker, True)
    sock.chmod(0o666)
    Path(str(sock)+'.operator').chmod(0o600)
    threading.Thread(target=broker.work, daemon=True).start()
    threading.Thread(target=private.serve_forever, daemon=True).start()
    public.serve_forever()


if __name__ == '__main__':
    main()
